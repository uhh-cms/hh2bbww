# coding: utf-8

"""
Tasks to produce yield tables.
Taken from columnflow and customized.
TODO: after merging MultiConfigPlotting, we should propagate the changes back to columnflow
"""

import math
from collections import defaultdict, OrderedDict

import law
import luigi
import order as od
from scinum import Number

from columnflow.tasks.framework.base import Requirements, ConfigTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorMixin, ProducersMixin,
    DatasetsProcessesMixin, CategoriesMixin, HistProducerMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.histograms import MergeHistograms
from columnflow.util import dev_sandbox, try_int

from hbw.tasks.base import HBWTask

logger = law.logger.get_logger(__name__)


class CustomCreateYieldTable(
    HBWTask,
    DatasetsProcessesMixin,
    CategoriesMixin,
    HistProducerMixin,
    ProducersMixin,
    # MLModelsMixin,
    SelectorMixin,
    CalibratorsMixin,
    ConfigTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    yields_variable = "mll"

    table_format = luigi.Parameter(
        default="fancy_grid",
        significant=False,
        description="format of the yield table; accepts all formats of the tabulate package; "
        "default: fancy_grid",
    )
    number_format = luigi.Parameter(
        default="pdg",
        significant=False,
        description="rounding format of each number in the yield table; accepts all formats "
        "understood by scinum.Number.str(), e.g. 'pdg', 'publication', '%.1f' or an integer "
        "(number of signficant digits); default: pdg",
    )
    skip_uncertainties = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, uncertainties are not displayed in the table; default: False",
    )
    normalize_yields = luigi.ChoiceParameter(
        choices=(law.NO_STR, "per_process", "per_category", "all"),
        default=law.NO_STR,
        significant=False,
        description="string parameter to define the normalization of the yields; "
        "choices: '', per_process, per_category, all; empty default",
    )
    output_suffix = luigi.Parameter(
        default=law.NO_STR,
        description="Adds a suffix to the output name of the yields table; empty default",
    )
    transpose = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Transpose the yield table; default: False",
    )
    ratio = law.CSVParameter(
        default=("data", "background"),
        significant=False,
        description="Ratio of two processes to be calculated and added to the table",
    )

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeHistograms=MergeHistograms,
    )

    # dummy branch map
    def create_branch_map(self):
        return [0]

    def requires(self):
        return {
            d: self.reqs.MergeHistograms.req(
                self,
                dataset=d,
                variables=(self.yields_variable,),
                _prefer_cli={"variables"},
            )
            for d in self.datasets
        }

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["merged_hists"] = [
            self.reqs.MergeHistograms.req(
                self,
                dataset=d,
                variables=(self.yields_variable,),
                _exclude={"branches"},
            )
            for d in self.datasets
        ]

        return reqs

    @classmethod
    def resolve_param_values(cls, params):
        params = super().resolve_param_values(params)

        if "number_format" in params and try_int(params["number_format"]):
            # convert 'number_format' in integer if possible
            params["number_format"] = int(params["number_format"])

        return params

    def output(self):
        suffix = ""
        if self.output_suffix and self.output_suffix != law.NO_STR:
            suffix = f"__{self.output_suffix}"

        return {
            "table": self.target(f"table__proc_{self.processes_repr}__cat_{self.categories_repr}{suffix}.txt"),
            "csv": self.target(f"table__proc_{self.processes_repr}__cat_{self.categories_repr}{suffix}.csv"),
            "yields": self.target(f"yields__proc_{self.processes_repr}__cat_{self.categories_repr}{suffix}.json"),
        }

    @law.decorator.notify
    @law.decorator.log
    def run(self):
        import hist
        from tabulate import tabulate

        inputs = self.input()
        outputs = self.output()

        category_insts = list(map(self.config_inst.get_category, sorted(self.categories)))
        process_insts = list(map(self.config_inst.get_process, self.processes))
        sub_process_insts = {
            proc: [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        }

        # histogram data per process
        hists = {}

        with self.publish_step(f"Creating yields for processes {self.processes}, categories {self.categories}"):
            for dataset, inp in inputs.items():
                dataset_inst = self.config_inst.get_dataset(dataset)

                # load the histogram of the variable named self.yields_variable
                h_in = inp["hists"][self.yields_variable].load(formatter="pickle")

                # loop and extract one histogram per process
                for process_inst in process_insts:
                    # skip when the dataset is already known to not contain any sub process
                    if not any(map(dataset_inst.has_process, sub_process_insts[process_inst])):
                        continue

                    # work on a copy
                    h = h_in.copy()

                    # axis selections
                    h = h[{
                        "process": [
                            hist.loc(p.id)
                            for p in sub_process_insts[process_inst]
                            if p.id in h.axes["process"]
                        ],
                    }]

                    # axis reductions
                    h = h[{"process": sum, "shift": sum, self.yields_variable: sum}]

                    # add the histogram
                    if process_inst in hists:
                        hists[process_inst] += h
                    else:
                        hists[process_inst] = h

            # there should be hists to plot
            if not hists:
                raise Exception("no histograms found to plot")

            # sort hists by process order
            hists = OrderedDict(
                (process_inst, hists[process_inst])
                for process_inst in sorted(hists, key=process_insts.index)
            )

            yields, processes = defaultdict(list), []

            # read out yields per category and per process
            for process_inst, h in hists.items():
                processes.append(process_inst)

                # TODO: ratio

                for category_inst in category_insts:
                    leaf_category_insts = category_inst.get_leaf_categories() or [category_inst]

                    h_cat = h[{"category": [
                        hist.loc(c.name)
                        for c in leaf_category_insts
                        if c.name in h.axes["category"]
                    ]}]
                    h_cat = h_cat[{"category": sum}]

                    value = Number(h_cat.value)
                    if not self.skip_uncertainties:
                        # set a unique uncertainty name for correct propagation below
                        value.set_uncertainty(
                            f"mcstat_{process_inst.name}_{category_inst.name}",
                            math.sqrt(h_cat.variance),
                        )
                    yields[category_inst].append(value)

            if self.ratio:
                missing_for_ratio = set(self.ratio[:2]) - set(p.name for p in processes)
                if missing_for_ratio:
                    logger.warning(f"Cannot do ratio, missing requested processes: {', '.join(missing_for_ratio)}")
                else:
                    processes.append(od.Process("Ratio", id=-9871, label=f"{self.ratio[0]} / {self.ratio[1]}"))
                    num_idx, den_idxs = [processes.index(self.config_inst.get_process(_p)) for _p in self.ratio]
                    for category_inst in category_insts:
                        num = yields[category_inst][num_idx]
                        den = yields[category_inst][den_idxs]
                        yields[category_inst].append(num / den)

            # obtain normalizaton factors
            norm_factors = 1
            if self.normalize_yields == "all":
                norm_factors = sum(
                    sum(category_yields)
                    for category_yields in yields.values()
                )
            elif self.normalize_yields == "per_process":
                norm_factors = [
                    sum(yields[category][i] for category in yields.keys())
                    for i in range(len(yields[category_insts[0]]))
                ]
            elif self.normalize_yields == "per_category":
                norm_factors = {
                    category: sum(category_yields)
                    for category, category_yields in yields.items()
                }

            # initialize dicts
            main_label = "Category" if self.transpose else "Process"
            yields_str = defaultdict(list, {main_label: [proc.label for proc in processes]})
            raw_yields = defaultdict(dict, {})

            # apply normalization and format
            for category, category_yields in yields.items():
                for i, value in enumerate(category_yields):
                    # get correct norm factor per category and process
                    if self.normalize_yields == "per_process":
                        norm_factor = norm_factors[i]
                    elif self.normalize_yields == "per_category":
                        norm_factor = norm_factors[category]
                    else:
                        norm_factor = norm_factors

                    raw_yield = (value / norm_factor).nominal
                    raw_yields[category.name][processes[i].name] = raw_yield

                    # format yields into strings
                    yield_str = (value / norm_factor).str(
                        combine_uncs="all",
                        format=self.number_format,
                        style="latex" if "latex" in self.table_format else "plain",
                    )
                    if "latex" in self.table_format:
                        yield_str = f"${yield_str}$"
                    cat_label = category.name.replace("__", " ")
                    # cat_label = category.label
                    yields_str[cat_label].append(yield_str)

            # Transposing the table
            data = [list(yields_str.keys())] + list(zip(*yields_str.values()))
            if self.transpose:
                data = list(zip(*data))

            headers = data[0]

            # create, print and save the yield table
            yield_table = tabulate(data[1:], headers=headers, tablefmt=self.table_format)

            with_grid = True
            if with_grid and self.table_format == "latex_raw":
                # identify line breaks and add hlines after every line break
                yield_table = yield_table.replace("\\\\", "\\\\ \\hline")
                # TODO: lll -> |l|l|l|, etc.
            self.publish_message(yield_table)

            outputs["table"].dump(yield_table, formatter="text")
            outputs["yields"].dump(raw_yields, formatter="json")

            outputs["csv"].touch()
            with open(outputs["csv"].abspath, "w", newline="") as csvfile:
                import csv
                writer = csv.writer(csvfile)
                writer.writerows(data)
