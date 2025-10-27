# coding: utf-8

"""
Tasks to produce yield tables.
Taken from columnflow and customized.
"""

import math
from collections import defaultdict

import law
import luigi
import order as od
from scinum import Number

from columnflow.tasks.framework.base import Requirements
# from columnflow.tasks.yields import _CreateYieldTable
# from columnflow.tasks.framework.mixins import MLModelsMixin
from hbw.tasks.histograms import HistogramsUserSingleShiftBase
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.histograms import MergeHistograms
from columnflow.util import dev_sandbox, try_int


from hbw.tasks.base import HBWTask

logger = law.logger.get_logger(__name__)


class CustomCreateYieldTable(
    HBWTask,
    HistogramsUserSingleShiftBase,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    # TODO: order of categories is not always maintained, see: https://github.com/columnflow/columnflow/issues/717
    # flag that does not yet exist :)
    categories_order_sensitive = True
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    variables = HistogramsUserSingleShiftBase.variables.copy(
        default=("mli_mbb",),
        description="Variable to use for the yields table. Should be a single variable",
        add_default_to_description=True,
    )

    table_format = luigi.Parameter(
        default="fancy_grid",
        significant=False,
        description="format of the yield table; accepts all formats of the tabulate package"
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
    ratio_modes = law.CSVParameter(
        default=("ratio", "subtract"),
        significant=False,
        description="Mode for the ratio calculation; "
        "choices: 'ratio' (default) or 'subtract'; "
        "if 'ratio', the ratio of the two processes is calculated, "
        "if 'subtract', (p[0] - p[1] + p[2]) / p[2] is calculated, ",
    )

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeHistograms=MergeHistograms,
    )

    # dummy branch map
    def create_branch_map(self):
        return [0]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # datasets/processes instances
        self.process_insts = {
            config_inst: list(map(config_inst.get_process, processes))
            for config_inst, processes in zip(self.config_insts, self.processes)
        }
        self.dataset_insts = {
            config_inst: list(map(config_inst.get_dataset, datasets))
            for config_inst, datasets in zip(self.config_insts, self.datasets)
        }

    @classmethod
    def resolve_param_values(cls, params):
        params = super().resolve_param_values(params)

        if len(params.get("variables")) != 1:
            raise ValueError("The yields table requires exactly one variable to be specified.")

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

    @property
    def category_insts(self):
        """
        Returns the category instances for the first config.
        """
        return list(map(self.config_insts[0].get_category, self.categories))

    @property
    def yields_variable(self):
        """
        Returns the variable to be used for the yields table.
        This is the first variable in the variables list.
        """
        return self.variables[0]

    def prepare_yields(self):
        inputs = self.input()
        import hist
        variable = self.variables[0]
        yields = {}
        processes = defaultdict(list)
        if len(set(self.processes)) != 1:
            raise NotImplementedError("The yields table task currently assumes same processes for all configs.")

        yields_hist = hist.Hist.new.StrCat(
            [], name="process", growth=True,
        ).StrCat(
            [], name="category", growth=True,
        ).Weight()

        for process in self.processes[0]:
            for category in self.categories:
                yields_hist.fill(process, category, weight=0)

        for i, config_inst in enumerate(self.config_insts):
            yields[config_inst.name] = defaultdict(list)
            # processes = []
            hist_per_config = None
            sub_processes = self.processes[i]
            for dataset in self.datasets[i]:
                # sum over all histograms of the same variable and config
                if hist_per_config is None:
                    hist_per_config = self.load_histogram(inputs, config_inst, dataset, variable)
                else:
                    hist_per_config += self.load_histogram(inputs, config_inst, dataset, variable)

            # slice histogram per config according to the sub_processes and categories
            hist_per_config = self.slice_histogram(
                histogram=hist_per_config,
                config_inst=config_inst,
                processes=sub_processes,
                categories=self.categories,
                shifts=self.shift,
            )
            hist_per_config = hist_per_config[{"shift": sum, self.yields_variable: sum}]

            for process in sub_processes:
                processes[config_inst.name].append(process)
                for category in self.categories:
                    h_cat = self.slice_histogram(
                        histogram=hist_per_config,
                        config_inst=config_inst,
                        processes=[process],
                        categories=[category],
                    )
                    yields_hist[{"process": process, "category": category}] += h_cat.sum()

        yields = defaultdict(list)
        processes = []

        # use process and category insts from 1st config
        for process_inst in self.process_insts[self.config_insts[0]]:
            if process_inst.name not in yields_hist.axes["process"]:
                logger.warning(f"Process {process_inst.name} not found in histogram axes, skipping.")
                continue
            processes.append(process_inst)
            for category_inst in self.category_insts:
                if category_inst.name not in yields_hist.axes["category"]:
                    logger.warning(f"Category {category_inst.name} not found in histogram axes, skipping.")
                    continue

                # get the histogram for the process and category
                h_cat = yields_hist[{"process": process_inst.name, "category": category_inst.name}]

                # create a Number object for the yield
                value = Number(h_cat.value)
                if not self.skip_uncertainties:
                    # set a unique uncertainty name for correct propagation below
                    value.set_uncertainty(
                        f"mcstat_{process_inst.name}_{category_inst.name}",
                        math.sqrt(h_cat.variance),
                    )
                yields[category_inst].append(value)
        return yields, processes

    def apply_ratio(self, yields, processes):
        """
        Function that applies the ratio calculations to the yields and processes.
        It checks if the requested processes for the ratio are available and applies the ratio calculations.
        If the processes are not available, it logs a warning and skips the ratio calculation.
        It adds the ratio processes to the list of processes and returns the updated yields and processes.
        If the ratio is not requested, it simply returns the yields and processes as they are.
        """
        config_inst = self.config_insts[0]  # use the first config instance to get process instances
        if self.ratio:
            missing_for_ratio = set(self.ratio[:2]) - set(p.name for p in processes)
            if missing_for_ratio:
                logger.warning(f"Cannot do ratio, missing requested processes: {', '.join(missing_for_ratio)}")
                return yields, processes

            # default ratio
            if len(self.ratio) >= 2 and "ratio" in self.ratio_modes:
                processes.append(od.Process("Ratio", id=-9870, label=f"{self.ratio[0]} / {self.ratio[1]}"))
                num_idx, den_idxs = [processes.index(config_inst.get_process(_p)) for _p in self.ratio[:2]]
                for category_inst in self.category_insts:
                    num = yields[category_inst][num_idx]
                    den = yields[category_inst][den_idxs]
                    yields[category_inst].append(num / den)

            if len(self.ratio) >= 3 and "subtract" in self.ratio_modes:
                num_idx = processes.index(config_inst.get_process(self.ratio[0]))
                subtract_idx = processes.index(config_inst.get_process(self.ratio[1]))

                for i, estimate_proc in enumerate(self.ratio[2:]):
                    estimate_idx = processes.index(config_inst.get_process(estimate_proc))

                    # subtract the second process from the first one
                    processes.append(od.Process(
                        f"Ratio_diff_{estimate_proc}",
                        id=-9871 + i,
                        label=f"({self.ratio[0]} - {self.ratio[1]} + {estimate_proc}) / {estimate_proc}",
                    ))
                    for category_inst in self.category_insts:
                        num = yields[category_inst][num_idx]
                        subtract = yields[category_inst][subtract_idx]
                        estimate = yields[category_inst][estimate_idx]
                        yields[category_inst].append((num - subtract + estimate) / estimate)

        return yields, processes

    def prepare_yields_str(self, yields, processes):
        """
        Function that prepares the yields for the table output.
        It normalizes the yields according to the specified normalization method and formats them into strings.
        It returns a dictionary of raw yields and a dictionary of formatted yields strings.
        """
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
                for i in range(len(yields[self.category_insts[0]]))
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
                # cat_label = category.name.replace("__", " ")
                cat_label = category.label.replace("\n", ", ")
                # cat_label = f"\\labelfunc{{{category.name}}}"
                yields_str[cat_label].append(yield_str)

        return raw_yields, yields_str

    def make_table(self, raw_yields, yields_str):
        from tabulate import tabulate
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

        outputs = self.output()
        outputs["table"].dump(yield_table, formatter="text")
        outputs["yields"].dump(raw_yields, formatter="json")

        outputs["csv"].touch()
        with open(outputs["csv"].abspath, "w", newline="") as csvfile:
            import csv
            writer = csv.writer(csvfile)
            writer.writerows(data)

    @law.decorator.notify
    @law.decorator.log
    def run(self):
        with self.publish_step(f"Creating yields for processes {self.processes}, categories {self.categories}"):
            yields, processes = self.prepare_yields()
            yields, processes = self.apply_ratio(yields, processes)
            raw_yields, yields_str = self.prepare_yields_str(yields, processes)
            self.make_table(raw_yields, yields_str)
