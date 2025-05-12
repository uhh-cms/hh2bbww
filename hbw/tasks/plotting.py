# coding: utf-8

"""
Custom plotting tasks
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict

import law
import order as od

from columnflow.tasks.framework.base import Requirements, ShiftTask, TaskShifts
from columnflow.tasks.framework.mixins import (
    CalibratorClassesMixin, SelectorClassMixin, ProducerClassesMixin,
    # CalibratorsMixin, SelectorMixin, ProducersMixin,
    MLModelsMixin,
    CategoriesMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase1D, ProcessPlotSettingMixin, VariablePlotSettingMixin,
    # PlotShiftMixin,
)
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.histograms import MergeHistograms
from columnflow.util import DotDict, dev_sandbox, maybe_import

from hbw.tasks.base import HBWTask


logger = law.logger.get_logger(__name__)


# imports for plot function

hist = maybe_import("hist")
plt = maybe_import("matplotlib.pyplot")


from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import (
    prepare_style_config,
    remove_residual_axis,
    apply_variable_settings,
    # apply_process_settings,
    # apply_density_to_hists,
)


def plot_multi_hist_producer(
    hists: dict[str, OrderedDict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool | None = False,
    yscale: str | None = "log",
    hide_errors: bool | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:

    variable_inst = variable_insts[0]

    # take processes from the first weight producer (they should always be the same)
    processes = list(list(hists.values())[0].keys())

    # merge over processes
    for hist_producer, w_hists in hists.items():
        hists[hist_producer] = sum(w_hists.values())

    remove_residual_axis(hists, "shift")
    hists = apply_variable_settings(hists, variable_insts, variable_settings)

    plot_config = OrderedDict()

    # add hists
    for hist_producer, h in hists.items():
        norm = sum(h.values()) if shape_norm else 1
        plot_config[hist_producer] = plot_cfg = {
            "method": "draw_hist",
            "hist": h,
            "kwargs": {
                "norm": norm,
                "label": hist_producer,
            },
            "ratio_kwargs": {
                "norm": hists[list(hists.keys())[0]].values(),
                "yerr": None,
            },
        }
        if hide_errors:
            for key in ("kwargs", "ratio_kwargs"):
                if key in plot_cfg:
                    plot_cfg[key]["yerr"] = None

    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    default_style_config["rax_cfg"]["ylabel"] = f"Ratio to {list(hists.keys())[0]}"

    # set process label as legend title
    process_label = processes[0].label if len(processes) == 1 else "Processes"
    default_style_config["legend_cfg"]["title"] = process_label

    return plot_all(plot_config, default_style_config, **kwargs)


class PlotVariablesMultiHistProducer(
    HBWTask,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ProducerClassesMixin,
    MLModelsMixin,
    CategoriesMixin,
    ProcessPlotSettingMixin,
    VariablePlotSettingMixin,
    # HistHookMixin,
    ShiftTask,
    PlotBase1D,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    # use the MergeHistograms task to trigger upstream TaskArrayFunction initialization
    single_config = True
    resolution_task_cls = MergeHistograms
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    hist_producers = law.CSVParameter(
        default=(),
        description="Weight producers to use for plotting",
    )

    plot_function = PlotBase.plot_function.copy(
        default="hbw.tasks.plotting.plot_multi_hist_producer",
        add_default_to_description=True,
    )

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeHistograms=MergeHistograms,
    )

    @classmethod
    def build_taf_insts(cls, params, shifts: TaskShifts | None = None):
        # TODO: HistProducersMixin
        if not cls.resolution_task_cls:
            raise ValueError(f"resolution_task_cls must be set for multi-config task {cls.task_family}")

        if shifts is None:
            shifts = TaskShifts()
        # we loop over all configs/datasets, but return initial params
        for i, config_inst in enumerate(params["config_insts"]):
            if cls.has_single_config():
                datasets = params["datasets"]
            else:
                datasets = params["datasets"][i]

            for hist_producer in params["hist_producers"]:
                for dataset in datasets:
                    # NOTE: we need to copy here, because otherwise taf inits will only be triggered once
                    _params = params.copy()
                    _params["config_inst"] = config_inst
                    _params["config"] = config_inst.name
                    _params["dataset"] = dataset
                    _params["hist_producer"] = hist_producer
                    logger.warning(f"building taf insts for {hist_producer} {config_inst.name}, {dataset}")
                    _params = cls.resolution_task_cls.build_taf_insts(_params, shifts)
                    cls.resolution_task_cls.get_known_shifts(_params, shifts)

        params["known_shifts"] = shifts

        return params

    def requires(self):
        return {
            hist_producer: {
                d: self.reqs.MergeHistograms.req(
                    self,
                    dataset=d,
                    branch=-1,
                    hist_producer=hist_producer,
                    _exclude={"branches"},
                    _prefer_cli={"variables"},
                )
                for d in self.datasets
            }
            for hist_producer in self.hist_producers
        }

    def output(self):
        b = self.branch_data
        return {"plots": [
            self.target(name)
            for name in self.get_plot_names(f"plot__proc_{self.processes_repr}__cat_{b.category}__var_{b.variable}")
        ]}

    def get_plot_shifts(self):
        return [self.global_shift_inst]

    @property
    def hist_producers_repr(self):
        return "_".join(self.hist_producers)

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "plot", f"datasets_{self.datasets_repr}")
        parts.insert_before("version", "weights", f"weights_{self.hist_producers_repr}")
        return parts

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name, "variable": var_name})
            for cat_name in sorted(self.categories)
            for var_name in sorted(self.variables)
        ]

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["merged_hists"] = self.requires_from_branch()

        return reqs

    @law.decorator.log
    @view_output_plots
    def run(self):
        import hist

        # get the shifts to extract and plot
        plot_shifts = law.util.make_list(self.get_plot_shifts())

        # prepare config objects
        variable_tuple = self.variable_tuples[self.branch_data.variable]
        variable_insts = [
            self.config_inst.get_variable(var_name)
            for var_name in variable_tuple
        ]
        category_inst = self.config_inst.get_category(self.branch_data.category)
        leaf_category_insts = category_inst.get_leaf_categories() or [category_inst]
        process_insts = list(map(self.config_inst.get_process, self.processes))
        sub_process_insts = {
            proc: [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        }

        # histogram data per process
        hists = defaultdict(OrderedDict)

        # NOTE: loading histograms as implemented here might not be consistent anymore with
        # how it's done in PlotVariables1D (important when datasets/shifts differ per config)
        with self.publish_step(f"plotting {self.branch_data.variable} in {category_inst.name}"):
            for hist_producer, inputs in self.input().items():
                for dataset, inp in inputs.items():
                    dataset_inst = self.config_inst.get_dataset(dataset)
                    h_in = inp["collection"][0]["hists"].targets[self.branch_data.variable].load(formatter="pickle")

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
                            "category": [
                                hist.loc(c.name)
                                for c in leaf_category_insts
                                if c.name in h.axes["category"]
                            ],
                            "shift": [
                                hist.loc(s.name)
                                for s in plot_shifts
                                if s.name in h.axes["shift"]
                            ],
                        }]

                        # axis reductions
                        h = h[{"process": sum, "category": sum}]

                        # add the histogram
                        if hist_producer in hists and process_inst in hists[hist_producer]:
                            hists[hist_producer][process_inst] = hists[hist_producer][process_inst] + h
                        else:
                            hists[hist_producer][process_inst] = h

                # there should be hists to plot
                if not hists:
                    raise Exception(
                        "no histograms found to plot; possible reasons:\n" +
                        "  - requested variable requires columns that were missing during histogramming\n" +
                        "  - selected --processes did not match any value on the process axis of the input histogram",
                    )

                # sort hists by process order
                hists[hist_producer] = OrderedDict(
                    (process_inst.copy_shallow(), hists[hist_producer][process_inst])
                    for process_inst in sorted(hists[hist_producer], key=process_insts.index)
                )

            hists = dict(hists)

            # call the plot function
            fig, _ = self.call_plot_func(
                self.plot_function,
                hists=hists,
                config_inst=self.config_inst,
                category_inst=category_inst.copy_shallow(),
                variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
                **self.get_plot_parameters(),
            )

            # save the plot
            for outp in self.output()["plots"]:
                outp.dump(fig, formatter="mpl")
