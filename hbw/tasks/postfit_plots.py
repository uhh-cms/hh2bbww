# coding: utf-8

"""
Tasks for postfit plots
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict

import law
import luigi
import order as od
from columnflow.tasks.framework.base import ShiftTask
from columnflow.tasks.framework.mixins import (
    CalibratorClassesMixin, SelectorClassMixin, ReducerClassMixin, ProducerClassesMixin,
    HistProducerClassMixin,
    InferenceModelMixin, MLModelsMixin, DatasetsProcessesMixin,
)
from columnflow.tasks.histograms import MergeHistograms
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase1D, ProcessPlotSettingMixin,
)
from columnflow.tasks.framework.decorators import view_output_plots
from hbw.tasks.base import HBWTask

from columnflow.util import dev_sandbox, DotDict, maybe_import
from columnflow.plotting.plot_util import get_position

logger = law.logger.get_logger(__name__)


def load_hists_uproot(fit_diagnostics_path, fit_type):
    import uproot
    """ Helper to load histograms from a fit_diagnostics file """
    with uproot.open(fit_diagnostics_path) as tfile:
        if any("shapes_fit_s" in _k for _k in tfile.keys()):
            if fit_type == "postfit":
                fit_type = "fit_s"
            hists = get_hists_from_fit_diagnostics(tfile)[f"shapes_{fit_type}"]
        else:
            # TODO: add switch for multidimfit vs merged multidimfit
            hists = get_hists_from_merged_multidimfit(tfile)[f"{fit_type}"]
            # hists = get_hists_from_multidimfit(tfile)[f"{fit_type}"]

    return hists


def get_hists_from_fit_diagnostics(tfile):
    """ Helper function to load histograms from root file created by FitDiagnostics """

    # prepare output dict
    hists = DotDict()

    keys = [key.split("/") for key in tfile.keys()]
    for key in keys:
        if len(key) != 3:
            continue

        # get the histogram from the tfile
        h_in = tfile["/".join(key)]

        # unpack key
        fit, channel, process = key
        process = process.split(";")[0]

        if "total" in process:
            continue
        else:
            # transform TH1F to hist
            h_in = h_in.to_hist()

        # set the histogram in a deep dictionary
        hists = law.util.merge_dicts(hists, DotDict.wrap({fit: {channel: {process: h_in}}}), deep=True)
    return hists


def get_hists_from_merged_multidimfit(tfile):
    """ Helper function to load histograms from root file created by MultiDimFit """
    # prepare output dict
    hists = DotDict()
    keys = [key.split("/") for key in tfile.keys()]

    variables = set()
    for key in keys:
        if len(key) != 3:
            continue
        # get the histogram from the tfile
        h_in = tfile["/".join(key)]

        # unpack key
        variable, fit_and_channel, process = key
        variables.add(variable)

        fit = fit_and_channel.split("_")[-1]
        fit = fit.replace("_", "")
        channel = fit_and_channel.replace(f"_{fit}", "")
        process = process.split(";")[0]

        if "Total" in process and process != "TotalBkg":
            continue
        else:
            h_in = h_in.to_hist()

        # set the histogram in a deep dictionary
        hists = law.util.merge_dicts(
            hists,
            DotDict.wrap({fit: {variable: {channel: {process: h_in}}}}),
            deep=True,
        )

    return hists


def get_hists_from_multidimfit(tfile):
    """ Helper function to load histograms from root file created by MultiDimFit """
    # prepare output dict
    hists = DotDict()
    keys = [key.split("/") for key in tfile.keys()]
    for key in keys:
        if len(key) != 2:
            continue
        # get the histogram from the tfile
        h_in = tfile["/".join(key)]

        # unpack key
        fit_and_channel, process = key
        fit = fit_and_channel.split("_")[-1]
        fit = fit.replace("_", "")
        channel = fit_and_channel.replace(f"_{fit}", "")
        process = process.split(";")[0]

        if "Total" in process:
            continue
        else:
            h_in = h_in.to_hist()

        # set the histogram in a deep dictionary
        hists = law.util.merge_dicts(hists, DotDict.wrap({fit: {channel: {process: h_in}}}), deep=True)

    return hists


from columnflow.types import TYPE_CHECKING
if TYPE_CHECKING:
    plt = maybe_import("matplotlib.pyplot")
    hist = maybe_import("hist")

from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import (
    prepare_stack_plot_config,
    prepare_style_config,
    apply_process_settings,
    apply_process_scaling,
    remove_residual_axis,
    apply_density,
)


def plot_postfit_shapes(
    hists: OrderedDict[od.Process, hist.Hist],
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    shift_insts: list[od.Shift],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool | None = False,
    yscale: str | None = "",
    hide_errors: bool | None = None,
    hide_signal_errors: bool | None = False,
    lumi: float | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    total_bkg: hist.Hist | None = None,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes]]:
    variable_inst = law.util.make_tuple(variable_insts)[0]
    hists, process_style_config = apply_process_settings(hists, process_settings)
    # process scaling
    hists = apply_process_scaling(hists)

    # density scaling per bin
    if density:
        hists = apply_density(hists, density)
        total_bkg = apply_density({"bkg": total_bkg}, density)["bkg"]

    if len(shift_insts) == 1:
        # when there is exactly one shift bin, we can remove the shift axis
        remove_residual_axis(hists, "shift", select_value=shift_insts[0].name)
    else:
        # remove shift axis of histograms that are not to be stacked
        unstacked_hists = {
            proc_inst: h
            for proc_inst, h in hists.items()
            if proc_inst.is_mc and getattr(proc_inst, "unstack", False)
        }
        hists |= remove_residual_axis(unstacked_hists, "shift", select_value="nominal")

    plot_config = prepare_stack_plot_config(
        hists,
        shape_norm=shape_norm,
        hide_errors=hide_errors,
        shift_insts=shift_insts,
        density=density,
        **kwargs,
    )
    if total_bkg:
        if any((diff := abs(1 - plot_config["mc_stat_unc"]["hist"].values() / total_bkg.values())) > 1e-5):
            raise ValueError(
                "The provided total_bkg histogram (used for variances) "
                f"does not match the sum of the background histograms. Difference: {diff}",
            )
        plot_config["mc_stat_unc"]["hist"] = total_bkg
        plot_config["mc_stat_unc"]["ratio_kwargs"]["norm"] = total_bkg.values()
    try:
        plot_config["mc_stat_unc"]["kwargs"]["label"] = "Syst. unc."
        plot_config["mc_stat_unc"]["kwargs"]["hatch"] = "\\\\\\"
    except KeyError:
        logger.warning("Tried to update label of mc_stat_unc, but it does not exist in the plot_config")

    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    if hide_signal_errors:
        for key in plot_config.keys():
            if "line_" in key:
                # remove line yerr
                plot_config[key]["kwargs"]["yerr"] = 0

    # since we are rebinning, the xlim should be defined based on the histograms itself
    bin_edges = list(hists.values())[0].axes[0].edges
    default_style_config["ax_cfg"]["xlim"] = (bin_edges[0], bin_edges[-1])

    style_config = law.util.merge_dicts(
        default_style_config,
        process_style_config,
        style_config,
        deep=True,
    )
    if shape_norm:
        style_config["ax_cfg"]["ylabel"] = r"$\Delta N/N$"

    if lumi:
        style_config["cms_label_cfg"]["lumi"] = f"{lumi:.0f}"
    return plot_all(plot_config, style_config, **kwargs)


class PlotPostfitShapes(
    HBWTask,
    PlotBase1D,
    ShiftTask,
    ProcessPlotSettingMixin,
    DatasetsProcessesMixin,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ReducerClassMixin,
    ProducerClassesMixin,
    MLModelsMixin,
    HistProducerClassMixin,
    InferenceModelMixin,
    # HistHookMixin,
):
    """
    Task that creates Postfit shape plots based on a fit_diagnostics file.

    Work in Progress!
    TODO:
    - include data
    - include correct uncertainty bands
    - pass correct binning information
    """

    single_config = True
    resolution_task_cls = MergeHistograms

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))
    # datasets = None

    plot_function = PlotBase.plot_function.copy(
        default="hbw.tasks.postfit_plots.plot_postfit_shapes",
        add_default_to_description=True,
    )

    fit_diagnostics_file = luigi.Parameter(
        default=law.NO_STR,
        description="fit_diagnostics file that is used to load histograms",
    )

    prefit = luigi.BoolParameter(
        default=False,
        description="Whether to do prefit or postfit plots; defaults to False",
    )

    merged_only = luigi.BoolParameter(
        default=True,
        description="Whether to only plot the merged category; defaults to True",
    )

    @property
    def fit_type(self) -> str:
        if self.prefit:
            return "prefit"
        else:
            return "postfit"

    def requires(self):
        return {}

    def output(self):
        return {"plots": self.target(f"plots_{self.fit_type}", dir=True)}

    # map processes in root shape to corresponding process instance used for plotting
    def prepare_processes_map(self, hist_processes: set, process_insts: list) -> defaultdict(list):
        """
        Helper function to map processes from the datacards to the processes that were
        requested for plotting.
        :param hist_processes: set of process names that are present in the root file
        :param process_insts: list of process instances that are requested for plotting
        :return: dict mapping process instances to list of process names in the root file
        """

        processes_map = defaultdict(list)

        for proc_key in hist_processes:
            proc_inst = None
            # try getting the config process via InferenceModel
            # NOTE: Only the first category is taken to get the config process instance,
            # assuming they are the same for all categories
            channel = self.inference_model_inst.get_categories_with_process(proc_key)

            if not channel:  # sometimes channel can be an empty list
                has_category = False
            else:
                channel = channel[0]
                has_category = self.inference_model_inst.has_category(channel)

            if has_category and proc_key != "data_obs":
                # config_data = channel.config_data.get(self.config_inst.name)
                inference_process = self.inference_model_inst.get_process(proc_key, channel)
                proc_inst = self.config_inst.get_process(inference_process.config_data[self.config_inst.name].process)
            # Mao data_obs to data process
            elif proc_key == "data_obs":
                proc_inst = self.config_inst.get_process("data", default=None)
            else:
                # try getting proc inst directly via config
                proc_inst = self.config_inst.get_process(proc_key, default=None)

            # replace string keys with process instances
            # map HHinference processes to process instances for plotting
            if proc_inst:
                plot_proc = [
                    proc for proc in process_insts if proc.has_process(proc_inst) or proc.name == proc_inst.name
                ]
                if len(plot_proc) > 1:
                    plot_proc_names = [p.name for p in plot_proc]
                    if len(plot_proc) == 2 and "background" in plot_proc_names:
                        plot_proc = [p for p in plot_proc if p.name != "background"]
                    else:
                        logger.warning(
                            f"{proc_key} was assigned to ({','.join([p.name for p in plot_proc])})",
                            f" but {plot_proc[0].name} was chosen",
                        )
                elif len(plot_proc) == 0:
                    logger.info(f"{proc_key} in root file, but won't be plotted.")
                    continue
                plot_proc = plot_proc[0]
                processes_map[plot_proc].append(proc_key)

        return processes_map

    def get_shift_insts(self):
        return [self.config_inst.get_shift(self.shift)]

    from hbw.util import timeit_multiple

    def sort_categories(self, categories_set):
        """
        Sort processes based on:
        1. Primary: existence of substrings ggf, vbf, tt, st, dy, h
        2. Secondary: existence of substrings 1b, 2b, boosted
        """
        primary_order = ["_sig_ggf", "_sig_vbf", "_tt", "_st", "_dy", "_h", "_bkg"]
        secondary_order = ["1b", "2b", "boosted"]

        def sort_key(item):
            # Primary sort: find first matching substring from primary_order
            primary_idx = len(primary_order)  # default to end if no match
            for i, substring in enumerate(primary_order):
                if substring in item.lower():
                    primary_idx = i
                    break

            # Secondary sort: find first matching substring from secondary_order
            secondary_idx = len(secondary_order)  # default to end if no match
            for i, substring in enumerate(secondary_order):
                if substring in item.lower():
                    secondary_idx = i
                    break

            # Return tuple for multi-level sorting
            return (primary_idx, secondary_idx, item)  # item as tiebreaker for alphabetical

        return sorted(list(categories_set), key=sort_key)

    @timeit_multiple
    def create_merged_hist(self, all_hists) -> None:
        """
        Helper function to merge all categories per process into a single hist object
        """
        import boost_histogram
        import numpy as np
        import hist
        dummy_view = boost_histogram.view.WeightedSumView(0, dtype=[("value", "<f8"), ("variance", "<f8")])

        # TODO: at the moment we only consider processes that are present in all cards
        # all_processes = set.intersection(*[set(hists.keys()) for hists in all_hists.values()])
        all_processes = set.union(*[set(hists.keys()) for hists in all_hists.values()])

        bins_dict = {}
        categories_sorted = self.sort_categories(all_hists.keys())
        logger.info(f"Categories sorted for merging: {categories_sorted}")
        for category in categories_sorted:
            hist_dict = all_hists[category]
            axis = list(hist_dict.values())[0].axes[0]
            bins_dict[category] = {
                "count": len(axis),
                "edges": axis.edges,
            }

        # initialize histogram
        n_bins = sum([b["count"] for b in bins_dict.values()])
        h_out = {
            process: hist.Hist.new.Integer(0, n_bins, name="xaxis").Weight()
            for process in all_processes
        }
        view_dict = {process: dummy_view.copy() for process in all_processes}

        # merge histograms over categories per process
        for category in categories_sorted:
            n_bins_cat = bins_dict[category]["count"]
            hist_dict = all_hists[category]
            for process in all_processes:
                if process in hist_dict.keys():
                    view_dict[process] = np.concatenate(
                        (view_dict[process], hist_dict[process].view(flow=False)),
                    )
                else:
                    empty_view = np.array([(0, 0)] * n_bins_cat, dtype=[("value", "<f8"), ("variance", "<f8")])
                    view_dict[process] = np.concatenate(
                        (view_dict[process], empty_view),
                    )

        for process, h in h_out.items():
            h_out[process][...] = view_dict[process]

        all_hists["merged"] = h_out

        # extend bins_dict with some additional info for plotting
        ml_proc_bins = defaultdict(int)
        for cat_name, bins_info in bins_dict.items():
            ml_proc = cat_name.split("ml_")[-1].split("__")[0].replace("sig_", "HH").replace("dy_m10toinf", "DY").replace("h", "H").replace("bkg", "")  # noqa: E501
            ml_proc = {
                "HHggf": "gluon-gluon fusion (HH ggF)",
                "HHvbf": "vector boson fusion (HH VBF)",
                "tt": r"$t\bar{t}$",
                "st": "t",
            }.get(ml_proc, ml_proc)
            bins_dict[cat_name]["ml_proc"] = ml_proc
            ml_proc_bins[ml_proc] += bins_info["count"]
        for cat_name, bins_info in bins_dict.items():
            bins_info["ml_proc_count"] = ml_proc_bins[bins_info["ml_proc"]]

        return bins_dict

    @view_output_plots
    def run(self):
        logger.warning(
            f"Note! It is important that the requested inference_model {self.inference_model} "
            "is identical to the one that has been used to create the datacards",
        )
        # Load all required histograms corresponding to the fit_type from the root input file
        all_hists = load_hists_uproot(self.fit_diagnostics_file, self.fit_type)

        for variable, _all_hists in all_hists.items():
            self.build_plots(variable, _all_hists)

    def build_plots(self, variable, all_hists):
        outp = self.output()

        plot_parameters = self.get_plot_parameters()
        # Get list of all process instances required for plotting
        process_insts = list(map(self.config_inst.get_process, self.processes))
        # map processes in root shape to corresponding process instance used for plotting
        hist_processes = {key for _, h_in in all_hists.items() for key in h_in.keys()}
        processes_map = self.prepare_processes_map(hist_processes, process_insts)

        # make a combined histogram of all categories
        bins_dict = self.create_merged_hist(all_hists)

        # Plot Pre/Postfit plot for each channel
        for channel, h_in in all_hists.items():
            if self.merged_only and "merged" not in channel:
                continue
            channel = channel.replace("__2022_2023", "")
            # Check for coherence between inference and pre/postfit categories
            has_category = self.inference_model_inst.has_category(channel)

            if not has_category:
                logger.warning(f"Category {channel} is not part of the inference model {self.inference_model}")

            # Create Histograms
            total_bkg = h_in.pop("TotalBkg", None)
            hists = defaultdict(OrderedDict)
            for proc, sub_procs in processes_map.items():
                plot_proc = proc.copy()  # NOTE: copy produced, so actual process is not modified by process settings

                if not any(sub_proc in h_in.keys() for sub_proc in sub_procs):
                    logger.warning(f"No histograms for {proc.name} found in {channel}.")
                    continue
                hists[plot_proc] = sum([
                    h_in[sub_proc] for sub_proc in sub_procs
                    if sub_proc in h_in.keys()
                ])

            if has_category:
                inference_category = self.inference_model_inst.get_category(channel)
                config_data = inference_category.config_data.get(self.config_inst.name)
                config_category = self.config_inst.get_category(config_data.category)

                variable_inst = self.config_inst.get_variable(config_data.variable)
            else:
                # default to dummy Category and Variable
                config_category = self.config_inst.get_category(
                    channel,
                    default=od.Category(channel, id=1, label=""),
                )
                variable_inst = self.config_inst.get_variable(
                    variable,
                    default=od.Variable("dummy", aux={"x_min": None, "x_max": None}),
                )
            if self.fit_type not in config_category.label.lower():
                config_category.label = self.fit_type + "\n" + config_category.label

            # sort histograms
            hists = {
                proc_inst: hists[proc_inst]
                for proc_inst in sorted(hists.keys(), key=self.processes.index)
            }

            # call the plot function
            h = hists.copy()  # NOTE: copy produced, so actual process is not modified by process settings
            fig, axs = self.call_plot_func(
                self.plot_function,
                hists=h,
                config_inst=self.config_inst,
                category_inst=config_category,
                variable_insts=variable_inst,
                shift_insts=self.get_shift_insts(),
                total_bkg=total_bkg,
                **plot_parameters,
            )

            # some adjustments for the merged plot
            if channel == "cat_merged" or channel == "merged":
                line_pos = 0.71
                bins_count = 0
                axs[0].axhline(
                    get_position(*axs[0].get_ylim(), factor=line_pos, logscale=True),
                    color="grey", linewidth=2.5,
                )
                for cat_name, bins_info in bins_dict.items():

                    bjet_cat = None
                    if "1b" in cat_name:
                        bjet_cat = "1b"
                    elif "2b" in cat_name:
                        bjet_cat = "2b"
                    elif "boosted" in cat_name:
                        bjet_cat = "Boost"
                        # bjet_cat = "boosted"

                    annotate_kwargs = dict(
                        text=bjet_cat,
                        xy=(
                            bins_count + 0.5 * bins_info["count"],
                            get_position(*axs[0].get_ylim(), factor=line_pos - 0.02, logscale=True),
                        ),
                        xycoords="data",
                        fontsize=22,
                        horizontalalignment="center",
                        verticalalignment="top",
                        color="black",
                    )
                    if bins_info["ml_proc"] == "":
                        annotate_kwargs["rotation"] = 90
                        annotate_kwargs["xy"] = (
                            bins_count + 0.5 * bins_info["count"],
                            get_position(*axs[0].get_ylim(), factor=line_pos - 0.02, logscale=True),
                        )
                        annotate_kwargs["text"] = annotate_kwargs["text"] + "ed CR"
                        # annotate_kwargs["text"] = annotate_kwargs["text"] + " CR"
                    elif bins_info["count"] == 1:
                        annotate_kwargs["rotation"] = 90
                    axs[0].annotate(**annotate_kwargs)

                    # axs[0].annotate(
                    #     text=bjet_cat,
                    #     xy=(
                    #         bins_count + 0.5 * bins_info["count"],
                    #         get_position(*axs[0].get_ylim(), factor=line_pos - 0.02, logscale=True),
                    #     ),
                    #     xycoords="data",
                    #     fontsize=16,
                    #     horizontalalignment="center",
                    #     verticalalignment="top",
                    #     color="black",
                    #     rotation=45,
                    # )

                    do_full_sep = (
                        (bjet_cat == "1b") or ("boosted__ml_bkg" in cat_name)
                    )

                    if do_full_sep:
                        axs[0].annotate(
                            text=bins_info["ml_proc"],
                            xy=(
                                bins_count + 0.5 * (bins_info["ml_proc_count"]),  # TODO: position better
                                get_position(*axs[0].get_ylim(), factor=line_pos + 0.05, logscale=True),
                            ),
                            xycoords="data",
                            fontsize=24,
                            horizontalalignment="center",
                            verticalalignment="top",
                            color="black",
                        )

                        ymax = line_pos + 0.04
                        kwargs = {
                            "color": "grey",
                            "linewidth": 2.5,
                        }
                    else:
                        ymax = line_pos
                        kwargs = {
                            "color": "grey",
                            "linewidth": 2.5,
                            "linestyle": "--",
                        }

                    axs[0].axvline(bins_count, ymax=ymax, **kwargs)
                    axs[1].axvline(bins_count, **kwargs)
                    bins_count += bins_info["count"]

            outp["plots"].child(f"{variable_inst.name}_{channel}_{self.fit_type}.pdf", type="f").dump(
                fig,
                formatter="mpl",
            )

        self.publish_message(f"plots written to {outp['plots'].path}")
