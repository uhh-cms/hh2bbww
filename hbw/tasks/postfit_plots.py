# coding: utf-8

"""
Tasks for postfit plots
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict

import law
import luigi
import order as od
from columnflow.tasks.framework.base import ConfigTask
from columnflow.tasks.framework.mixins import (
    InferenceModelMixin, MLModelsMixin, ProducersMixin, SelectorStepsMixin,
    CalibratorsMixin, DatasetsProcessesMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase1D, ProcessPlotSettingMixin,  # VariablesPlotSettingMixin,
)
from columnflow.tasks.framework.decorators import view_output_plots
from hbw.tasks.base import HBWTask

from columnflow.util import dev_sandbox, DotDict, maybe_import

# from hbw.inference.base import inf_proc

uproot = maybe_import("uproot")
hist = maybe_import("hist")


logger = law.logger.get_logger(__name__)


def load_hists_uproot(fit_diagnostics_path, fit_type):
    """ Helper to load histograms from a fit_diagnostics file """
    with uproot.open(fit_diagnostics_path) as tfile:
        if any("shapes_fit_s" in _k for _k in tfile.keys()):
            if fit_type == "postfit":
                fit_type = "fit_s"
            hists = get_hists_from_fit_diagnostics(tfile)[f"shapes_{fit_type}"]
        else:
            hists = get_hists_from_multidimfit(tfile)[f"{fit_type}"]

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

        if "data" in process or "total" in process:
            continue
        else:
            # transform TH1F to hist
            h_in = h_in.to_hist()

        # set the histogram in a deep dictionary
        hists = law.util.merge_dicts(hists, DotDict.wrap({fit: {channel: {process: h_in}}}), deep=True)
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

        if "data" in process or "Total" in process:
            continue
            # transform TH1F to hist
        else:
            h_in = h_in.to_hist()

        # set the histogram in a deep dictionary
        hists = law.util.merge_dicts(hists, DotDict.wrap({fit: {channel: {process: h_in}}}), deep=True)

    return hists


# imports regarding plot function
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")

from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import (
    prepare_stack_plot_config,
    prepare_style_config,
    apply_process_settings,
)


def plot_postfit_shapes(
    hists: OrderedDict[od.Process, hist.Hist],
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool | None = False,
    yscale: str | None = "",
    hide_errors: bool | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> tuple(plt.Figure, tuple(plt.Axes)):
    variable_inst = law.util.make_tuple(variable_insts)[0]
    hists = apply_process_settings(hists, process_settings)

    plot_config = prepare_stack_plot_config(
        hists,
        shape_norm=shape_norm,
        hide_errors=hide_errors,
    )

    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )

    # since we are rebinning, the xlim should be defined based on the histograms itself
    bin_edges = list(hists.values())[0].axes[0].edges
    default_style_config["ax_cfg"]["xlim"] = (bin_edges[0], bin_edges[-1])

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)
    if shape_norm:
        style_config["ax_cfg"]["ylabel"] = r"$\Delta N/N$"

    return plot_all(plot_config, style_config, **kwargs)


class PlotPostfitShapes(
    # NOTE: mixins might be wrong and could (should?) be extended to MultiConfigTask
    HBWTask,
    PlotBase1D,
    ProcessPlotSettingMixin,
    DatasetsProcessesMixin,
    # to correctly setup our InferenceModel, we need all these mixins, but hopefully, all these
    # parameters are automatically resolved correctly
    InferenceModelMixin,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    ConfigTask,
):
    """
    Task that creates Postfit shape plots based on a fit_diagnostics file.

    Work in Progress!
    TODO:
    - include data
    - include correct uncertainty bands
    - pass correct binning information
    """

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))
    datasets = None

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
    def prepare_hist_map(self, hist_processes: set, process_insts: list) -> defaultdict(list):

        hist_map = defaultdict(list)

        for proc_key in hist_processes:
            proc_inst = None
            # try getting the config process via InferenceModel
            # NOTE: Only the first category is taken to get the config process instance,
            # assuming they are the same for all categories
            channel = self.inference_model_inst.get_categories_with_process(proc_key)[0]
            has_category = self.inference_model_inst.has_category(channel)
            if has_category:
                inference_process = self.inference_model_inst.get_process(proc_key, channel)
                proc_inst = self.config_inst.get_process(inference_process.config_process)
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
                    logger.warning(
                        f"{proc_key} was assigned to ({','.join([p.name for p in plot_proc])})",
                        f" but {plot_proc[0].name} was chosen",
                    )
                elif len(plot_proc) == 0:
                    logger.warning(f"{proc_key} in root file, but won't be plotted.")
                    continue
                plot_proc = plot_proc[0]

                if plot_proc not in hist_map:
                    hist_map[plot_proc] = [proc_key]
                else:
                    hist_map[plot_proc].append(proc_key)

        return hist_map

    @view_output_plots
    def run(self):
        logger.warning(
            f"Note! It is important that the requested inference_model {self.inference_model} "
            "is identical to the one that has been used to create the datacards",
        )

        outp = self.output()

        # Load all required histograms corresponding to the fit_type from the root input file
        all_hists = load_hists_uproot(self.fit_diagnostics_file, self.fit_type)

        # Get list of all process instances required for plotting
        process_insts = list(map(self.config_inst.get_process, self.processes))

        # map processes in root shape to corresponding process instance used for plotting
        hist_processes = {key for _, h_in in all_hists.items() for key in h_in.keys()}
        hist_map = self.prepare_hist_map(hist_processes, process_insts)

        # Plot Pre/Postfit plot for each channel
        for channel, h_in in all_hists.items():

            # Check for coherence between inference and pre/postfit categories
            has_category = self.inference_model_inst.has_category(channel)
            if not has_category:
                logger.warning(f"Category {channel} is not part of the inference model {self.inference_model}")
                continue

            # Create Histograms
            hists = defaultdict(OrderedDict)
            for proc in hist_map:
                plot_proc = proc.copy()  # NOTE: copy produced, so actual process is not modified by process settings
                if hist_map[proc][0] not in h_in:
                    logger.warning(f"No histograms for {proc.name} found in {channel}.")
                    continue
                hists[plot_proc] = h_in[hist_map[proc][0]]

                for p in hist_map[proc][1:]:
                    if p in h_in.keys():
                        hists[plot_proc] += h_in[p]

            if has_category:
                inference_category = self.inference_model_inst.get_category(channel)
                config_category = self.config_inst.get_category(inference_category.config_category)
                variable_inst = self.config_inst.get_variable(inference_category.config_variable)
            else:
                # default to dummy Category and Variable
                config_category = od.Category(channel, id=1)
                variable_inst = od.Variable("dummy")

            # call the plot function
            h = hists.copy()  # NOTE: copy produced, so actual process is not modified by process settings
            fig, _ = self.call_plot_func(
                self.plot_function,
                hists=h,
                config_inst=self.config_inst,
                category_inst=config_category,
                variable_insts=variable_inst,
                **self.get_plot_parameters(),
            )

            outp["plots"].child(f"{channel}_{self.fit_type}.pdf", type="f").dump(fig, formatter="mpl")
