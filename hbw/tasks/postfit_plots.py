# coding: utf-8

"""
Tasks for postfit plots
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict

import law
import luigi
import order as od
import re
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


def reverse_inf_proc(proc):
    """
    Helper function that reverses the transformations done by inf_proc.
    """
    if proc.startswith("ggHH_"):
        # Adjust pattern to split the last part into two groups
        pattern = r"ggHH_kl_([mp\d]+)_kt_([mp\d]+)_([a-zA-Z\d]{3})([a-zA-Z\d]+)"
        replacement = r"hh_ggf_\3_\4_kl\1_kt\2"
        return re.sub(pattern, replacement, proc)
    elif proc.startswith("qqHH_"):
        # Adjust pattern to split the last part into two groups
        pattern = r"qqHH_CV_([mp\d]+)_C2V_([mp\d]+)_kl_([mp\d]+)_([a-zA-Z\d]{3})([a-zA-Z\d]+)"
        replacement = r"hh_vbf_\4_\5_kv\1_k2v\2_kl\3"
        return re.sub(pattern, replacement, proc)
    elif proc == "qqH":
        pattern = r"qqH"
        replacement = r"h_vbf"
        return re.sub(pattern, replacement, proc)
    elif proc == "ggH":
        pattern = r"ggH"
        replacement = r"h_ggf"
        return re.sub(pattern, replacement, proc)
    elif proc == "ggZH":
        pattern = r"ggZH"
        replacement = r"zh_gg"
        return re.sub(pattern, replacement, proc)
    elif "H" in proc:
        proc = proc.lower()
        return proc
    else:
        # If the string doesn't match the patterns, return it unchanged
        return proc


def load_hists_uproot(fit_diagnostics_path, fit_type):
    """ Helper to load histograms from a fit_diagnostics file """
    with uproot.open(fit_diagnostics_path) as tfile:
        if any("shapes_fit_s" in _k for _k in tfile.keys()):
            if fit_type != "prefit":
                fit_type = "fit_s"
            hists = get_hists_from_fit_diagnostics(tfile)[f"shapes_{fit_type}"]
        else:
            # if fit_type != "prefit":
            #     fit_type = "postfit"
            hists = get_hists_from_multidimfit(tfile)[f"{fit_type}"]

    return hists


def get_hists_from_fit_diagnostics(tfile):
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
    """ Helper to load histograms from a fit_diagnostics file """
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
    prepare_plot_config,
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
    plot_config = prepare_plot_config(
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

    def requires(self):
        return {}

    def output(self):
        return {"plots": self.target("plots", dir=True)}

    @view_output_plots
    def run(self):
        logger.warning(
            f"Note! It is important that the requested inference_model {self.inference_model} "
            "is identical to the one that has been used to create the datacards",
        )

        outp = self.output()
        if self.prefit:
            fit_type = "prefit"
        else:
            fit_type = "postfit"

        all_hists = load_hists_uproot(self.fit_diagnostics_file, fit_type)
        process_insts = list(map(self.config_inst.get_process, self.processes))

        for channel, h_in in all_hists.items():
            has_category = self.inference_model_inst.has_category(channel)
            if not has_category:
                logger.warning(f"Category {channel} is not part of the inference model {self.inference_model}")

        hists = defaultdict(OrderedDict)

        for proc_key in list(h_in.keys()):
            proc_inst = None
            # try getting the config process via InferenceModel
            if has_category:
                # TODO: process customization based on inference process? e.g. scale
                inference_process = self.inference_model_inst.get_process(proc_key, channel)
                proc_inst = self.config_inst.get_process(inference_process.config_process)
            else:
                # try getting proc inst directly via config
                proc_inst = self.config_inst.get_process(proc_key, default=None)

            # replace string keys with process instances
            if proc_inst:
                plot_proc = [
                    proc for proc in process_insts if proc.has_process(proc_inst) or proc.name == proc_inst.name
                ]
                if len(plot_proc) != 1:
                    if len(plot_proc) > 1:
                        raise Exception(f"{proc_key} was assigned to more then one porcess insts ({plot_proc}) ")
                    else:
                        logger.warning(f"{proc_key} in root file, but won't be plotted.")
                    continue

                if plot_proc[0] not in hists:
                    hists[plot_proc[0]] = {}
                    hists[plot_proc[0]] = h_in[proc_key]
                else:
                    hists[plot_proc[0]] = hists[plot_proc[0]] + h_in[proc_key]

            if has_category:
                inference_category = self.inference_model_inst.get_category(channel)
                config_category = self.config_inst.get_category(inference_category.config_category)
                variable_inst = self.config_inst.get_variable(inference_category.config_variable)
            else:
                # default to dummy Category and Variable
                config_category = od.Category(channel, id=1)
                variable_inst = od.Variable("dummy")

            # call the plot function
            fig, _ = self.call_plot_func(
                self.plot_function,
                hists=hists,
                config_inst=self.config_inst,
                category_inst=config_category,
                variable_insts=variable_inst,
                **self.get_plot_parameters(),
            )

            outp["plots"].child(f"{channel}_{fit_type}.pdf", type="f").dump(fig, formatter="mpl")
