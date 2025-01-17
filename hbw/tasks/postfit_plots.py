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
import hist
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


def load_hists_uproot(fit_diagnostics_path):
    """ Helper to load histograms from a fit_diagnostics file """
    # prepare output dict
    hists = DotDict()
    with uproot.open(fit_diagnostics_path) as tfile:
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

            # if "data" not in process:
            # transform TH1F to hist
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
        all_hists = load_hists_uproot(self.fit_diagnostics_file)

        outp = self.output()
        if self.prefit:
            fit_type = "prefit"
        else:
            fit_type = "postfit"

        all_hists = all_hists[f"{fit_type}"]

        process_insts = list(map(self.config_inst.get_process, self.processes))
        sub_process_insts = {
            proc: [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        }

        # histogram data per process

        for channel, h_in in all_hists.items():
            has_category = self.inference_model_inst.has_category(channel)
            if not has_category:
                logger.warning(f"Category {channel} is not part of the inference model {self.inference_model}")

            hists = defaultdict(OrderedDict)

            for process_inst in process_insts:
                for proc in sub_process_insts[process_inst]:
                    for proc_key in list(h_in.keys()):

                        p = reverse_inf_proc(proc_key)

                        if p == proc.name:

                            if process_inst not in hists:
                                hists[process_inst] = {}
                                hists[process_inst] = h_in[proc_key]
                            else:
                                try:
                                    hists[process_inst] = hists[process_inst] + h_in[proc_key]
                                except Exception:
                                    __import__("IPython").embed()

            # try getting the config category and variable via InferenceModel
            if has_category:
                # TODO: category/variable customization based on inference model?
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
