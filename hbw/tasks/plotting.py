# coding: utf-8

"""
Convenience wrapper tasks to simplify producing results and fetching & deleting their outputs
e.g. default sets of plots or datacards
"""

from __future__ import annotations

from collections import OrderedDict

import law
import luigi
import order as od

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    InferenceModelMixin, MLModelsMixin, ProducersMixin, SelectorStepsMixin,
    CalibratorsMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase1D, VariablePlotSettingMixin, ProcessPlotSettingMixin,
)
from columnflow.tasks.plotting import PlotVariables1D
# from columnflow.tasks.framework.remote import RemoteWorkflow
from hbw.tasks.base import HBWTask

from columnflow.util import dev_sandbox, DotDict, maybe_import

uproot = maybe_import("uproot")


logger = law.logger.get_logger(__name__)


class InferencePlots(
    HBWTask,
    # pass mixins to directly use plot parameters on command line
    PlotBase1D,
    VariablePlotSettingMixin,
    ProcessPlotSettingMixin,
    InferenceModelMixin,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    # law.LocalWorkflow,
    # RemoteWorkflow,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    plot_function = PlotVariables1D.plot_function

    # disable some parameters
    datasets = None
    processes = "st","tt","dy_lep","ggHH_kl_1_kt_1_ddl_hbbhww" # None
    categories = None

    inference_variables = law.CSVParameter(
        default=("config_variable", "variables_to_plot"),
        description="Inference category attributes to use to determine which variables to plot",
    )
    skip_variables = luigi.BoolParameter(default=False)
    # skip_data = luigi.BoolParameter(default=False)

    # def create_branch_map(self):
    #     # create a dummy branch map so that this task could run as a job
    #     return {0: None}

    # upstream requirements
    reqs = Requirements(
        # RemoteWorkflow.reqs,
        PlotVariables1D=PlotVariables1D,
    )

    # def workflow_requires(self):
    #     reqs = super().workflow_requires()
    #     return reqs

    def requires(self):
        reqs = {}

        inference_model = self.inference_model_inst

        # NOTE: this is not generally included in an inference model, but only in hbw analysis
        ml_model_name = inference_model.ml_model_name

        for inference_category in inference_model.categories:
            # decide which variables to plot based on the inference model and the variables parameter
            variables = []
            for attr in self.inference_variables:
                variables.extend(law.util.make_list(getattr(inference_category, attr, [])))
            if not self.skip_variables:
                variables.extend(self.variables)

            category = inference_category.config_category
            processes = ["st","tt","dy_lep","ggHH_sig_all","w_lnu"] #inference_category.data_from_processes

            # data_datasets = inference_category.config_data_datasets

            reqs[inference_category.name] = self.reqs.PlotVariables1D.req(
                self,
                variables=variables,
                categories=(category,),
                processes=processes,
                ml_models=(ml_model_name,),
            )

        return reqs

    def output(self):
        # use the input also as output (makes it easier to fetch and delete outputs)
        return {"0": self.target("dummy.txt")}

        # return self.requires()

    def run(self):
        pass


def load_hists_uproot(fit_diagnostics_path):
    """ Helper to load histograms from a fit_diagnostics file """
    # prepare output dict
    hists = DotDict()
    with uproot.open(fit_diagnostics_path) as tfile:
        keys = [key.split("/") for key in tfile.keys()]
        for key in keys:
            if len(key) != 3:
                continue

            # get the histogram from the tfile
            h_in = tfile["/".join(key)]

            # unpack key
            fit, channel, process = key
            process = process.split(";")[0]

            if "data" not in process:
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
)


def plot_postfit_shapes(
    hists: OrderedDict,
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

    plot_config = prepare_plot_config(
        hists,
        shape_norm=shape_norm,
        hide_errors=hide_errors,
    )

    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    default_style_config["ax_cfg"].pop("xlim")

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)
    if shape_norm:
        style_config["ax_cfg"]["ylabel"] = r"$\Delta N/N$"

    return plot_all(plot_config, style_config, **kwargs)


class PlotPostfitShapes(
    HBWTask,
    PlotBase1D,
    # to correctly setup our InferenceModel, we need all these mixins, but hopefully, all these
    # parameters are automatically resolved correctly
    InferenceModelMixin,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
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
        default="hbw.tasks.plotting.plot_postfit_shapes",
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
            fit_type = "fit_s"

        all_hists = all_hists[f"shapes_{fit_type}"]

        for channel, hists in all_hists.items():
            has_category = self.inference_model_inst.has_category(channel)
            if not has_category:
                logger.warning(f"Category {channel} is not part of the inference model {self.inference_model}")

            for proc_key in list(hists.keys()):
                # remove unnecessary histograms
                if "data" in proc_key or "total" in proc_key:
                    hists.pop(proc_key)
                    continue

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
                    hists[proc_inst] = hists[proc_key]
                    hists.pop(proc_key)

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
