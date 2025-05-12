# coding: utf-8

"""
Convenience wrapper tasks to simplify producing results and fetching & deleting their outputs
e.g. default sets of plots or datacards
"""

from __future__ import annotations


import law
import luigi

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    InferenceModelMixin, MLModelsMixin, ProducersMixin, SelectorStepsMixin,
    CalibratorsMixin, WeightProducerMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase1D, VariablePlotSettingMixin, ProcessPlotSettingMixin,
)
from columnflow.tasks.plotting import PlotVariables1D, PlotShiftedVariables1D
from columnflow.tasks.yields import CreateYieldTable
# from columnflow.tasks.framework.remote import RemoteWorkflow
from hbw.tasks.base import HBWTask

from columnflow.util import dev_sandbox


logger = law.logger.get_logger(__name__)


class ControlPlots(
    HBWTask,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
):
    split_resolved_boosted = False

    """
    Helper task to produce default set of control plots
    """
    def requires(self):
        lepton_tag = self.config_inst.x.lepton_tag
        lepton_channels = self.config_inst.x.lepton_channels
        reqs = {}

        reqs[f"control_plots_{lepton_tag}"] = PlotVariables1D.req(
            self,
            processes=(f"d{lepton_tag}",),
            process_settings=[["scale_signal"]],
            variables=[f"{lepton_tag}"],
            categories=(f"{lepton_tag}",),
            yscale="log",
            cms_label="pw",
        )

        for l_channel in lepton_channels:
            reqs[f"control_plots_{l_channel}"] = PlotVariables1D.req(
                self,
                processes=(f"d{l_channel}ch",),
                process_settings=[["scale_signal"]],
                variables=[f"{lepton_tag}"],
                categories=(f"{lepton_tag}_{l_channel}ch",),
                yscale="log",
                cms_label="pw",
            )
            reqs[f"yields_{l_channel}"] = CreateYieldTable.req(
                self,
                processes=(f"d{l_channel}ch",),
                categories=(f"{lepton_tag}_{l_channel}ch",),
            )
            if self.split_resolved_boosted:
                for j_channel in ("resolved", "boosted"):
                    reqs[f"control_plots_{l_channel}_{j_channel}"] = PlotVariables1D.req(
                        self,
                        processes=(f"d{l_channel}ch",),
                        process_settings=[["scale_signal"]],
                        variables=[f"{lepton_tag}_{j_channel}"],
                        categories=(f"{lepton_tag}_{l_channel}ch_{j_channel}",),
                        yscale="log",
                        cms_label="pw",
                    )

        return reqs

    def output(self):
        return self.requires()

    def run(self):
        pass


class MLInputPlots(
    HBWTask,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
):
    """
    Helper task to produce default set of control plots
    """
    def requires(self):
        lepton_tag = self.config_inst.x.lepton_tag
        lepton_channels = self.config_inst.x.lepton_channels
        reqs = {}

        reqs[f"ml_input_plots_{lepton_tag}"] = PlotVariables1D.req(
            self,
            processes=(f"d{lepton_tag}",),
            process_settings=[["scale_signal"]],
            variables=["mli_*"],
            categories=(f"{lepton_tag}",),
            yscale="log",
            cms_label="pw",
        )

        for l_channel in lepton_channels:
            reqs[f"ml_input_plots_{l_channel}"] = PlotVariables1D.req(
                self,
                processes=(f"d{l_channel}ch",),
                # process_settings=[["scale_signal"]],
                variables=["mli_*"],
                categories=(f"{lepton_tag}_{l_channel}ch",),
                yscale="log",
                cms_label="pw",
            )

        return reqs

    def output(self):
        return self.requires()

    def run(self):
        pass


class InferencePlots(
    law.WrapperTask,
    HBWTask,
    # pass mixins to directly use plot parameters on command line
    PlotBase1D,
    VariablePlotSettingMixin,
    ProcessPlotSettingMixin,
    InferenceModelMixin,
    WeightProducerMixin,
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
    # datasets = None
    # processes = None
    categories = None

    inference_variables = law.CSVParameter(
        default=("config_variable", "variables_to_plot"),
        description="Inference category attributes to use to determine which variables to plot",
    )
    add_variables = luigi.BoolParameter(
        default=False,
        description="whether to add plotting the variables from the --variables parameter; default: False")
    # skip_data = luigi.BoolParameter(default=False)

    # upstream requirements
    reqs = Requirements(
        PlotVariables1D=PlotVariables1D,
    )

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
            if self.add_variables:
                variables.extend(self.variables)

            category = inference_category.config_category
            processes = inference_category.data_from_processes

            # data_datasets = inference_category.config_data_datasets

            reqs[inference_category.name] = self.reqs.PlotVariables1D.req(
                self,
                variables=variables,
                categories=(category,),
                # processes=processes,
                ml_models=(ml_model_name,),
            )

        return reqs


class ShiftedInferencePlots(
    law.WrapperTask,
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
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    plot_function = PlotShiftedVariables1D.plot_function

    # disable some parameters
    datasets = None
    processes = None
    categories = None

    inference_variables = law.CSVParameter(
        default=("config_variable", "variables_to_plot"),
        description="Inference category attributes to use to determine which variables to plot",
    )
    add_variables = luigi.BoolParameter(default=False)

    # upstream requirements
    reqs = Requirements(
        PlotShiftedVariables1D=PlotShiftedVariables1D,
    )
    from hbw.util import timeit_multiple

    @timeit_multiple
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
            if self.add_variables:
                variables.extend(self.variables)

            category = inference_category.config_category
            processes = inference_category.data_from_processes

            for process in processes:
                inference_process = inference_model.get_process(process, inference_category.name)
                shifts = [
                    param.config_shift_source
                    for param in inference_process.parameters
                    if param.config_shift_source
                ]
                if not shifts:
                    continue
                logger.info(f"require plotting shifts {shifts} for process {process}")

                branch_name = f"{inference_category.name}_{process}"
                reqs[branch_name] = self.reqs.PlotShiftedVariables1D.req(
                    self,
                    variables=variables,
                    categories=(category,),
                    processes=(process,),
                    shift_sources=shifts,
                    ml_models=(ml_model_name,),
                )

        return reqs
