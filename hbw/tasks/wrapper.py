# coding: utf-8

"""
Convenience wrapper tasks to simplify producing results and fetching & deleting their outputs
e.g. default sets of plots or datacards
NOTE: these tasks have not been tested after TAF init changes, so they might not work anymore.
"""

from __future__ import annotations


import law
import luigi

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    InferenceModelMixin, MLModelsMixin, HistProducerClassMixin,
    ProducerClassesMixin, ReducerClassMixin, SelectorClassMixin, CalibratorClassesMixin,
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
    law.WrapperTask,
    HBWTask,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ProducerClassesMixin,
):
    """
    Helper task to produce default set of control plots
    """
    is_single_config = False
    split_resolved_boosted = False
    output_collection_cls = law.NestedSiblingFileCollection

    @property
    def config_inst(self):
        return self.config_insts[0]

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

    # def output(self):
    #     return self.requires()

    # def run(self):
    #     pass


class MLInputPlots(
    HBWTask,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ProducerClassesMixin,
):
    """
    Helper task to produce default set of control plots
    """

    output_collection_cls = law.NestedSiblingFileCollection

    @property
    def config_inst(self):
        return self.config_insts[0]

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

    # def output(self):
    #     return self.requires()

    # def run(self):
    #     pass


from columnflow.tasks.histograms import MergeHistograms


class InferencePlots(
    law.WrapperTask,
    HBWTask,
    # pass mixins to directly use plot parameters on command line
    CalibratorClassesMixin,
    SelectorClassMixin,
    ReducerClassMixin,
    ProducerClassesMixin,
    MLModelsMixin,
    HistProducerClassMixin,
    InferenceModelMixin,
    # HistHookMixin,
    PlotBase1D,
    VariablePlotSettingMixin,
    ProcessPlotSettingMixin,
    # law.LocalWorkflow,
    # RemoteWorkflow,
):
    single_config = False
    resolution_task_cls = MergeHistograms
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    plot_function = PlotVariables1D.plot_function

    # disable some parameters (NOTE: disabling datasets/processes breaks things)
    # datasets = None
    # processes = None
    # categories = None

    # inference_variables = law.CSVParameter(
    #     default=("config_data.variables", "variables_to_plot"),
    #     description="Inference category attributes to use to determine which variables to plot",
    # )
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
        for config in self.configs:
            for inference_category in inference_model.categories:
                # decide which variables to plot based on the inference model and the variables parameter
                variables = []
                if self.add_variables:
                    variables.extend(self.variables)
                config_data = inference_category.config_data[config]
                variables.append(config_data.variable)
                category = config_data.category

                if not variables:
                    raise ValueError(
                        f"No variables to plot for inference category '{inference_category.name}' "
                        f"and config '{config}'. Please check the inference model and the variables parameter."
                    )

                # mc_processes = [p.config_data[config].process for p in inference_category.processes]
                # data_datasets = config_data.data_datasets
                # processes = config_data.data_from_processes

                # data_datasets = inference_category.config_data_datasets

                reqs[inference_category.name] = self.reqs.PlotVariables1D.req(
                    self,
                    variables=variables,
                    categories=(category,),
                    # processes=self.processes,
                    ml_models=law.util.make_tuple(ml_model_name),
                )

        return reqs


class ShiftedInferencePlots(
    law.WrapperTask,
    HBWTask,
    # pass mixins to directly use plot parameters on command line
    CalibratorClassesMixin,
    SelectorClassMixin,
    ProducerClassesMixin,
    PlotBase1D,
    VariablePlotSettingMixin,
    ProcessPlotSettingMixin,
    InferenceModelMixin,
    MLModelsMixin,
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
