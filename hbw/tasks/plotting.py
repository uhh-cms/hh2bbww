# coding: utf-8

"""
Convenience wrapper tasks to simplify producing results and fetching & deleting their outputs
e.g. default sets of plots or datacards
"""

import law
# import luigi

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    InferenceModelMixin, MLModelsMixin, ProducersMixin, SelectorStepsMixin,
    CalibratorsMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase1D, VariablePlotSettingMixin, ProcessPlotSettingMixin,
)
from columnflow.tasks.plotting import PlotVariables1D
# from columnflow.tasks.framework.remote import RemoteWorkflow
from hbw.tasks.base import HBWTask

from columnflow.util import dev_sandbox


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
    processes = None
    variables = ()
    categories = None

    # def create_branch_map(self):
    #     # create a dummy branch map so that this task could run as a job
    #     return {0: None}

    # upstream requirements
    reqs = Requirements(
        # RemoteWorkflow.reqs,
        PlotVariables1D=PlotVariables1D,
    )

    # skip_data = luigi.BoolParameter(default=False)

    # def workflow_requires(self):
    #     reqs = super().workflow_requires()

    #     from hbw.util import debugger; debugger()

    #     return reqs

    def requires(self):
        reqs = {}

        inference_model = self.inference_model_inst

        # NOTE: this is not generally included in an inference model, but only in hbw analysis
        ml_model_name = inference_model.ml_model_name

        for inference_category in inference_model.categories:
            # create one plot for each
            variable = inference_category.config_variable
            category = inference_category.config_category
            processes = inference_category.data_from_processes

            # data_datasets = inference_category.config_data_datasets

            reqs[inference_category.name] = self.reqs.PlotVariables1D.req(
                self,
                variables=(variable,),
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
