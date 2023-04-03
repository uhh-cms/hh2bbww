# coding: utf-8

"""
Convenience wrapper tasks to simplify producing results and fetching & deleting their outputs
e.g. default sets of plots or datacards
"""

from columnflow.tasks.framework.mixins import (
    CalibratorsMixin,  # SelectorStepsMixin, ProducersMixin, MLModelsMixin,
)
from columnflow.tasks.plotting import PlotVariables1D
# from columnflow.tasks.framework.remote import RemoteWorkflow

from columnflow.util import dev_sandbox


class DefaultPlots(
    # MLModelsMixin,
    # ProducersMixin,
    # SelectorStepsMixin,
    CalibratorsMixin,
):
    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh")

    def requires(self):
        reqs = {}

        for channel in ("mu", "e"):

            # control plots with data
            reqs[f"control_plots_{channel}"] = PlotVariables1D.req(
                self,
                config="config_2017",
                producers=("features",),
                processes=(f"d{channel}ch",),
                process_settings=[["scale_signal"]],
                variables=["*"],
                categories=(f"{channel}ch",),
                yscale="log",
                cms_label="pw",
            )

            # ML input features
            reqs[f"ml_inputs_{channel}"] = PlotVariables1D.req(
                self,
                config="config_2017",
                producers=("ml_inputs",),
                processes=(f"{channel}ch",),
                # process_settings=[["scale_signal"]],
                variables=["mli_*"],
                categories=(f"{channel}ch",),
                yscale="log",
                cms_label="simpw",
            )

            # ML output nodes
            ml_model = "default"
            reqs[f"ml_outputs_{channel}"] = PlotVariables1D.req(
                self,
                config="config_2017",
                producers=("ml_inputs",),
                ml_models=(ml_model,),
                processes=(f"{channel}ch",),
                # process_settings=[["scale_signal"]],
                variables=[f"{ml_model}.*"],
                categories=(f"{channel}ch",),
                yscale="log",
                cms_label="simpw",
            )

        # ML outputs categorized (TODO)

        return reqs

    def output(self):

        # use the input also as output
        # (makes it easier to fetch and delete outputs)
        return self.requires()

    def run(self):
        pass
