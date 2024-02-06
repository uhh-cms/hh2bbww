# coding: utf-8

"""
Convenience wrapper tasks to simplify producing results and fetching & deleting their outputs
e.g. default sets of plots or datacards
"""

import law

# from columnflow.tasks.framework.base import ConfigTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin,  # , MLModelsMixin,
)
from columnflow.tasks.plotting import PlotVariables1D
from columnflow.tasks.framework.remote import RemoteWorkflow
from hbw.tasks.base import HBWTask

from columnflow.util import dev_sandbox


class ControlPlotsSL(
    HBWTask,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
):
    """
    Helper task to produce default set of control plots
    """
    def requires(self):
        reqs = {}

        for l_channel in ("mu", "e"):
            for j_channel in ("resolved", "boosted"):
                reqs[f"control_plots_{l_channel}_{j_channel}"] = PlotVariables1D.req(
                    self,
                    processes=(f"{l_channel}ch",),
                    # processes=("ggHH_kl_1_kt_1_sl_hbbhww",),
                    process_settings=[["scale_signal"]],
                    variables=[f"sl_{j_channel}"],
                    categories=(f"sl_{l_channel}ch_{j_channel}",),
                    yscale="log",
                    cms_label="pw",
                )

        return reqs

    def output(self):
        return self.requires()

    def run(self):
        pass


class MLInputPlotsSL(
    HBWTask,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
):
    """
    Helper task to produce default set of control plots
    """
    def requires(self):
        reqs = {}

        for l_channel in ("mu", "e"):
            for j_channel in ("resolved", "boosted"):
                reqs[f"ml_input_plots_{l_channel}_{j_channel}"] = PlotVariables1D.req(
                    self,
                    processes=(f"{l_channel}ch",),
                    # process_settings=[["scale_signal"]],
                    variables=["mli_*"],
                    categories=(f"sl_{l_channel}ch_{j_channel}",),
                    yscale="log",
                    cms_label="simpw",
                )

        return reqs

    def output(self):
        return self.requires()

    def run(self):
        pass


class DefaultPlots(
    HBWTask,
    # MLModelsMixin,
    # ProducersMixin,
    # SelectorStepsMixin,
    CalibratorsMixin,
    # we only add workflow mixins to be able to directly pass --workflow instead of --cf.PlotVariables1d-workflow
    law.LocalWorkflow,
    RemoteWorkflow,
):
    # TODO: this should be three separate tasks with one additional "wrapper"
    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh")

    def create_branch_map(self):
        # create a dummy branch map so that this task could as a job
        return {0: None}

    def requires(self):
        reqs = {}

        for channel in ("mu", "e"):

            # control plots with data
            reqs[f"control_plots_{channel}"] = PlotVariables1D.req(
                self,
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
                producers=("ml_inputs",),
                processes=(f"{channel}ch",),
                # process_settings=[["scale_signal"]],
                variables=["mli_*"],
                categories=(f"{channel}ch",),
                yscale="log",
                cms_label="simpw",
            )

            # ML output nodes
            ml_model = "dense_default"
            reqs[f"ml_outputs_{channel}"] = PlotVariables1D.req(
                self,
                producers=(f"ml_{ml_model}",),
                ml_models=(ml_model,),
                processes=(f"{channel}ch",),
                # process_settings=[["scale_signal"]],
                variables=["mlscore.*"],
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
