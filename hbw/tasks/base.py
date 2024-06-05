# coding: utf-8

"""
Custom base tasks.
"""


import law

from columnflow.tasks.framework.base import Requirements, BaseTask
from columnflow.tasks.framework.mixins import (
    ProducersMixin, MLModelsMixin,
)
from columnflow.tasks.reduction import ReducedEventsUser, MergeReducedEvents
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.ml import MLEvaluation
from columnflow.util import dev_sandbox, maybe_import

ak = maybe_import("awkward")


class HBWTask(BaseTask):

    task_namespace = "hbw"


class ColumnsBaseTask(
    HBWTask,
    MLModelsMixin,
    ProducersMixin,
    ReducedEventsUser,
    law.LocalWorkflow,
):
    """
    Bask task to handle columns after Reduction, Production and MLEvaluation.
    An exemplary implementation of how to handle the inputs in a run method can be
    found in columnflow/tasks/union.py
    """

    exclude_index = True

    # upstream requirements
    reqs = Requirements(
        ReducedEventsUser.reqs,
        MergeReducedEvents=MergeReducedEvents,
        ProduceColumns=ProduceColumns,
        MLEvaluation=MLEvaluation,
    )

    # sandbox = dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_plotting.sh")
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    def workflow_requires(self):
        reqs = super().workflow_requires()

        # require the full merge forest
        reqs["events"] = self.reqs.MergeReducedEvents.req(self, tree_index=-1)

        if not self.pilot:
            if self.producers:
                reqs["producers"] = [
                    self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                    for producer_inst in self.producer_insts
                    if producer_inst.produced_columns
                ]
            if self.ml_models:
                reqs["ml"] = [
                    self.reqs.MLEvaluation.req(self, ml_model=m)
                    for m in self.ml_models
                ]

        return reqs

    def requires(self):
        reqs = {
            "events": self.reqs.MergeReducedEvents.req(self, tree_index=self.branch, _exclude={"branch"}),
        }

        if self.producers:
            reqs["producers"] = [
                self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                for producer_inst in self.producer_insts
                if producer_inst.produced_columns
            ]
        if self.ml_models:
            reqs["ml"] = [
                self.reqs.MLEvaluation.req(self, ml_model=m)
                for m in self.ml_models
            ]

        return reqs
