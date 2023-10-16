# coding: utf-8

"""
Custom base tasks.
"""


import law

from columnflow.tasks.framework.base import Requirements, BaseTask, ShiftTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin,
    VariablesMixin, DatasetsProcessesMixin, CategoriesMixin,
)
# from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.reduction import MergeReducedEventsUser, MergeReducedEvents
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.ml import MLEvaluation
from columnflow.tasks.histograms import MergeHistograms
from columnflow.util import dev_sandbox, maybe_import

ak = maybe_import("awkward")


class HBWTask(BaseTask):

    task_namespace = "hbw"


class ColumnsBaseTask(
    HBWTask,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    MergeReducedEventsUser,
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
        MergeReducedEventsUser.reqs,
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


class HistogramsBaseTask(
    HBWTask,
    DatasetsProcessesMixin,
    CategoriesMixin,
    VariablesMixin,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    ShiftTask,
    # law.LocalWorkflow,
    # RemoteWorkflow,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        # RemoteWorkflow.reqs,
        MergeHistograms=MergeHistograms,
    )

    # def create_branch_map(self):
    #     return {0: None}

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts

    def requires(self):
        return {
            d: self.reqs.MergeHistograms.req(
                self,
                dataset=d,
                branch=-1,
                _exclude={"branches"},
                _prefer_cli={"variables"},
            )
            for d in self.datasets
        }

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["merged_hists"] = self.requires_from_branch()

        return reqs

    def load_histogram(self, dataset, variable):
        histogram = self.input()[dataset]["collection"][0]["hists"].targets[variable].load(formatter="pickle")
        return histogram

    def reduce_histogram(self, histogram, processes, categories, shifts):
        import hist

        def flatten_nested_list(nested_list):
            return [item for sublist in nested_list for item in sublist]

        # transform into lists if necessary
        processes = law.util.make_list(processes)
        categories = law.util.make_list(categories)
        shifts = law.util.make_list(shifts)

        # get all leaf categories
        category_insts = list(map(self.config_inst.get_category, categories))
        leaf_category_insts = set(flatten_nested_list([
            category_inst.get_leaf_categories() or [category_inst]
            for category_inst in category_insts
        ]))

        # get all sub processes
        process_insts = list(map(self.config_inst.get_process, processes))
        sub_process_insts = set(flatten_nested_list([
            [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        ]))

        # get all shift instances
        shift_insts = [self.config_inst.get_shift(shift) for shift in shifts]

        # work on a copy
        h = histogram.copy()

        # axis selections
        h = h[{
            "process": [
                hist.loc(p.id)
                for p in sub_process_insts
                if p.id in h.axes["process"]
            ],
            "category": [
                hist.loc(c.id)
                for c in leaf_category_insts
                if c.id in h.axes["category"]
            ],
            "shift": [
                hist.loc(s.id)
                for s in shift_insts
                if s.id in h.axes["shift"]
            ],
        }]

        # axis reductions
        h = h[{"process": sum, "category": sum, "shift": sum}]

        return h
