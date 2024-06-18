# coding: utf-8

"""
Collection of patches of underlying columnflow tasks.
"""

import law
import luigi
from columnflow.util import memoize
from columnflow.tasks.framework.base import AnalysisTask
from columnflow.tasks.selection import SelectEvents
from columnflow.tasks.cutflow import CreateCutflowHistograms
from columnflow.tasks.reduction import ReduceEvents
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.histograms import CreateHistograms
from columnflow.tasks.ml import MLTraining, PrepareMLEvents, MLEvaluation

logger = law.logger.get_logger(__name__)


@memoize
def patch_mltraining():
    logger.info("Patching MLTraining to use NestedSiblingFileCollection and remove unnecessary requires...")
    from columnflow.tasks.framework.remote import RemoteWorkflow

    # patch the MLTraining output collection
    MLTraining.output_collection_cls = law.NestedSiblingFileCollection

    # remove unnceessary requires from MLTraining
    def workflow_requires(self):
        # cannot call super() here; hopefully we only need the workflow_requires from RemoteWorkflow
        reqs = RemoteWorkflow.workflow_requires(self)
        reqs["model"] = self.ml_model_inst.requires(self)
        return reqs

    def requires(self):
        reqs = {}
        reqs["model"] = self.ml_model_inst.requires(self)
        return reqs

    MLTraining.requires = requires
    MLTraining.workflow_requires = workflow_requires


@memoize
def patch_column_alias_strategy():
    # NOTE: not used since checks always fail
    # patch the missing_column_alias_strategy for all tasks
    # at SelectEvents, the btag_weight alias is missing, therefore the check cannot be used
    SelectEvents.missing_column_alias_strategy = "raise"
    CreateCutflowHistograms.missing_column_alias_strategy = "raise"
    ReduceEvents.missing_column_alias_strategy = "raise"
    ProduceColumns.missing_column_alias_strategy = "raise"

    # I would like to add this tag, but since we need to request column aliases for JEC and cannot
    # apply aliases of the Jet.pt here,
    CreateHistograms.missing_column_alias_strategy = "raise"
    PrepareMLEvents.missing_column_alias_strategy = "raise"
    MLEvaluation.missing_column_alias_strategy = "raise"


@memoize
def patch_all():
    patch_mltraining()
    # patch_column_alias_strategy()

    # setting the default version from the law.cfg
    AnalysisTask.version = luigi.Parameter(
        default=law.config.get_expanded("analysis", "default_version", None),
        description="mandatory version that is encoded into output paths",
    )
