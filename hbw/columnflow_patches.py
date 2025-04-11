# coding: utf-8

"""
Collection of patches of underlying columnflow tasks.
"""

import getpass

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
from columnflow.columnar_util import TaskArrayFunction

logger = law.logger.get_logger(__name__)


@memoize
def patch_mltraining():
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
    logger.info("patched MLTraining to use NestedSiblingFileCollection and remove unnecessary requires")


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
def patch_htcondor_workflow_naf_resources():
    """
    Patches the HTCondorWorkflow task to declare user-specific resources when running on the NAF.
    """
    from columnflow.tasks.framework.remote import HTCondorWorkflow

    def htcondor_job_resources(self, job_num, branches):
        # one "naf_<username>" resource per job, indendent of the number of branches in the job
        return {f"naf_{getpass.getuser()}": 1}

    HTCondorWorkflow.htcondor_job_resources = htcondor_job_resources

    logger.debug(f"patched htcondor_job_resources of {HTCondorWorkflow.task_family}")


@memoize
def patch_csp_versioning():
    """
    Patches the TaskArrayFunction to add the version to the string representation of the task.
    """

    from columnflow.tasks.framework.mixins import ArrayFunctionClassMixin

    def TaskArrayFunction_str(self):
        version = self.version() if callable(getattr(self, "version", None)) else getattr(self, "version", None)
        if version and not isinstance(version, (int, str)):
            raise Exception(f"version must be an integer or string, but is {version} ({type(version)})")
        version_str = f"V{version}" if version is not None else ""
        return f"{self.cls_name}{version_str}"

    def array_function_cls_repr(self, array_function):
        # NOTE: this might be a problem when we have identical names between different types of
        # TaskArrayFunctions...
        array_function_cls = TaskArrayFunction.get_cls(array_function)
        return TaskArrayFunction_str(array_function_cls)

    ArrayFunctionClassMixin.array_function_cls_repr = array_function_cls_repr
    TaskArrayFunction.__str__ = TaskArrayFunction_str
    logger.info(
        "patched TaskArrayFunction.__str__ to include the CSP version attribute",
    )


@memoize
def patch_default_version():
    # setting the default version from the law.cfg
    default_version = law.config.get_expanded("analysis", "default_version", None)
    AnalysisTask.version = luigi.Parameter(
        default=default_version,
        description="mandatory version that is encoded into output paths",
    )
    logger.info(f"using default version '{default_version}' for all AnalysisTasks")


@memoize
def patch_materialization_strategy():
    """
    Simple patch function to switch to the PARTITIONS materialization strategy for DaskArrayReader.
    We might want to try in the future if this improves memory usage, but this requires us to
    reproduce all existing outputs with this type of partitioning.
    """
    from columnflow.columnar_util import DaskArrayReader

    # Save the original __init__ method
    _original_init = DaskArrayReader.__init__

    def patched_init(self, *args, **kwargs):
        logger.debug(f"patched DaskArrayReader.__init__ with {DaskArrayReader.MaterializationStrategy.PARTITIONS}")
        # Modify the materialization_strategy before calling the original __init__
        kwargs["materialization_strategy"] = (
            DaskArrayReader.MaterializationStrategy.PARTITIONS
        )
        _original_init(self, *args, **kwargs)

    # Replace the original __init__ with the patched version
    DaskArrayReader.__init__ = patched_init


@memoize
def patch_all():
    # change the "retries" parameter default
    from columnflow.tasks.framework.remote import RemoteWorkflow
    RemoteWorkflow.retries = RemoteWorkflow.retries.copy(default=3)

    patch_mltraining()
    patch_htcondor_workflow_naf_resources()
    # patch_column_alias_strategy()
    patch_csp_versioning()
    patch_default_version()
