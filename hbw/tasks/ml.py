# coding: utf-8

"""
Simple task that compares the stats of each of the requested MLModels and stores for each stat
the name of the best model and the corresponding value.
Example usage:
```
law run hbw.MLOptimizer --version prod1 --ml-models dense_3x64,dense_3x128,dense_3x256,dense_3x512
```
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from hbw.ml.data_loader import get_proc_mask
from hbw.util import log_memory
import law
import luigi

from columnflow.tasks.framework.base import Requirements, DatasetTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin,
    SelectorMixin,
    ReducerMixin,
    ProducersMixin,
    MLModelTrainingMixin,
    MLModelsMixin,
    MLModelDataMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.util import dev_sandbox, DotDict
from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.tasks.ml import PrepareMLEvents, MergeMLEvents, MergeMLStats, MLTraining
from hbw.tasks.base import HBWTask

logger = law.logger.get_logger(__name__)


class SimpleMergeMLEvents(
    CalibratorsMixin,
    SelectorMixin,
    ReducerMixin,
    ProducersMixin,
    MLModelDataMixin,
    DatasetTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    Custom reimplementation of the MergeMLEvents without ForestMerge
    took 2m 40s for 6 files, all 5 folds
    MergeMLEvents: 1m 40s for 6 files, 1 fold
    """
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # disable the shift parameter
    shift = None
    effective_shift = None
    allow_empty_shift = True

    allow_empty_ml_model = False

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        PrepareMLEvents=PrepareMLEvents,
    )

    def create_branch_map(self):
        return {0: None}

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["mlevents"] = self.reqs.PrepareMLEvents.req_different_branching(self, branch=-1)
        return reqs

    def requires(self):
        reqs = {}
        reqs["mlevents"] = self.reqs.PrepareMLEvents.req_different_branching(self, branch=-1)
        return reqs

    def output(self):
        k = self.ml_model_inst.folds
        return {
            "mlevents": [
                self.target(f"mlevents_f{i}of{k}.parquet")
                for i in range(k)
            ],
        }

    @law.decorator.timeit()
    def run(self):
        inputs = self.input()
        outputs = self.output()

        fold_paths = defaultdict(list)
        for inp in inputs["mlevents"]["collection"].targets.values():
            for i, target in enumerate(inp["mlevents"].targets):
                fold_paths[i].append(target.path)

        for i, paths in fold_paths.items():
            with self.publish_step(f"merge fold {i} from dataset {self.dataset} ({len(paths)} files)"):
                law.pyarrow.merge_parquet_files(paths, outputs["mlevents"][i].path)


class MLOptimizer(
    CalibratorsMixin,
    SelectorMixin,
    ReducerMixin,
    ProducersMixin,
    HBWTask,
    MLModelsMixin,
):
    reqs = Requirements(MLTraining=MLTraining)

    # sandbox = dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_plotting.sh")
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    ml_folds = luigi.Parameter(
        default="0",  # NOTE: this seems to work but is most likely not optimally implemented
        description="Fold of which ML model is supposed to be run",
    )

    def requires(self):
        reqs = {
            "models": [self.reqs.MLTraining.req(
                self,
                branches=self.ml_folds,
                ml_model=ml_model,
            ) for ml_model in self.ml_models],
        }
        return reqs

    def output(self):
        # use the input also as output
        # (makes it easier to fetch and delete outputs)
        return {
            "model_summary": self.target("model_summary.yaml"),
        }

    def run(self):
        model_names = self.ml_models

        # store for each key in the stats dict, which model performed the best (assuming larger=better)
        # "0_" prefix to have them as the first elements in the yaml file
        model_summary = {"0_best_model": {}, "0_best_value": {}}

        for model_name, inp in zip(model_names, self.input()["models"]):
            stats = inp["collection"][0]["stats"].load(formatter="yaml")

            for stat_name, value in stats.items():
                best_value = model_summary["0_best_value"].get(stat_name, EMPTY_FLOAT)
                if value > best_value:
                    # store for each stat the best model and the best value
                    model_summary["0_best_value"][stat_name] = value
                    model_summary["0_best_model"][stat_name] = model_name

                # store for each stat all models + values (with value as key to have them ordered)
                stat_summary = model_summary.get(stat_name, {})
                stat_summary[value] = model_name
                model_summary[stat_name] = stat_summary

        self.output()["model_summary"].dump(model_summary, formatter="yaml")


# TODO: Cross check if the trian weights work as intended --> by reading out porcess_id
class MLPreTraining(
    HBWTask,
    MLModelTrainingMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    This task prepares input arrays for MLTraining, producing one numpy array per fold, process,
    array type, and training/validation/test set.

    Attributes that are required by the requested MLModel are:
    - folds: number of folds to be used
    - processes: list of processes to be used
    - data_loader: class to be used to load the data
    - input_arrays: list of input arrays to be loaded. Each array is a string, pointing to a property
    of the data_loader class that returns the data.
    """
    resolution_task_cls = SimpleMergeMLEvents

    # never run this task on GPU
    htcondor_gpus = 0

    allow_empty_ml_model = False

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeMLEvents=MergeMLEvents,
        SimpleMergeMLEvents=SimpleMergeMLEvents,
        MergeMLStats=MergeMLStats,
    )

    @property
    def sandbox(self):
        # use the default columnar sandbox
        return dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    @property
    def accepts_messages(self):
        return self.ml_model_inst.accepts_scheduler_messages

    @property
    def fold(self):
        if not self.is_branch():
            return None
        return self.branch_data["fold"]

    @property
    def config_inst(self):
        return self.config_insts[0]

    def create_branch_map(self):
        return [
            DotDict({"fold": fold, "process": process})
            for fold in range(self.ml_model_inst.folds)
            for process in self.ml_model_inst.processes
        ]

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["events"] = {
            config_inst.name: {
                dataset_inst.name: self.reqs.SimpleMergeMLEvents.req_different_branching(
                    self,
                    config=config_inst.name,
                    dataset=dataset_inst.name,
                    branch=-1,
                )
                for dataset_inst in dataset_insts
            }
            for config_inst, dataset_insts in self.ml_model_inst.used_datasets.items()
        }
        reqs["stats"] = {
            config_inst.name: {
                dataset_inst.name: self.reqs.MergeMLStats.req_different_branching(
                    self,
                    config=config_inst.name,
                    dataset=dataset_inst.name,
                    branch=-1,
                )
                for dataset_inst in dataset_insts
            }
            for config_inst, dataset_insts in self.ml_model_inst.used_datasets.items()
        }
        return reqs

    def requires(self):

        reqs = {}
        if not self.is_branch():
            return reqs

        process = self.branch_data["process"]

        # load events
        reqs["events"] = {
            config_inst.name: {
                dataset_inst.name: self.reqs.SimpleMergeMLEvents.req_different_branching(
                    self,
                    config=config_inst.name,
                    dataset=dataset_inst.name,
                    branch=-1,
                )
                for dataset_inst in dataset_insts
                if dataset_inst.x.ml_process == process
            }
            for config_inst, dataset_insts in self.ml_model_inst.used_datasets.items()
        }

        # load stats for all processes
        reqs["stats"] = {
            config_inst.name: {
                dataset_inst.name: self.reqs.MergeMLStats.req_different_branching(
                    self,
                    config=config_inst.name,
                    dataset=dataset_inst.name,
                    branch=-1,
                )
                for dataset_inst in dataset_insts
            }
            for config_inst, dataset_insts in self.ml_model_inst.used_datasets.items()
        }

        return reqs

    def store_parts(self):
        parts = super().store_parts()

        # we do not want to store the MLModel instance, but only the relevant settings from the
        # MLModel (e.g. the data_loader class, the train_val_test_split and the input_features)
        # could also be represented by a singular attribute of the MLModel, but that would be needed to be updated
        # manually, which is error-prone

        # replace the ml_model entry
        # NOTE: at the moment, we bookkeep preml via a single attribute; we could also bookkeep the relevant settings
        # from the MLModel (e.g. the data_loader class, the train_val_test_split and the input_features)
        store_name = self.ml_model_inst.store_name or self.ml_model_inst.cls_name
        preml_store_name = getattr(self.ml_model_inst, "preml_store_name", "")
        parts.insert_before("ml_model", "ml_data", f"ml__{store_name}__{preml_store_name}")

        parts.pop("ml_model")
        return parts

    def output(self):
        k = self.ml_model_inst.folds
        process = self.branch_data["process"]

        outputs = {
            input_array: {
                train_val_test: {process: {self.fold: (
                    self.target(f"{input_array}_{train_val_test}_{process}_fold{self.fold}of{k}.npy")
                )}}
                for train_val_test in ("train", "val", "test")
            }
            for input_array in self.ml_model_inst.data_loader.input_arrays
        }

        # NOTE: this is stored per fold and process, since we cannot do the check that they are all
        # the same since we parallelized over processes/folds
        outputs["input_features"] = {process: {self.fold: (
            self.target(f"input_features_{process}_fold{self.fold}of{k}.pickle")
        )}}

        # the stats dict is created per process+fold, but should always be identical, therefore we store it only once
        outputs["stats"] = self.target("stats.json")

        outputs["cross_check_weights"] = self.target(f"cross_check_weights_{process}_{self.fold}.yaml")

        outputs["parameters"] = self.target("parameters.yaml")

        return outputs

    def merge_stats(self, inputs) -> dict:
        """
        Merge stats from different inputs into a single dict.
        """
        merged_stats = defaultdict(dict)
        for config_inst in self.ml_model_inst.config_insts:
            used_datasets = inputs["stats"][config_inst.name].keys()
            for dataset in used_datasets:
                # gather stats per ml process
                stats = inputs["stats"][config_inst.name][dataset]["collection"][0]["stats"].load(formatter="json")
                process = config_inst.get_dataset(dataset).x.ml_process
                proc_inst = config_inst.get_process(process)
                sub_id = [
                    proc_inst.id
                    for proc_inst, _, _ in proc_inst.walk_processes(include_self=True)
                ]

                for id in list(stats["num_events_per_process"].keys()):
                    if int(id) not in sub_id:
                        for key in list(stats.keys()):
                            stats[key].pop(id, None)

                MergeMLStats.merge_counts(merged_stats[process], stats)

        return merged_stats

    def merge_datasets(self, inputs):
        """
        Merge datasets from different inputs into a single awkward array.
        """
        import awkward as ak
        events = defaultdict(list)
        for config_inst in self.ml_model_inst.config_insts:
            used_datasets = inputs["events"][config_inst.name].keys()
            for dataset in used_datasets:
                input_target = inputs["events"][config_inst.name][dataset].collection[0]["mlevents"][self.fold]
                # input_target = inputs["events"][config_inst.name][dataset]["mlevents"]
                process = config_inst.get_dataset(dataset).x.ml_process
                try:
                    events[process].append(ak.from_parquet(input_target.path))
                except IndexError as e:
                    if e.args[0] == "0 out of bounds":
                        logger.warning(
                            f"No events found for process {process} in dataset {dataset} "
                            f"for config {config_inst.name} and fold {self.fold}. "
                            "File will be skipped after checking stats that it is indeed empty "
                            "(if not, exception will be raised).",
                        )
                        stats = inputs["stats"][config_inst.name][dataset]["collection"][0]["stats"].load(formatter="json")  # noqa: E501
                        if stats["num_fold_events"][str(self.fold)] == 0:
                            continue
                        else:
                            raise e


        for process in events.keys():
            events[process] = ak.concatenate(events[process])
        return dict(events)

    @law.decorator.log
    @law.decorator.safe_output
    @law.decorator.timeit()
    def run(self):
        # prepare inputs and outputs
        inputs = self.input()
        outputs = self.output()

        # prepare objects
        process = self.branch_data["process"]
        fold = self.fold

        # load stats and input data
        stats = self.merge_stats(inputs)
        events = self.merge_datasets(inputs)[process]

        events = self.ml_model_inst.patch_events(events)

        # dump stats
        outputs["stats"].dump(stats, formatter="json")

        # initialize the DatasetLoader
        if self.ml_model_inst.negative_weights == "ignore":
            ml_dataset = self.ml_model_inst.data_loader(self.ml_model_inst, process, events, stats)
            logger.info(
                f"{self.ml_model_inst.negative_weights} method chosen to handle negative weights: "
                "All negative weights will be removed from training.",
            )
        else:
            ml_dataset = self.ml_model_inst.data_loader(self.ml_model_inst, process, events, stats, skip_mask=True)

        sum_train_weights = np.sum(ml_dataset.train_weights)
        n_events_per_fold = len(ml_dataset.train_weights)
        logger.info(f"Sum of training weights is: {sum_train_weights} for {n_events_per_fold} {process} events")
        xcheck = {}
        if self.ml_model_inst.config_inst.get_process(process).x("ml_config", None):
            xcheck[process] = {}
            if self.ml_model_inst.config_inst.get_process(process).x.ml_config.weighting == "equal":
                for sub_proc in self.ml_model_inst.config_inst.get_process(process).x.ml_config.sub_processes:
                    proc_mask, _ = get_proc_mask(ml_dataset._events, sub_proc, self.ml_model_inst.config_inst)
                    xcheck_weight_sum = np.sum(ml_dataset.train_weights[proc_mask])
                    xcheck_n_events = len(ml_dataset.train_weights[proc_mask])
                    logger.info(
                        f"For the equal weighting method the sum of weights for {sub_proc} is {xcheck_weight_sum} "
                        f"(No. of events: {xcheck_n_events})",
                    )
                    if sub_proc not in xcheck[process]:
                        xcheck[process][sub_proc] = {}
                    xcheck[process][sub_proc]["weight_sum"] = int(xcheck_weight_sum)
                    xcheck[process][sub_proc]["num_events"] = xcheck_n_events

        outputs["cross_check_weights"].dump(xcheck, formatter="yaml")

        for input_array in self.ml_model_inst.data_loader.input_arrays:
            logger.info(f"loading data for input array {input_array}")
            # load data and split into training, validation, and testing
            train, val, test = ml_dataset.load_split_data(input_array)

            # store loaded data
            outputs[input_array]["train"][process][fold].dump(train, formatter="numpy")
            outputs[input_array]["val"][process][fold].dump(val, formatter="numpy")
            outputs[input_array]["test"][process][fold].dump(test, formatter="numpy")

            outputs["input_features"][process][fold].dump(ml_dataset.input_features, formatter="pickle")
        # dump parameters of the DatasetLoader
        outputs["parameters"].dump(ml_dataset.parameters, formatter="yaml")

    @law.workflow.base.workflow_property
    def workflow_run(self):
        pass


class MLEvaluationSingleFold(
    # NOTE: mixins might need fixing, needs to be checked
    HBWTask,
    MLModelTrainingMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    This task creates evaluation outputs for a single trained MLModel.
    """
    resolution_task_cls = SimpleMergeMLEvents

    sandbox = None

    allow_empty_ml_model = False

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MLPreTraining=MLPreTraining,
        MLTraining=MLTraining,
    )

    fold = luigi.IntParameter(
        default=0,
        description="the fold index of the MLTraining to use; must be compatible with the "
        "number of folds defined in the ML model; default: 0",
    )

    data_split = luigi.Parameter(
        default="test",
        description="the data split to evaluate; must be one of 'train', 'val', 'test'; default: 'test'",
    )

    @property
    def config_inst(self):
        return self.config_insts[0]

    def create_branch_map(self):
        return [
            DotDict({"process": process})
            for process in self.ml_model_inst.processes
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set the sandbox
        self.sandbox = self.ml_model_inst.sandbox(self)

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["models"] = self.reqs.MLTraining.req_different_branching(
            self,
            branches=(self.fold,),
            configs=(self.config_inst.name,),
        )
        reqs["preml"] = self.reqs.MLPreTraining.req_different_branching(
            self,
            configs=(self.config_inst.name,),
            branch=-1,
        )
        # reqs["preml"] = self.reqs.MLPreTraining.req_different_branching(self, branch=-1)

        return reqs

    def requires(self):
        reqs = {}
        if not self.is_branch():
            return reqs

        reqs["training"] = self.reqs.MLTraining.req_different_branching(
            self,
            branches=(self.fold,),
            branch=self.fold,
            configs=(self.config_inst.name,),
        )
        reqs["preml"] = self.reqs.MLPreTraining.req_different_branching(
            self,
            configs=(self.config_inst.name,),
            branch=-1,
        )
        # reqs["preml"] = self.reqs.MLPreTraining.req_different_branching(self, branch=-1)
        return reqs

    def output(self):
        k = self.ml_model_inst.folds
        process = self.branch_data["process"]

        outputs = {
            evaluation_array: {
                self.data_split: {process: {fold: (
                    self.target(f"{evaluation_array}_{self.data_split}_{process}_fold{fold}of{k}.npy")
                ) for fold in range(self.ml_model_inst.folds)
                }}}
            for evaluation_array in self.ml_model_inst.data_loader.evaluation_arrays
        }

        return outputs

    @law.decorator.log
    @law.decorator.localize
    @law.decorator.safe_output
    @law.decorator.timeit()
    def run(self):
        from hbw.ml.data_loader import MLProcessData

        # prepare inputs and outputs
        inputs = self.input()
        output = self.output()
        process = self.branch_data["process"]

        logger.info(f"evaluating model {str(self.ml_model_inst)} for process {process} and fold {self.fold}")

        # open trained model and assign to ml_model_inst
        training_results = self.ml_model_inst.open_model(inputs["training"])
        self.ml_model_inst.trained_model = training_results["model"]
        self.ml_model_inst.best_model = training_results["best_model"]

        input_files = inputs["preml"]["collection"]
        input_files = law.util.merge_dicts(*[input_files[key] for key in input_files.keys()], deep=True)

        for fold in range(self.ml_model_inst.folds):
            data = MLProcessData(
                self.ml_model_inst, input_files, self.data_split, process, fold, fold_modus="evaluation_only",
            )

            for evaluation_array in self.ml_model_inst.data_loader.evaluation_arrays:
                # store loaded data
                output[evaluation_array][self.data_split][self.branch_data["process"]][fold].dump(
                    getattr(data, evaluation_array), formatter="numpy",
                )


class PlotMLResultsSingleFold(
    # NOTE: mixins might need fixing, needs to be checked
    HBWTask,
    MLModelTrainingMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    This task creates plots for the results of a single trained MLModel.
    """
    resolution_task_cls = SimpleMergeMLEvents

    sandbox = None

    allow_empty_ml_model = False

    # strategy for handling missing source columns when adding aliases on event chunks
    missing_column_alias_strategy = "original"

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MLPreTraining=MLPreTraining,
        MLEvaluationSingleFold=MLEvaluationSingleFold,
        MLTraining=MLTraining,
    )

    fold = luigi.IntParameter(
        default=0,
        description="the fold index of the MLTraining to use; must be compatible with the "
        "number of folds defined in the ML model; default: 0",
    )

    view_cmd = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="a command to execute after the task has run to visualize plots right in the "
        "terminal; no default",
    )

    data_splits = ("train", "val", "test")

    @property
    def config_inst(self):
        return self.config_insts[0]

    def create_branch_map(self):
        return {0: None}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set the sandbox
        self.sandbox = self.ml_model_inst.sandbox(self)

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["inputs"] = self.requires_from_branch()

        return reqs

    def requires(self):
        reqs = {
            "training": self.reqs.MLTraining.req_different_branching(
                self,
                configs=(self.config_inst.name,),
                branches=(self.fold,),
            ),
        }

        reqs["preml"] = self.reqs.MLPreTraining.req_different_branching(
            self,
            configs=(self.config_inst.name,),
            branch=-1,
        )
        reqs["mlpred"] = {
            data_split:
            self.reqs.MLEvaluationSingleFold.req_different_branching(self, data_split=data_split, branch=-1)
            for data_split in self.data_splits
        }

        return reqs

    def output(self):
        return {
            "plots": self.target("plots", dir=True),
            "stats": self.target("stats.json"),
        }

    @law.decorator.log
    @law.decorator.timeit()
    @view_output_plots
    def run(self):
        logger.info(f"creating plots for model {str(self.ml_model_inst)} for fold {self.fold}")
        log_memory("Start")
        # prepare inputs and outputs
        inputs = self.input()

        # this initializes some process information (e.g. proc_inst.x.ml_id), but feels kind of hacky
        self.ml_model_inst.datasets(self.config_inst)

        # open all model files
        training_results = self.ml_model_inst.open_model(inputs["training"])
        self.ml_model_inst.trained_model = training_results["model"]
        self.ml_model_inst.best_model = training_results["best_model"]

        # Load data lazily and process incrementally to reduce memory usage
        input_files_preml = inputs["preml"]["collection"]
        input_files_mlpred = {data_split: value["collection"] for data_split, value in inputs["mlpred"].items()}
        input_files = law.util.merge_dicts(
            *[input_files_preml[key] for key in input_files_preml.keys()],
            *[
                input_files_mlpred[data_split][key]
                for data_split in input_files_mlpred.keys()
                for key in input_files_mlpred[data_split].keys()
            ],
            deep=True,
        )

        # Load all data at once (original approach) but with memory-efficient plotting
        from hbw.ml.data_loader import MLProcessData
        data = DotDict({
            data_split: MLProcessData(
                self.ml_model_inst,
                input_files,
                data_split,
                self.ml_model_inst.processes,
                self.fold,
            ) for data_split in self.data_splits
        })
        self.create_plots(data)

    def create_plots(self, data):
        """Create plots with matplotlib cleanup for memory efficiency"""
        import gc
        import matplotlib
        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt

        output = self.output()
        stats = {}
        from hbw.ml.plotting import (
            plot_confusion,
            plot_roc_ovr,
            plot_roc_ovo,
            plot_output_nodes,
            plot_input_features,
            plot_introspection,
        )

        # create plots
        # NOTE: could be more parallelized since input reading is quite fast,
        # while producing certain plots takes a long time

        # NOTE: making plots per sub-process (`self.ml_model_inst.process_insts`) might be nice,
        # but at the moment we can only plot per DNN node process (`self.ml_model_inst.train_node_process_insts`)

        # input features
        log_memory("Before plotting input features")
        plot_input_features(
            self.ml_model_inst,
            data.train,
            data.val,
            output["plots"],
            self.ml_model_inst.train_node_process_insts,
        )
        log_memory("After plotting input features")
        data.train.cleanup()
        data.val.cleanup()
        plt.close("all")
        gc.collect()

        # output nodes
        log_memory("Before plotting output nodes")
        plot_output_nodes(
            self.ml_model_inst,
            data,
            output["plots"],
            self.ml_model_inst.train_node_process_insts,
        )
        log_memory("After plotting output nodes")
        plt.close("all")
        gc.collect()

        # iterate over all data splits and create plots
        for data_split in self.data_splits:
            logger.info(f"Creating plots for {data_split} split...")

            log_memory("Before plotting confusion matrix")
            # confusion matrix
            plot_confusion(
                self.ml_model_inst,
                data[data_split],
                output["plots"],
                data_split,
                self.ml_model_inst.train_node_process_insts,
                stats,
            )
            log_memory("After plotting confusion matrix")
            plt.close("all")
            gc.collect()

            log_memory("Before plotting ROC curves")
            # ROC curves
            plot_roc_ovr(
                self.ml_model_inst,
                data[data_split],
                output["plots"],
                data_split,
                self.ml_model_inst.train_node_process_insts,
                stats,
            )
            log_memory("After plotting ROC curves")
            plt.close("all")
            gc.collect()

            # ROC curves for one-vs-one classification
            log_memory("Before plotting ROC OVO curves")
            plot_roc_ovo(
                self.ml_model_inst,
                data[data_split],
                output["plots"],
                data_split,
                self.ml_model_inst.train_node_process_insts,
            )
            log_memory("After plotting ROC OVO curves")
            plt.close("all")
            gc.collect()

            if data_split != "test":
                # remove the train and val data to save memory
                data[data_split].cleanup()
                gc.collect()
                log_memory(f"After cleaning up {data_split} split")

        # # remove the train and val data to save memory
        # del data.train
        # del data.val

        # introspection plot for variable importance ranking
        log_memory("Before plotting introspection")
        plot_introspection(
            self.ml_model_inst,
            output["plots"],
            data.test,
            input_features=self.ml_model_inst.input_features_ordered,
            stats=stats,
        )
        log_memory("After plotting introspection")
        plt.close("all")
        gc.collect()

        # dump all stats into yaml file
        output["stats"].dump(stats, formatter="json")


class PlotMLResultsSingleFoldTest(PlotMLResultsSingleFold):
    data_splits = ("test",)

    def create_plots(self, data):
        """Memory-efficient plotting for test-only data"""
        import gc
        import matplotlib
        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt

        output = self.output()
        stats = {}
        from hbw.ml.plotting import (
            plot_confusion,
            plot_roc_ovr,
            plot_roc_ovo,
            plot_output_nodes,
            plot_introspection,
        )

        # Since we only have test data, process sequentially with cleanup
        for data_split in self.data_splits:
            logger.info(f"Creating plots for {data_split} split...")

            # confusion matrix
            plot_confusion(
                self.ml_model_inst,
                data[data_split],
                output["plots"],
                data_split,
                self.ml_model_inst.train_node_process_insts,
                stats,
            )
            plt.close("all")
            gc.collect()

            # ROC curves
            plot_roc_ovr(
                self.ml_model_inst,
                data[data_split],
                output["plots"],
                data_split,
                self.ml_model_inst.train_node_process_insts,
                stats,
            )
            plt.close("all")
            gc.collect()

            plot_roc_ovo(
                self.ml_model_inst,
                data[data_split],
                output["plots"],
                data_split,
                self.ml_model_inst.train_node_process_insts,
            )
            plt.close("all")
            gc.collect()

        # output nodes
        plot_output_nodes(
            self.ml_model_inst,
            data,
            output["plots"],
            self.ml_model_inst.train_node_process_insts,
        )
        plt.close("all")
        gc.collect()

        # introspection plot for variable importance ranking
        plot_introspection(
            self.ml_model_inst,
            output["plots"],
            data.test,
            input_features=self.ml_model_inst.input_features_ordered,
            stats=stats,
        )
        plt.close("all")
        gc.collect()

        # dump all stats into yaml file
        output["stats"].dump(stats, formatter="json")
