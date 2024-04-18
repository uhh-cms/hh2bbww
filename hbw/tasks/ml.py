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

import law
import luigi

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    SelectorMixin,
    CalibratorsMixin,
    ProducersMixin,
    MLModelTrainingMixin,
    MLModelMixin,
    MLModelsMixin,
    SelectorStepsMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.util import dev_sandbox, DotDict
from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.tasks.ml import MergeMLEvents, MergeMLStats, MLTraining
from hbw.tasks.base import HBWTask

logger = law.logger.get_logger(__name__)


class MLOptimizer(
    HBWTask,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
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

    # never run this task on GPU
    htcondor_gpus = 0

    allow_empty_ml_model = False

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeMLEvents=MergeMLEvents,
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
                dataset_inst.name: [
                    self.reqs.MergeMLEvents.req(
                        self,
                        config=config_inst.name,
                        dataset=dataset_inst.name,
                        calibrators=_calibrators,
                        selector=_selector,
                        producers=_producers,
                        fold=f,
                        tree_index=-1,
                    )
                    for f in range(self.ml_model_inst.folds)
                ]
                for dataset_inst in dataset_insts
            }
            for (config_inst, dataset_insts), _calibrators, _selector, _producers in zip(
                self.ml_model_inst.used_datasets.items(),
                self.calibrators,
                self.selectors,
                self.producers,
            )
        }
        reqs["stats"] = {
            config_inst.name: {
                dataset_inst.name: self.reqs.MergeMLStats.req(
                    self,
                    config=config_inst.name,
                    dataset=dataset_inst.name,
                    calibrators=_calibrators,
                    selector=_selector,
                    producers=_producers,
                    tree_index=-1,
                )
                for dataset_inst in dataset_insts
            }
            for (config_inst, dataset_insts), _calibrators, _selector, _producers in zip(
                self.ml_model_inst.used_datasets.items(),
                self.calibrators,
                self.selectors,
                self.producers,
            )
        }
        return reqs

    def requires(self):

        reqs = {}
        if not self.is_branch():
            return reqs

        process = self.branch_data["process"]
        # load events only for specified process and fold
        reqs["events"] = {
            config_inst.name: {
                dataset_inst.name: self.reqs.MergeMLEvents.req(
                    self,
                    config=config_inst.name,
                    dataset=dataset_inst.name,
                    calibrators=_calibrators,
                    selector=_selector,
                    producers=_producers,
                    fold=self.fold,
                )
                for dataset_inst in dataset_insts
                if dataset_inst.x.ml_process == process
            }
            for (config_inst, dataset_insts), _calibrators, _selector, _producers in zip(
                self.ml_model_inst.used_datasets.items(),
                self.calibrators,
                self.selectors,
                self.producers,
            )
        }

        # load stats for all processes
        reqs["stats"] = {
            config_inst.name: {
                dataset_inst.name: self.reqs.MergeMLStats.req(
                    self,
                    config=config_inst.name,
                    dataset=dataset_inst.name,
                    calibrators=_calibrators,
                    selector=_selector,
                    producers=_producers,
                    tree_index=-1)
                for dataset_inst in dataset_insts
            }
            for (config_inst, dataset_insts), _calibrators, _selector, _producers in zip(
                self.ml_model_inst.used_datasets.items(),
                self.calibrators,
                self.selectors,
                self.producers,
            )
        }

        return reqs

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
                stats = inputs["stats"][config_inst.name][dataset]["stats"].load(formatter="json")
                process = config_inst.get_dataset(dataset).x.ml_process
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
                input_target = inputs["events"][config_inst.name][dataset]["mlevents"]
                process = config_inst.get_dataset(dataset).x.ml_process
                events[process].append(ak.from_parquet(input_target.path))

        for process in events.keys():
            events[process] = ak.concatenate(events[process])
        return dict(events)

    @law.decorator.log
    @law.decorator.safe_output
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

        # dump stats
        outputs["stats"].dump(stats, formatter="json")

        # initialize the DatasetLoader
        ml_dataset = self.ml_model_inst.data_loader(self.ml_model_inst, process, events, stats)

        for input_array in self.ml_model_inst.data_loader.input_arrays:
            logger.info(f"loading data for input array {input_array}")
            # load data and split into training, validation, and testing
            train, val, test = ml_dataset.load_split_data(input_array)

            # store loaded data
            outputs[input_array]["train"][process][fold].dump(train, formatter="numpy")
            outputs[input_array]["val"][process][fold].dump(val, formatter="numpy")
            outputs[input_array]["test"][process][fold].dump(test, formatter="numpy")

            outputs["input_features"][process][fold].dump(ml_dataset.input_features, formatter="pickle")


class MLEvaluationSingleFold(
    HBWTask,
    MLModelMixin,
    ProducersMixin,
    SelectorMixin,
    CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    This task creates evaluation outputs for a single trained MLModel.
    """
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
            branch=self.fold,
            configs=(self.config_inst.name,),
            calibrators=(self.calibrators,),
            selectors=(self.selector,),
            producers=(self.producers,),
        )
        reqs["preml"] = self.reqs.MLPreTraining.req_different_branching(self, branch=-1)

        return reqs

    def requires(self):
        reqs = {}
        if not self.is_branch():
            return reqs

        reqs["training"] = self.reqs.MLTraining.req_different_branching(
            self,
            branch=self.fold,
            configs=(self.config_inst.name,),
            calibrators=(self.calibrators,),
            selectors=(self.selector,),
            producers=(self.producers,),
        )
        reqs["preml"] = self.reqs.MLPreTraining.req_different_branching(self, branch=-1)
        return reqs

    def output(self):
        k = self.ml_model_inst.folds
        process = self.branch_data["process"]

        outputs = {
            evaluation_array: {
                train_val_test: {process: {fold: (
                    self.target(f"{evaluation_array}_{train_val_test}_{process}_fold{fold}of{k}.npy")
                ) for fold in range(self.ml_model_inst.folds)
                }}
                for train_val_test in ("train", "val", "test")
            }
            for evaluation_array in self.ml_model_inst.data_loader.evaluation_arrays
        }

        return outputs

    @law.decorator.log
    @law.decorator.localize
    @law.decorator.safe_output
    def run(self):
        from hbw.ml.data_loader import MLProcessData

        # prepare inputs and outputs
        inputs = self.input()
        output = self.output()
        process = self.branch_data["process"]

        # open trained model and assign to ml_model_inst
        training_results = self.ml_model_inst.open_model(inputs["training"])
        self.ml_model_inst.trained_model = training_results["model"]
        self.ml_model_inst.best_model = training_results["best_model"]

        self.ml_model_inst.process_insts = [
            self.ml_model_inst.config_inst.get_process(proc)
            for proc in self.ml_model_inst.processes
        ]

        input_files = inputs["preml"]["collection"]
        input_files = law.util.merge_dicts(*[input_files[key] for key in input_files.keys()], deep=True)

        for data_split in ("train", "val", "test"):
            for fold in range(self.ml_model_inst.folds):
                data = MLProcessData(
                    self.ml_model_inst, input_files, data_split, process, fold, fold_modus="evaluation_only",
                )

                for evaluation_array in self.ml_model_inst.data_loader.evaluation_arrays:
                    # store loaded data
                    output[evaluation_array][data_split][self.branch_data["process"]][fold].dump(
                        getattr(data, evaluation_array), formatter="numpy",
                    )


class PlotMLResultsSingleFold(
    HBWTask,
    MLModelMixin,
    ProducersMixin,
    SelectorMixin,
    CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    This task creates plots for the results of a single trained MLModel.
    """
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

    def create_branch_map(self):
        return {0: None}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set the sandbox
        self.sandbox = self.ml_model_inst.sandbox(self)

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["training"] = self.reqs.MLTraining.req_different_branching(
            self,
            branch=self.fold,
            configs=(self.config_inst.name,),
            calibrators=(self.calibrators,),
            selectors=(self.selector,),
            producers=(self.producers,),
        )

        reqs["preml"] = self.reqs.MLPreTraining.req_different_branching(self, branch=-1)
        reqs["mlpred"] = self.reqs.MLEvaluationSingleFold.req_different_branching(self, branch=-1)

        return reqs

    def requires(self):
        reqs = {
            "training": self.reqs.MLTraining.req_different_branching(
                self,
                configs=(self.config_inst.name,),
                calibrators=(self.calibrators,),
                selectors=(self.selector,),
                producers=(self.producers,),
                branch=self.fold,
            ),
        }

        reqs["preml"] = self.reqs.MLPreTraining.req_different_branching(self, branch=-1)
        reqs["mlpred"] = self.reqs.MLEvaluationSingleFold.req_different_branching(self, branch=-1)

        return reqs

    def output(self):
        return {
            "plots": self.target("plots", dir=True),
            "stats": self.target("stats.json"),
        }

    @law.decorator.log
    @view_output_plots
    def run(self):
        # imports
        from hbw.ml.data_loader import MLProcessData
        from hbw.ml.plotting import (
            plot_confusion,
            plot_roc_ovr,
            plot_roc_ovo,
            plot_output_nodes,
            plot_input_features,
            plot_introspection,
        )

        # prepare inputs and outputs
        inputs = self.input()
        output = self.output()
        stats = {}

        # this initializes some process information (e.g. proc_inst.x.ml_id), but feels kind of hacky
        self.ml_model_inst.datasets(self.config_inst)

        # open all model files
        training_results = self.ml_model_inst.open_model(inputs["training"])
        self.ml_model_inst.trained_model = training_results["model"]
        self.ml_model_inst.best_model = training_results["best_model"]

        self.ml_model_inst.process_insts = [
            self.ml_model_inst.config_inst.get_process(proc)
            for proc in self.ml_model_inst.processes
        ]

        # load data
        input_files_preml = inputs["preml"]["collection"]
        input_files_mlpred = inputs["mlpred"]["collection"]
        input_files = law.util.merge_dicts(
            *[input_files_preml[key] for key in input_files_preml.keys()],
            *[input_files_mlpred[key] for key in input_files_mlpred.keys()],
            deep=True,
        )
        data = DotDict({
            "train": MLProcessData(self.ml_model_inst, input_files, "train", self.ml_model_inst.processes, self.fold),
            "val": MLProcessData(self.ml_model_inst, input_files, "val", self.ml_model_inst.processes, self.fold),
            "test": MLProcessData(self.ml_model_inst, input_files, "test", self.ml_model_inst.processes, self.fold),
        })

        # create plots
        # NOTE: this is currently hard-coded, could be made customizable and could also be parallelized since
        # input reading is quite fast, while producing certain plots takes a long time

        for data_split in ("train", "val", "test"):
            # confusion matrix
            plot_confusion(
                self.ml_model_inst,
                data[data_split],
                output["plots"],
                data_split,
                self.ml_model_inst.process_insts,
                stats,
            )

            # ROC curves
            plot_roc_ovr(
                self.ml_model_inst,
                data[data_split],
                output["plots"],
                data_split,
                self.ml_model_inst.process_insts,
                stats,
            )
            plot_roc_ovo(
                self.ml_model_inst,
                data[data_split],
                output["plots"],
                data_split,
                self.ml_model_inst.process_insts,
            )

        # input features
        plot_input_features(
            self.ml_model_inst,
            data.train,
            data.val,
            output["plots"],
            self.ml_model_inst.process_insts,
        )

        # output nodes
        plot_output_nodes(
            self.ml_model_inst,
            data.train,
            data.val,
            output["plots"],
            self.ml_model_inst.process_insts,
        )

        # introspection plot for variable importance ranking
        plot_introspection(
            self.ml_model_inst,
            output["plots"],
            data.test,
            input_features=self.ml_model_inst.input_features_ordered,
            stats=stats,
        )

        # dump all stats into yaml file
        output["stats"].dump(stats, formatter="json")
