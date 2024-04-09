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
    # SelectorMixin,
    CalibratorsMixin,
    # ProducerMixin,
    ProducersMixin,
    MLModelTrainingMixin,
    # MLModelMixin,
    MLModelsMixin,
    # ChunkedIOMixin,
    SelectorStepsMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
# from columnflow.tasks.reduction import MergeReducedEventsUser, MergeReducedEvents
# from columnflow.tasks.production import ProduceColumns
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

        # ml model requirements
        reqs["model"] = self.ml_model_inst.requires(self)

        return reqs

    def requires(self):

        reqs = {}

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

        # ml model requirements
        reqs["model"] = self.ml_model_inst.requires(self)

        return reqs

    def output(self):
        k = self.ml_model_inst.folds
        process = self.branch_data["process"]

        outputs = {
            input_array: {
                train_val_test: self.target(f"{input_array}_{train_val_test}_{process}_fold{self.fold}of{k}.npy")
                for train_val_test in ("train", "val", "test")
            } for input_array in self.ml_model_inst.input_arrays
        }
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
        # fold = self.fold

        # load stats and input data
        stats = self.merge_stats(inputs)
        events = self.merge_datasets(inputs)[process]

        # initialize the DatasetLoader
        ml_dataset = self.ml_model_inst.data_loader(self.ml_model_inst, process, events, stats)

        for input_array in self.ml_model_inst.input_arrays:
            logger.info(f"loading data for input array {input_array}")
            # load data and split into training, validation, and testing
            train, val, test = ml_dataset.load_split_data(input_array)

            # store loaded data
            outputs[input_array]["train"].dump(train, formatter="numpy")
            outputs[input_array]["val"].dump(val, formatter="numpy")
            outputs[input_array]["test"].dump(test, formatter="numpy")
