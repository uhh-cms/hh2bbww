# coding: utf-8

"""
Simple task that compares the stats of each of the requested MLModels and stores for each stat
the name of the best model and the corresponding value.
Example usage:
```
law run hbw.MLOptimizer --version prod1 --ml-models dense_3x64,dense_3x128,dense_3x256,dense_3x512
```
"""

import law
import luigi

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin,
)

from columnflow.util import dev_sandbox
from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.tasks.ml import MLTraining
from hbw.tasks.base import HBWTask


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
