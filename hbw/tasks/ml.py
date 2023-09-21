# coding: utf-8

"""
Simple task that compares the stats of each of the requested MLModels and stores for each stat
the name of the best model and the corresponding value.
Example usage:
```
law run cf.MLModelOptimizer--version prod1 --ml-folds 0  --ml-models model_1,model_2,model_3
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


class MLModelOptimizer(
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
):
    reqs = Requirements(MLTraining=MLTraining)

    # sandbox = dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_plotting.sh")
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    ml_folds = luigi.Parameter(
        default=[0],  # NOTE: for some reason, the default is not working as intended
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
            "best_model": self.target("dummy.yaml"),
        }

    def run(self):
        model_names = self.ml_models

        # store for each key in the stats dict, which model performed the best (assuming larger=better)
        best_model = {}

        for model_name, inp in zip(model_names, self.input()["models"]):
            stats = inp["collection"][0]["stats"].load(formatter="yaml")

            for key, stat in stats.items():
                if stat > best_model.get(key, (EMPTY_FLOAT, "None"))[0]:
                    best_model[key] = (stat, model_name)

        self.output()["best_model"].dump(best_model, formatter="yaml")
