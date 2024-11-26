# coding: utf-8

"""
Optimization tasks
```
"""

from __future__ import annotations

# import os

import luigi
import law

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    # SelectorMixin,
    # CalibratorsMixin,
    # ProducersMixin,
    # MLModelMixin,
    MLModelTrainingMixin,
)
# from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.util import DotDict
from hbw.tasks.base import HBWTask
from hbw.tasks.ml import PlotMLResultsSingleFold


logger = law.logger.get_logger(__name__)


class GetAUCScores(PlotMLResultsSingleFold):
    """
    This is quite copy-pastey, I just need to produce some AUC scores...
    """
    data_splits = ("test",)

    def run(self):
        # imports
        from hbw.ml.data_loader import MLProcessData
        from hbw.ml.plotting import (
            plot_roc_ovr,
            plot_roc_ovo,
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
        input_files_mlpred = inputs["mlpred"]["test"]["collection"]
        input_files = law.util.merge_dicts(
            *[input_files_preml[key] for key in input_files_preml.keys()],
            *[input_files_mlpred[key] for key in input_files_mlpred.keys()],
            deep=True,
        )
        data = DotDict({
            # "train": MLProcessData(self.ml_model_inst, input_files, "train", self.ml_model_inst.processes, self.fold),
            # "val": MLProcessData(self.ml_model_inst, input_files, "val", self.ml_model_inst.processes, self.fold),
            "test": MLProcessData(self.ml_model_inst, input_files, "test", self.ml_model_inst.processes, self.fold),
        })

        # ROC curves
        plot_roc_ovr(
            self.ml_model_inst,
            data["test"],
            output["plots"],
            "test",
            self.ml_model_inst.process_insts,
            stats,
        )
        plot_roc_ovo(
            self.ml_model_inst,
            data["test"],
            output["plots"],
            "test",
            self.ml_model_inst.process_insts,
            stats,
        )

        # dump all stats into yaml file
        output["stats"].dump(stats, formatter="json")


class Optimizer(
    # NOTE: mixins might need fixing, needs to be tested
    HBWTask,
    MLModelTrainingMixin,
    law.LocalWorkflow,
    # RemoteWorkflow,
):
    """
    Workflow that runs optimization. Needs to be run from within the sandbox
    cf_sandbox venv_ml_plotting
    law run hbw.Optimizer --version prod2 --ml-model dl_22
    """
    sandbox = "bash::$HBW_BASE/sandboxes/venv_ml_plotting.sh"

    iterations = luigi.IntParameter(default=10, description="Number of iterations")
    n_parallel = luigi.IntParameter(default=4, description="Number of parallel evaluations")
    n_initial_points = luigi.IntParameter(default=10, description="Number of random sampled values \
        before starting optimizations")

    @property
    def hyperparameter_space(self):
        """
        Define the hyperparameter space for the Bayesian optimization.
        """
        # TODO: might be nice if we could implement hyperparameter sets as a parameter
        if hasattr(self, "_hyperparameter_space"):
            return self._hyperparameter_space
        from skopt.space import Integer, Real, Categorical  # noqa

        # define the hyperparameter space
        self._hyperparameter_space = {
            "negative_weights": Categorical(["handle", "abs", "ignore"]),
            "activation": Categorical(["relu", "elu"]),
            "learningrate": Real(1e-6, 1e-2, prior="log-uniform"),
            "layer": Integer(32, 1024, prior="log-uniform", base=2),
            # "layer1": Integer(32, 1024, prior="log-uniform", base=2),
            # "layer2": Integer(32, 1024, prior="log-uniform", base=2),
            # "layer3": Integer(32, 1024, prior="log-uniform", base=2),
            "dropout": Real(0.0, 0.5),
            "batchsize": Integer(2 ** 7, 2 ** 14, prior="log-uniform", base=2),
            "reduce_lr_factor": Real(0.1, 1.0),
            "reduce_lr_patience": Integer(1, 10),
            "epochs": Categorical([100]),
        }
        # self._hyperparameter_space = {
        #     "layer": Integer(32, 1024, prior="log-uniform", base=2),
        #     # "dropout": Real(0.0, 0.5),
        # }
        return self._hyperparameter_space

    def create_branch_map(self):
        return list(range(self.iterations))

    def requires(self):
        # NOTE: cache requirements?
        if self.branch == 0:
            return None
        return Optimizer.req(self, branch=self.branch - 1)

    def workflow_requires(self):
        return {}

    def output(self):
        return self.target(f"optimizer_{self.branch}.pkl")

    def run(self):
        import skopt
        optimizer = self.input().load() if self.branch != 0 else skopt.Optimizer(
            dimensions=list(self.hyperparameter_space.values()),
            random_state=42,
            n_initial_points=self.n_initial_points,
        )

        parameter_tuples = optimizer.ask(n_points=self.n_parallel)
        logger.info(f"Optimizing parameters {list(self.hyperparameter_space.keys())}")
        logger.info(f"yielding Objective for sets {parameter_tuples}")
        output = yield Objective.req(
            self,
            parameter_keys=list(self.hyperparameter_space.keys()),
            parameter_tuples=parameter_tuples,
            iteration=self.branch,
            branch=-1,
        )
        y = [f.load()["y"] for f in output["collection"].targets.values()]

        optimizer.tell(parameter_tuples, y)

        print(f"minimum after {self.branch + 1} iterations: {min(optimizer.yi)}")

        with self.output().localize("w") as tmp:
            tmp.dump(optimizer)


class Objective(
    # NOTE: mixins might need fixing, needs to be tested
    HBWTask,
    MLModelTrainingMixin,
    law.LocalWorkflow,
    # RemoteWorkflow,
):
    """
    Objective to optimize.
    """
    parameter_keys = law.CSVParameter()
    parameter_tuples = law.MultiCSVParameter()
    iteration = luigi.IntParameter()

    # upstream requirements
    reqs = Requirements(
        GetAUCScores=GetAUCScores,
    )

    @property
    def parameter_dicts(self):
        """
        Convert parameter sets to dictionaries.
        """
        if hasattr(self, "_parameter_dicts"):
            return self._parameter_dicts

        parameter_dicts = []
        for i, parameter_set in enumerate(self.parameter_tuples):
            parameter_dict = dict(zip(self.parameter_keys, parameter_set))
            if "layer" in parameter_dict:
                # replace singular layer parameter with list of layers
                parameter_dict["layers"] = [parameter_dict.pop("layer")] * 3
            elif "layer1" in parameter_dict:
                # replace layer1, ..., layer{N} with list of layers
                key = "layer1"
                parameter_dict["layers"] = []
                while key in parameter_dicts.keys():
                    parameter_dict["layers"].append(parameter_dict.pop(key))
                    key = f"layer{int(key[-1]) + 1}"

            parameter_dicts.append(parameter_dict)

        self._parameter_dicts = parameter_dicts
        return self._parameter_dicts

    def create_branch_map(self):
        return {i: parameter_dict for i, parameter_dict in enumerate(self.parameter_dicts)}

    def requires(self):
        reqs = {}
        if self.branch == -1:
            return reqs
        reqs["GetAUCScores"] = self.reqs.GetAUCScores.req(
            self,
            fold=0,
            ml_model_settings=self.branch_data,
        )
        return reqs

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["GetAUCScores"] = [self.reqs.GetAUCScores.req(
            self,
            fold=0,
            ml_model_settings=parameter_dict,
            workflow="htcondor",
        ) for parameter_dict in self.parameter_dicts]
        return reqs

    def output(self):
        return self.target(f"objective_{self.iteration:02d}_{self.branch:02d}.json")

    def run(self):
        logger.info("Running Objective Task")

        # load stats
        stats = self.input()["GetAUCScores"]["stats"].load(formatter="json")

        # calculate objective value
        auc_sum = sum(stats.values()) / len(stats.values())
        objective = -auc_sum

        results = {"x": self.branch_data, "y": objective}

        # store results
        self.output().dump(results)


class DummyObjective(
    # NOTE: mixins might need fixing, needs to be tested
    HBWTask,
    MLModelTrainingMixin,
    law.LocalWorkflow,
    # RemoteWorkflow,
):
    """
    Very simple objective to minimize
    """

    parameter_keys = law.CSVParameter()
    parameter_tuples = law.MultiCSVParameter()
    iteration = luigi.IntParameter()

    @property
    def parameter_dicts(self):
        if hasattr(self, "_parameter_dicts"):
            return self._parameter_dicts

        parameter_dicts = []
        for i, parameter_tuple in enumerate(self.parameter_tuples):
            parameter_dict = dict(zip(self.parameter_keys, parameter_tuple))
            # replace singular layer parameter with list of layers
            if "layer" in parameter_dict:
                parameter_dict["layers"] = [parameter_dict.pop("layer")] * 3

            parameter_dicts.append(parameter_dict)

        self._parameter_dicts = parameter_dicts
        return self._parameter_dicts

    def create_branch_map(self):
        return {i: parameter_dict for i, parameter_dict in enumerate(self.parameter_dicts)}

    def output(self):
        return self.target(f"x_{self.iteration}_{self.branch}.json")

    def run(self):
        logger.info("Running Objective Task")

        # load some x value
        x = int(self.branch_data["layers"][0])

        # calculate some objective value (will be minimal at 512)
        y = (512 - x) ** 2
        print(x, y)
        objective = y

        # store results
        self.output().dump({"x": self.branch_data, "y": objective})
