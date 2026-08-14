# coding: utf-8

"""
First implementation of DNN for HH analysis, generalized (TODO)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any
import yaml

import law
import order as od
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd

from columnflow.types import Sequence
from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox, DotDict, DerivableMeta
from columnflow.columnar_util import Route, set_ak_column
from columnflow.config_util import get_datasets_from_process
from hbw.config.dl.variables import add_gatja_scores_variables

from hbw.util import log_memory
from hbw.ml.data_loader import MLDatasetLoader, MLProcessData, input_features_sanity_checks
from hbw.config.processes import prepare_ml_processes

from hbw.tasks.ml import MLPreTraining

np = maybe_import("numpy")
ak = maybe_import("awkward")
pickle = maybe_import("pickle")

logger = law.logger.get_logger(__name__)

import logging
logger = logging.getLogger("luigi-interface")

# patch, allowing user to fall back to old versions
use_old_version = law.config.get_expanded("analysis", "use_old_version", False)


class ClassPropertyDescriptor:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):
        return self.fget(owner)


def classproperty(func):
    return ClassPropertyDescriptor(func)


class MLClassifierBase(MLModel):
    """
    Provides a base structure to implement Multiclass Classifier in Columnflow
    """
    # flag denoting whether the preparation_producer is invoked before evaluate()
    preparation_producer_in_ml_evaluation: bool = False

    # set some defaults, can be overwritten by subclasses or via cls_dict
    # NOTE: the order of processes is crucial! Do not change after training
    _default__processes: tuple = ("tt", "st")
    train_nodes: dict = {
        "tt": {"ml_id": 0},
        "st": {"ml_id": 1},
    }

    input_features: set = {"mli_ht", "mli_n_jet"}

    # name of the PreparationProducer class that is used to prepare the input features
    # this is also used to determine the preml_store_name
    preparation_producer_name: str = "prepml"

    @classproperty
    def store_name(cls) -> str:
        return cls.preparation_producer_name

    # Class for data loading and it's dependencies.
    data_loader = MLDatasetLoader
    # NOTE: we might want to use the data_loader.hyperparameter_deps instead
    preml_params: set[str] = {"data_loader", "input_features", "train_val_test_split"}

    # NOTE: we split each fold into train, val, test + do k-folding, so we have a 4-way split in total
    # TODO: test whether setting "test" to 0 is working
    train_val_test_split: tuple = (0.75, 0.15, 0.10)
    folds: int = 5

    # training-specific parameters. Only need to re-run training when changing these
    _default__class_factors: dict = {"st": 1, "tt": 1}
    _default__sub_process_class_factors: dict = {"st": 2, "tt": 1}
    _default__negative_weights: str = "handle"
    _default__epochs: int = 50
    _default__batchsize: int = 2 ** 10

    # parameters to add into the `parameters` attribute to determine the 'parameters_repr' and to store in a yaml file
    bookkeep_params: set[str] = {
        "data_loader", "input_features", "train_val_test_split",
        "processes", "train_nodes", "class_factors", "sub_process_class_factors",
        "negative_weights", "epochs", "batchsize", "folds",
    }

    # parameters that can be overwritten via command line
    settings_parameters: set[str] = {
        "processes", "class_factors", "sub_process_class_factors",
        "negative_weights", "epochs", "batchsize",
    }

    @classmethod
    def derive(
        cls,
        cls_name: str,
        bases: tuple = (),
        cls_dict: dict[str, Any] | None = None,
        module: str | None = None,
    ):
        """
        derive but rename classattributes included in settings_parameters to "_default__{attr}"
        """
        if cls_dict:
            for attr, value in cls_dict.copy().items():
                if attr in cls.settings_parameters:
                    cls_dict[f"_default__{attr}"] = cls_dict.pop(attr)

        return DerivableMeta.derive(cls, cls_name, bases, cls_dict, module)

    def __init__(
            self,
            *args,
            folds: int | None = None,
            **kwargs,
    ):
        """
        Initialization function of the MLModel. We first set properties using values from the
        *self.parameters* dictionary that is obtained via the `--ml-model-settings` parameter. If
        the parameter is not set via command line,cthe "_default__{attr}" classattribute is used as
        fallback. Then we cast the parameters to the correct types and store them as individual
        class attributes. Finally, we store the parameters in the `self.parameters` attribute,
        which is used both to create a hash for the output path and to store the parameters in a yaml file.

        Only the parameters in the `settings_parameters` attribute can be overwritten via the command line.
        Only the parameters in the `bookkeep_params` attribute are stored in the `self.parameters` attribute.
        Parameters defined in the `settings_parameters` must be named "_default__{attr}" in the main class definition.
        When deriving, the "_default__" is automatically added.
        Similarly, a parameter starting with "_default__" must be part of the `settings_parameters`.
        """
        super().__init__(*args, **kwargs)
        # logger.warning("Running MLModel init")

        # checks
        if diff := self.settings_parameters.difference(self.bookkeep_params):
            raise Exception(
                f"settings_parameters {diff} not in bookkeep_params; all customizable settings should"
                "be bookkept in the parameters.yaml file and the self.parameters_repr to ensure reproducibility",
            )
        if diff := self.preml_params.difference(self.bookkeep_params):
            raise Exception(
                f"preml_params {diff} not in bookkeep_params; all parameters that change the preml_store_name"
                "should be bookkept via the 'self.bookkeep_params' attribute",
            )
        if unknown_params := set(self.parameters.keys()).difference(self.settings_parameters):
            raise Exception(
                f"unknown parameters {unknown_params} passed to the MLModel; only the following "
                f"parameters are allowed: {', '.join(self.settings_parameters)}",
            )

        for param in self.settings_parameters:
            # param is not allowed to exist on class level
            if hasattr(self, param):
                raise ValueError(
                    f"{self.cls_name} has classatribute {param} (value: {getattr(self, param)}) on class level "
                    "but also requests it as configurable via settings_parameters. Maybe you have to rename the",
                    "classattribute in some base class to '_default__{param}'?",
                )
            # set to requested value, fallback on "__default_{param}"
            setattr(self, param, self.parameters.get(param, getattr(self, f"_default__{param}")))

        # check that all _default__ attributes are taken care of
        for attr in dir(self):
            if not attr.startswith("_default__"):
                continue
            if not hasattr(self, attr.replace("_default__", "", 1)):
                raise ValueError(
                    f"{self.cls_name} has classatribute {attr} but never sets corresponding property",
                )

        # cast the ml parameters to the correct types if necessary
        self.cast_ml_param_values()

        # overwrite self.parameters with the typecasted values
        for param in self.bookkeep_params:
            self.parameters[param] = getattr(self, param)
            if isinstance(self.parameters[param], set):
                # sets are not hashable, so convert them to sorted tuple
                self.parameters[param] = tuple(sorted(self.parameters[param]))

        # sort the self.settings_parameters
        self.parameters = DotDict(sorted(self.parameters.items()))

        # sanity check: for each process in "train_nodes", we need to have 1 process with "ml_id" in config

    def cast_ml_param_values(self):
        """
        Resolve the values of the parameters that are used in the MLModel
        """
        self.processes = tuple(self.processes)
        self.input_features = set(self.input_features)
        self.train_val_test_split = tuple(self.train_val_test_split)
        if not isinstance(self.sub_process_class_factors, dict):
            # cast tuple to dict
            self.sub_process_class_factor = {
                proc: weight for proc, weight in [s.split(":") for s in self.sub_process_class_factor]
            }
        # cast weights to int and remove processes not used in training
        self.ml_model_weights = {
            proc: int(weight)
            for proc, weight in self.sub_process_class_factors.items()
            if proc in self.processes
        }
        self.negative_weights = str(self.negative_weights)
        self.epochs = int(self.epochs)
        self.batchsize = int(self.batchsize)
        self.folds = int(self.folds)

        # checks
        if self.negative_weights not in ("ignore", "abs", "handle"):
            raise Exception(
                f"negative_weights {self.negative_weights} not in ('ignore', 'abs', 'handle')",
            )

    @property
    def preml_store_name(self):
        """
        Create a hash of the parameters that are used in the MLModel to determine the 'preml_store_name'.
        The preml_store_name is cached to ensure that it does not change during the lifetime of the object.
        """
        preml_params = {param: self.parameters[param] for param in self.preml_params}
        preml_store_name = law.util.create_hash(sorted(preml_params.items()))
        if hasattr(self, "_preml_store_name") and self._preml_store_name != preml_store_name:
            raise Exception(
                f"preml_store_name changed from {self._preml_store_name} to {preml_store_name};"
                "this should not happen",
            )
        self._preml_store_name = preml_store_name
        return self._preml_store_name

    @property
    def parameters_repr(self):
        """
        Create a hash of the parameters to store as part of the output path.
        The repr is cached to ensure that it does not change during the lifetime of the object.
        """
        if use_old_version:
            return {
                "ggfv1": "a07e93e269",
                "vbfv1": "8031129333",
                "multiclassv1": "bd236d50b5",
            }[self.cls_name]
        if not self.parameters:
            return ""
        parameters_repr = law.util.create_hash(sorted(self.parameters.items()))
        if hasattr(self, "_parameters_repr") and self._parameters_repr != parameters_repr:
            raise Exception(
                f"parameters_repr changed from {self._parameters_repr} to {parameters_repr};"
                "this should not happen",
            )
        self._parameters_repr = parameters_repr
        return self._parameters_repr

    from hbw.util import timeit_multiple

    def valid_ml_id_sanity_check(self):
        """
        ml_ids must include 0 and each following integer up to the number of requested train_nodes
        """
        for p in self.process_insts:
            sub_process_class_factor = p.x("sub_process_class_factor", None)
            if sub_process_class_factor is None:
                logger.warning(f"Process {p.name} has no 'sub_process_class_factor' aux; will be set to 1.")
                p.x.sub_process_class_factor = 1
            ml_id = p.x("ml_id", None)
            if ml_id is None:
                logger.warning(f"Process {p.name} has no 'ml_id' aux; will be set to -1.")
                p.x.ml_id = -1

        ml_ids = sorted(set(p.x.ml_id for p in self.process_insts) - {-1})

        if len(ml_ids) != len(self.train_nodes.keys()):
            raise Exception(f"ml_ids {ml_ids} does not match number of requested train_nodes {self.train_nodes.keys()}")

        expected_id = 0
        while ml_ids:
            _id = ml_ids.pop(0)
            if _id == expected_id:
                # next id should be previous value + 1
                expected_id += 1
                continue
            else:
                raise ValueError(f"Invalid combination of ml ids {set(p.x.ml_id for p in self.process_insts)}")

        logger.debug("ml_id_sanity_check passed")

    # @timeit_multiple
    def setup(self) -> None:
        """ function that is run as part of the setup phase. Most likely overwritten by subclasses """
        if self.config_inst.has_tag(f"{self.cls_name}_called"):
            # call this function only once per config
            return
        logger.debug(
            f"Setting up MLModel {self.cls_name} (parameter hash: {self.parameters_repr}), "
            f"parameters: \n{self.parameters}",
        )
        # dynamically add processes and variables for the quantities produced by this model
        # NOTE: this function might not be called for all configs when the requested configs
        # between MLTraining and the requested task are different

        # setup processes for training
        # NOTE: this function needs to be called per config, but there are still some issues here.
        prepare_ml_processes(self.config_inst, self.train_nodes, self.sub_process_class_factors)
        self.valid_ml_id_sanity_check()

        # setup variables
        # for proc in self.processes:
        for proc, node_config in self.train_nodes.items():
            x_title = f"DNN output score ${node_config.get('label', proc)}$"
            if len(self.train_nodes) <= 2:
                # binary NN
                x_title = f"${node_config.get('label', proc)}$ binary NN score"
            for config_inst in self.config_insts:
                if f"mlscore.{proc}" not in config_inst.variables:
                    config_inst.add_variable(
                        name=f"mlscore.{proc}",
                        expression=f"mlscore.{proc}",
                        null_value=-1,
                        binning=(40, 0., 1.),
                        x_title=x_title,
                    )
                    config_inst.add_variable(
                        name=f"rebinlogit_mlscore.{proc}",
                        expression=lambda events, proc=proc: np.log(events.mlscore[proc] / (1 - events.mlscore[proc])),
                        null_value=-1,
                        binning=(40, -10., 10.),
                        x_title=f"logit({x_title})",
                        aux={
                            "inputs": {f"mlscore.{proc}"},
                        },
                    )
                    config_inst.add_variable(
                        name=f"logit_mlscore.{proc}",
                        expression=lambda events, proc=proc: np.log(events.mlscore[proc] / (1 - events.mlscore[proc])),
                        null_value=-1,
                        binning=(1000, -2., 10.),
                        x_title=f"logit({x_title})",
                        aux={
                            "inputs": {f"mlscore.{proc}"},
                            "rebin": 25,
                            "rebin_config": {
                                "processes": [proc],
                                "n_bins": 4,
                            },
                        },  # automatically rebin to 40 bins for plotting tasks
                    )
                    config_inst.add_variable(
                        name=f"rebinned_logit2b.{proc}",  # used in histProducer specificially accessing this name to set weight=1  # noqa E501
                        expression=lambda events, proc=proc: np.log(events.mlscore[proc] / (1 - events.mlscore[proc])),
                        binning=[-2.0, -1.112, -0.488, 0.028000000000000025, 0.496, 0.964, 1.48, 2.0920000000000005, 3.088, 10.0],  # noqa E501
                        x_title="logit score rebinned",
                        aux={
                            "inputs": {f"mlscore.{proc}"},
                        },
                    )
                    config_inst.add_variable(
                        name=f"rebinned_logit3b.{proc}",  # used in histProducer specificially accessing this name to set weight=1   # noqa E501
                        expression=lambda events, proc=proc: np.log(events.mlscore[proc] / (1 - events.mlscore[proc])),
                        binning=[-2.0, 0.31599999999999984, 0.7840000000000003, 1.12, 1.432, 1.7080000000000002, 2.0200000000000005, 2.3440000000000003, 2.752, 3.232, 10.0],  # noqa E501
                        x_title="logit score rebinned",
                        aux={
                            "inputs": {f"mlscore.{proc}"},
                        },
                    )
                    config_inst.add_variable(
                        name=f"rebinned_logit4b.{proc}",  # used in histProducer specificially accessing this name to set weight=1  # noqa E501
                        expression=lambda events, proc=proc: np.log(events.mlscore[proc] / (1 - events.mlscore[proc])),
                        binning=[-2.0, 1.7320000000000002, 2.1879999999999997, 2.5360000000000005, 2.8360000000000003, 3.1000000000000005, 3.364, 3.652, 4.072, 4.708, 10.0],  # noqa E501
                        x_title="logit score rebinned",
                        aux={
                            "inputs": {f"mlscore.{proc}"},
                        },
                    )

        # add tag to allow running this function just once
        self.config_inst.add_tag(f"{self.cls_name}_called")

    @property
    def process_insts(self):
        if hasattr(self, "_process_insts"):
            return self._process_insts
        return [self.config_inst.get_process(proc) for proc in self.processes]

    @property
    def train_node_process_insts(self):
        if hasattr(self, "_train_node_process_insts"):
            return self._train_node_process_insts
        return [self.config_inst.get_process(proc) for proc in self.train_nodes.keys()]

    def preparation_producer(self: MLModel, analysis_inst: od.Analysis):
        """ producer that is run as part of PrepareMLEvents and MLEvaluation (before `evaluate`) """
        return self.preparation_producer_name

    def training_calibrators(self, analysis_inst: od.Analysis, requested_calibrators: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        # NOTE: since automatic resolving is not working here, we do it ourselves
        return requested_calibrators or [analysis_inst.x.default_calibrator]

    def training_producers(self, analysis_inst: od.Analysis, requested_producers: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        # NOTE: might be nice to keep the "pre_ml_cats" for consistency, but running two
        # categorization Producers in the same workflow is messy, so we skip it for now
        # return requested_producers or ["event_weights", "pre_ml_cats", analysis_inst.x.ml_inputs_producer]
        # return requested_producers or ["event_weights", analysis_inst.x.ml_inputs_producer]
        return ["event_weights", analysis_inst.x.ml_inputs_producer]

    def evaluation_producers(self, analysis_inst: od.Analysis, requested_producers: Sequence[str]) -> list[str]:
        # NOTE: there is still an issue that this can only remove (not add) Producers, so the
        # ml_inputs_producer also needs to be added in all task calls that use the evaluation of this model
        if use_old_version:
            return ["event_weights", analysis_inst.x.ml_inputs_producer]
        return [analysis_inst.x.ml_inputs_producer]

    def requires(self, task: law.Task) -> dict[str, Any]:
        # Custom requirements (none currently)
        reqs = {}

        reqs["preml"] = MLPreTraining.req_different_branching(task, branch=-1)
        return reqs

    def sandbox(self, task: law.Task) -> str:
        # venv_ml_tf sandbox but with scikit-learn and restricted to tf 2.11.0
        return dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_plotting.sh")

    def datasets(self, config_inst: od.Config) -> set[od.Dataset]:
        used_datasets = set()
        for i, proc in enumerate(self.processes):
            if not config_inst.has_process(proc):
                raise Exception(f"Process {proc} not included in the config {config_inst.name}")

            proc_inst = config_inst.get_process(proc)
            # NOTE: this info is accessible during training but probably not afterwards in other tasks
            # --> move to setup? or store in some intermediate output file?
            # proc_inst.x.ml_id = i
            # proc_inst.x.sub_process_class_factor = self.sub_process_class_factors.get(proc, 1)

            # get datasets corresponding to this process
            dataset_insts = [
                dataset_inst for dataset_inst in
                get_datasets_from_process(config_inst, proc, strategy="all", only_first=False)
            ]

            # store assignment of datasets and processes in the instances
            for dataset_inst in dataset_insts:
                dataset_inst.x.ml_process = proc
            proc_inst.x.ml_datasets = [dataset_inst.name for dataset_inst in dataset_insts]

            # check that no dataset is used multiple times
            if datasets_already_used := used_datasets.intersection(dataset_insts):
                raise Exception(f"{datasets_already_used} datasets are used for multiple processes")
            used_datasets |= set(dataset_insts)
        return used_datasets

    def uses(self, config_inst: od.Config) -> set[Route | str]:
        # if not all(var.startswith("mli_") for var in self.input_features):
        #     raise Exception(
        #         "We currently expect all input_features to start with 'mli_', which is not the case"
        #         f"for one of the variables in the 'input_features' {self.input_features}",
        #     )
        # include all variables starting with 'mli_' to enable reusing MergeMLEvents outputs
        columns = {"mli_*"}
        # TODO: switch to full event weight
        # TODO: this might not work with data, to be checked
        columns.add("process_id")
        columns.add("normalization_weight")
        columns.add("stitched_normalization_weight")
        columns.add("event_weight")
        return columns

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        produced = set()
        for proc in self.train_nodes.keys():
            produced.add(f"mlscore.{proc}")

        return produced

    def output(self, task: law.Task) -> dict[str, law.FileSystemTarget]:

        # declare the main target
        target = task.target(f"mlmodel_f{task.branch}of{self.folds}", dir=True)

        outp = {
            "mlmodel": target,
            "mlmodel_file": target.child("mlmodel.keras", type="f", optional=True),
            "plots": target.child("plots", type="d", optional=True),
            # "checkpoint": target.child("checkpoint", type="d", optional=True),
            "checkpoint": target.child("checkpoint.model.keras", type="f", optional=True),
        }

        # # define all files that need to be present
        # outp["required_files"] = [
        #     target.child(fname, type="f") for fname in
        #     ("saved_model.pb", "keras_metadata.pb", "fingerprint.pb", "parameters.yaml", "input_features.pkl")
        # ]
        outp["required_files"] = [
            target.child(fname, type="f") for fname in
            ("mlmodel.keras", "parameters.yaml", "input_features.pkl")
        ]
        return outp

    def open_model(self, target: law.LocalDirectoryTarget) -> dict[str, Any]:
        import tensorflow as tf

        models = {}

        models["input_features"] = tuple(target["mlmodel"].child(
            "input_features.pkl", type="f",
        ).load(formatter="pickle"))

        # NOTE: we cannot use the .load method here, because it's unable to read tuples etc.
        #       should check that this also works when running remote
        with open(target["mlmodel"].child("parameters.yaml", type="f").fn) as f:
            f_in = f.read()
        models["parameters"] = yaml.load(f_in, Loader=yaml.Loader)

        # custom loss needed due to output layer changes for negative weights
        # from hbw.ml.tf_util import cumulated_crossentropy
        models["model"] = tf.keras.models.load_model(
            target["mlmodel_file"].abspath,
            # custom_objects={cumulated_crossentropy.__name__: cumulated_crossentropy},
        )
        models["best_model"] = tf.keras.models.load_model(
            target["checkpoint"].abspath,
            # custom_objects={cumulated_crossentropy.__name__: cumulated_crossentropy},
        )

        return models

    def load_data(
        self,
        task: law.Task,
        input: Any,
        output: law.LocalDirectoryTarget,
    ):
        # we need to call this function for some process config setup
        self.datasets(self.config_inst)

        input_files = input["model"]["preml"]["collection"]
        input_files = law.util.merge_dicts(*[input_files[key] for key in input_files.keys()], deep=True)
        train = DotDict(
            {proc_inst: MLProcessData(
                self, input_files, "train", [proc_inst.name], task.fold,
            ) for proc_inst in self.process_insts},
        )
        for proc_data in train.values():
            # load into memory
            proc_data.load_all()

        log_memory("loading train data")

        validation = DotDict(
            {proc_inst: MLProcessData(
                self, input_files, "val", [proc_inst.name], task.fold,
            ) for proc_inst in self.process_insts},
        )
        for proc_data in validation.values():
            # load into memory
            proc_data.load_all()

        log_memory("loading validation data")

        # store input features as an output
        output["mlmodel"].child("input_features.pkl", type="f").dump(self.input_features_ordered, formatter="pickle")

        return train, validation

    def train(
        self,
        task: law.Task,
        input: Any,
        output: law.LocalDirectoryTarget,
    ) -> ak.Array:
        """ Training function that is called during the MLTraining task """
        import tensorflow as tf
        log_memory("start")
        # np.random.seed(1337)  # for reproducibility

        physical_devices = tf.config.list_physical_devices("GPU")
        logger.warning(f"Found {len(physical_devices)} physical GPU devices: {physical_devices}")
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

        # input preparation
        train, validation = self.load_data(task, input, output)

        # hyperparameter bookkeeping
        output["mlmodel"].child("parameters.yaml", type="f").dump(dict(self.parameters), formatter="yaml")
        logger.info(f"Training will be run with the following parameters: \n{self.parameters}")
        #
        # model preparation
        #

        model = self.prepare_ml_model(task)
        logger.info(model.summary())
        log_memory("prepare-model")

        #
        # training
        #
        self.fit_ml_model(task, model, train, validation, output)
        log_memory("training")
        # save the model and history; TODO: use formatter
        # output.dump(model, formatter="tf_keras_model")
        model.save(output["mlmodel_file"].abspath)

        return

    @abstractmethod
    def prepare_ml_model(
        self,
        task: law.Task,
    ):
        """ Function to define the ml model. Needs to be implemented in daughter class """
        return

    @abstractmethod
    def fit_ml_model(
        self,
        task: law.Task,
        model,
        train: DotDict[np.array],
        validation: DotDict[np.array],
        output: law.LocalDirectoryTarget,
    ) -> None:
        """ Function to run the ml training loop. Needs to be implemented in daughter class """
        return

    def patch_events(self, events):
        from columnflow.columnar_util import fill_at
        # TODO: this function is currently copy-pasted from MLPreTraining task
        # change padding value to -1 for btag scores
        # for col in (
        #     "mli_fj_particleNetWithMass_HbbvsQCD",
        #     "mli_fj_particleNet_XbbVsQCD",
        # ):
        #     events = fill_at(events, events[col] == -10, col, -1, value_type=np.float32)

        return events

    def evaluate(
        self,
        task: law.Task,
        events: ak.Array,
        models: list(Any),
        fold_indices: ak.Array,
        events_used_in_training: bool = True,
    ) -> None:
        """
        Evaluation function that is run as part of the MLEvaluation task
        """
        use_best_model = False  # TODO ML, hier auf True setzen?

        if len(events) == 0:
            logger.warning(f"Dataset {task.dataset} is empty. No columns are produced.")
            return events

        events = self.patch_events(events)

        # check that the input features are the same for all models
        for model in models:
            input_features_sanity_checks(self, model["input_features"])

        process = task.dataset_inst.x("ml_process", task.dataset_inst.processes.get_first().name)
        process_inst = task.config_inst.get_process(process)
        node_processes = list(self.train_nodes.keys())

        ml_dataset = self.data_loader(self, process_inst, events, skip_mask=True)

        # # store the ml truth label in the events
        # events = set_ak_column(
        #     events, f"{self.cls_name}.ml_truth_label",
        #     ml_dataset.labels,
        # )

        # check that all MLTrainings were started with the same set of parameters
        parameters = [model["parameters"] for model in models]
        from hbw.util import dict_diff
        for i, params in enumerate(parameters[1:]):
            if params != parameters[0]:
                diff = dict_diff(params, parameters[0])
                raise Exception(
                    "The MLTraining parameters (see 'parameters.yaml') from "
                    f"fold {i} differ from fold 0; diff: {diff}",
                )

        if use_best_model:
            models = [model["best_model"] for model in models]
        else:
            models = [model["model"] for model in models]

        # do prediction for all models and all inputs
        predictions = []
        for i, model in enumerate(models):
            # NOTE: the next line triggers some warning concering tf.function retracing
            pred = ak.from_numpy(model.predict_on_batch(ml_dataset.features))
            if len(pred[0]) != len(node_processes):
                raise Exception("Number of output nodes should be equal to number of processes")
            predictions.append(pred)
            # store predictions for each model
            for j, proc in enumerate(node_processes):
                events = set_ak_column(
                    events, f"fold{i}_mlscore.{proc}", pred[:, j],
                )

        # combine all models into 1 output score, using the model that has not yet seen the test set
        outputs = ak.where(ak.ones_like(predictions[0]), -1, -1)
        for i in range(self.folds):
            logger.info(f"Evaluation fold {i}")
            # reshape mask from N*bool to N*k*bool (TODO: simpler way?)
            idx = ak.to_regular(ak.concatenate([ak.singletons(fold_indices == i)] * len(node_processes), axis=1))
            outputs = ak.where(idx, predictions[i], outputs)

        # sanity check of the number of output nodes
        if len(outputs[0]) != len(node_processes):
            raise Exception(
                f"The number of output nodes {len(outputs[0])} should be equal to "
                f"the number of processes {len(node_processes)}",
            )

        for proc, node_config in self.train_nodes.items():
            events = set_ak_column(
                events, f"mlscore.{proc}", outputs[:, node_config["ml_id"]],
            )

        return events


class ExampleDNN(MLClassifierBase):
    """ Example class how to implement a DNN from the MLClassifierBase """

    # optionally overwrite input parameters
    _default__epochs: int = 10

    def prepare_ml_model(
        self,
        task: law.Task,
    ):
        """
        Minimal implementation of a ML model
        """
        import tensorflow.keras as keras

        from keras.models import Sequential
        from keras.layers import Dense, BatchNormalization
        from hbw.ml.tf_util import cumulated_crossentropy

        n_inputs = len(set(self.input_features))
        n_outputs = len(self.processes)

        model = Sequential()

        # input layer
        model.add(BatchNormalization(input_shape=(n_inputs,)))

        # hidden layers
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=64, activation="relu"))

        # output layer
        model.add(Dense(n_outputs, activation="softmax"))

        # compile the network
        # NOTE: the custom loss needed due to output layer changes for negative weights
        optimizer = keras.optimizers.Adam(learning_rate=0.00050)
        if self.negative_weights == "ignore":
            model.compile(
                loss="categorical_crossentropy",
                optimizer=optimizer,
                weighted_metrics=["categorical_accuracy"],
            )
        else:
            model.compile(
                loss=cumulated_crossentropy,
                optimizer=optimizer,
                weighted_metrics=["categorical_accuracy"],
            )

        return model

    def fit_ml_model(
        self,
        task: law.Task,
        model,
        train: DotDict[np.array],
        validation: DotDict[np.array],
        output: law.LocalDirectoryTarget,
    ) -> None:
        """
        Minimal implementation of training loop.
        """
        import tensorflow as tf
        from hbw.ml.tf_util import MultiDataset

        # with tf.device("CPU"):
        tf_train = MultiDataset(data=train, batch_size=self.batchsize, kind="train")
        tf_validation = tf.data.Dataset.from_tensor_slices(
            (validation.features, validation.target, validation.ml_weights),
        ).batch(self.batchsize)

        logger.info("Starting training...")
        model.fit(
            (x for x in tf_train),
            validation_data=tf_validation,
            # steps_per_epoch=tf_train.max_iter_valid,
            steps_per_epoch=tf_train.iter_smallest_process,
            epochs=self.epochs,
            verbose=2,
        )


# dervive another model from the ExampleDNN class with different class attributes
example_test = ExampleDNN.derive("example_test", cls_dict={"epochs": 5})


# load all ml modules here
if law.config.has_option("analysis", "ml_modules"):
    for m in law.config.get_expanded("analysis", "ml_modules", [], split_csv=True):
        logger.debug(f"loading ml module '{m}'")
        maybe_import(m.strip())


class GatjaTraining_MLClassifierBase_small_class_weights_05(MLClassifierBase):
    cls_name = "GatjaTraining_MLClassifierBase_small_class_weights_05"
    ml_inputs_producer_name = "gatja_inputs_jet_based_plus_b_jet_inputs_corrected_Higgs_Index_discrete_b"
    preparation_producer_name = "gatja_prepml"

    _default__seed: int = 1

    settings_parameters = MLClassifierBase.settings_parameters | {"seed"}
    bookkeep_params = MLClassifierBase.bookkeep_params | {"seed"}

    def cast_ml_param_values(self):
        super().cast_ml_param_values()
        self.seed = int(self.seed)

    def training_producers(self, analysis_inst, requested_producers):
        logger.warning(f"GATJA training_producers: {self.cls_name}")
        return ["gatja_event_weight", self.ml_inputs_producer_name]

    def evaluation_producers(self, analysis_inst, requested_producers: Sequence[str]) -> list[str]:
        return [self.ml_inputs_producer_name]

    _default__processes = (
        # "tt", "ttbb_custom",
        "hhh_4b2w_2l2nu_c30_d40",
        "tthh_4b",
        # "ttb_custom",
        # "tt2b_custom",
        "ttbb_custom",
        "tt_custom",
        "tth",
        # "ttzz",
        # "ttzh",
    )

    train_nodes = {
        "hhh_4b2w_2l2nu_c30_d40": {"ml_id": 0},
        "tthh_4b": {"ml_id": 1},
        "ttbb_custom": {"ml_id": 2},
        "tt_custom": {"ml_id": 3},
        "tth": {"ml_id": 4},
    }

    _default__class_factors = {"hhh_4b2w_2l2nu_c30_d40": 1, "tthh_4b": 1, "ttbb_custom": 1, "tt_custom": 1, "tth": 1}
    _default__sub_process_class_factors = {
        "hhh_4b2w_2l2nu_c30_d40": 1, "tthh_4b": 1, "ttbb_custom": 1, "tt_custom": 1, "tth": 1,
    }

    jet_classes = ("higgs", "top", "other")
    n_jets = 8
    padding_value = -6.0

    store_name = "test_v2"

    input_features = [
        "jetPT1", "jetPT2", "jetPT3", "jetPT4", "jetPT5", "jetPT6", "jetPT7", "jetPT8",
        "jetEta1", "jetEta2", "jetEta3", "jetEta4", "jetEta5", "jetEta6", "jetEta7", "jetEta8",
        "jetPhi1", "jetPhi2", "jetPhi3", "jetPhi4", "jetPhi5", "jetPhi6", "jetPhi7", "jetPhi8",
        "bjetAverageMass", "jetAverageMass",
        "bjetAverageMassSqr", "jetNumber", "bjetNumber",
        "minDeltaRbb", "btag_weight", "weights",
        "jetHT", "bjetHT", "lightjetHT",
        "leptonPT1", "leptonEta1", "leptonPhi1",
        "leptonPT2", "leptonEta2", "leptonPhi2",
        "met", "metPhi",
        "jetMinChiHiggsIndex1", "jetSecMinChiHiggsIndex1", "jetMinChiHiggsIndex2", "jetSecMinChiHiggsIndex2",
        "jetMinChiHiggsIndex3", "jetSecMinChiHiggsIndex3", "jetMinChiHiggsIndex4",
        "jetSecMinChiHiggsIndex4", "jetMinChiHiggsIndex5", "jetSecMinChiHiggsIndex5", "jetMinChiHiggsIndex6",
        "jetSecMinChiHiggsIndex6", "jetMinChiHiggsIndex7",
        "jetSecMinChiHiggsIndex7", "jetMinChiHiggsIndex8", "jetSecMinChiHiggsIndex8",
        "jetBTagDisc1", "jetBTagDisc2", "jetBTagDisc3", "jetBTagDisc4",
        "jetBTagDisc5", "jetBTagDisc6", "jetBTagDisc7", "jetBTagDisc8",
        "jetTopMatched1", "jetTopMatched2", "jetTopMatched3", "jetTopMatched4",
        "jetTopMatched5", "jetTopMatched6", "jetTopMatched7", "jetTopMatched8",
        "jetHiggsMatched1", "jetHiggsMatched2", "jetHiggsMatched3", "jetHiggsMatched4",
        "jetHiggsMatched5", "jetHiggsMatched6", "jetHiggsMatched7", "jetHiggsMatched8",
    ]

    train_val_test_split = (0.75, 0.15, 0.1)
    folds = 5
    _default__epochs = 300
    _default__batchsize = 2048
    _default__negative_weights = "handle"

    stage_one_index_node: int = 5
    stage_one_index_neigh1: int = 4
    stage_one_index_neigh2: int = 4
    initial_lr = 1e-4
    warmup_epochs = 2
    warmup_lr = 1e-9

    model_feature_order = (
        "btag_weight",
        "jetPT1", "jetPT2", "jetPT3", "jetPT4", "jetPT5", "jetPT6", "jetPT7", "jetPT8",
        "jetEta1", "jetEta2", "jetEta3", "jetEta4", "jetEta5", "jetEta6", "jetEta7", "jetEta8",
        "jetPhi1", "jetPhi2", "jetPhi3", "jetPhi4", "jetPhi5", "jetPhi6", "jetPhi7", "jetPhi8",
        "jetMinChiHiggsIndex1", "jetMinChiHiggsIndex2", "jetMinChiHiggsIndex3", "jetMinChiHiggsIndex4",
        "jetMinChiHiggsIndex5", "jetMinChiHiggsIndex6", "jetMinChiHiggsIndex7", "jetMinChiHiggsIndex8",
        "jetBTagDisc1", "jetBTagDisc2", "jetBTagDisc3", "jetBTagDisc4",
        "jetBTagDisc5", "jetBTagDisc6", "jetBTagDisc7", "jetBTagDisc8",
        "jetHT", "bjetHT", "lightjetHT",
        "jetNumber", "bjetNumber", "jetAverageMass",
        "leptonPT1", "leptonEta1", "leptonPhi1",
        "leptonPT2", "leptonEta2", "leptonPhi2",
        "met", "metPhi",
        "jetSecMinChiHiggsIndex1", "jetSecMinChiHiggsIndex2", "jetSecMinChiHiggsIndex3", "jetSecMinChiHiggsIndex4",
        "jetSecMinChiHiggsIndex5", "jetSecMinChiHiggsIndex6", "jetSecMinChiHiggsIndex7", "jetSecMinChiHiggsIndex8",
    )

    graph_feature_names = (
        "btag_weight",
        "node_pt", "node_eta", "node_phi", "node_minChiIdx", "node_btagDisc",
        "jetHT", "jetNumber", "jetAverageMass",
        "leptonPT1", "leptonEta1", "leptonPhi1",
        "leptonPT2", "leptonEta2", "leptonPhi2", "met",
        "n1_pt", "n1_eta", "n1_phi", "n1_btagDisc",
        "n2_pt", "n2_eta", "n2_phi", "n2_btagDisc",
    )

    @property
    def n_features(self) -> int:
        return len(self.graph_feature_names) - 1

    def uses(self, config_inst) -> set:
        columns = set(self.input_features)
        columns |= {
            "process_id",
            "normalization_weight",
            # "stitched_normalization_weight",
            "event_weight",
        }
        return columns

    def produces(self, config_inst) -> set:
        return {f"gatja_output_{i}" for i in range(self.n_jets * len(self.jet_classes))}

    def setup(self) -> None:
        add_gatja_scores_variables(self.config_inst)
        if self.config_inst.has_tag(f"{self.cls_name}_called"):
            return

        prepare_ml_processes(self.config_inst, self.train_nodes, self.sub_process_class_factors)
        self.valid_ml_id_sanity_check()

        for config_inst in self.config_insts:
            for i in range(self.n_jets * len(self.jet_classes)):
                name = f"gatja_output_{i}"
                if name not in config_inst.variables:
                    jet_slot = i // len(self.jet_classes) + 1
                    jet_class = self.jet_classes[i % len(self.jet_classes)]
                    config_inst.add_variable(
                        name=name,
                        expression=name,
                        null_value=-10,
                        binning=(40, 0., 1.),
                        x_title=f"GATJA {jet_class} score (jet {jet_slot})",
                    )

        self.config_inst.add_tag(f"{self.cls_name}_called")

    _downsample = {"tt_custom": 0.1}

    def _proc_data_to_dataframe(self, proc_data, process_name: str) -> "pd.DataFrame":
        import pandas as pd
        df = pd.DataFrame(
            np.asarray(proc_data.features),
            columns=list(proc_data.input_features),
        )
        frac = self._downsample.get(process_name, 1.0)
        if frac < 1.0:
            rng = np.random.default_rng(abs(hash(process_name)) % 2**32)
            df = df.loc[rng.random(len(df)) < frac].reset_index(drop=True)

        df["process"] = process_name
        return df

    def _safe_lookup(self, frame: "pd.DataFrame", row_labels: Sequence[int], column_names: Sequence[str]) -> np.ndarray:
        if len(row_labels) == 0:
            return np.array([], dtype=float)
        subset = frame.loc[row_labels]
        column_index = subset.columns.get_indexer(column_names)
        if np.any(column_index < 0):
            missing = [column_names[index] for index, value in enumerate(column_index) if value < 0]
            raise KeyError(f"Missing neighbour columns: {missing}")
        row_index = np.arange(len(row_labels))
        return subset.to_numpy()[row_index, column_index]

    def compute_padding_mask(self, working: "pd.DataFrame", index: int) -> np.ndarray:
        slot = index + 1
        by_count = (slot > working["jetNumber"].to_numpy())
        return by_count

    def _create_graphs_core(self, df: "pd.DataFrame", index: int, drop_empty: bool = True):
        import pandas as pd
        # working = df.copy()
        working = df.drop(columns=["process"], errors="ignore").copy()

        low_index_column = f"jetMinChiHiggsIndex{index + 1}"
        second_index_column = f"jetSecMinChiHiggsIndex{index + 1}"

        working.loc[working[low_index_column] > 7, low_index_column] = index
        working.loc[working[low_index_column] == -6, low_index_column] = index
        working.loc[working[second_index_column] > 7, second_index_column] = index
        working.loc[working[second_index_column] == -6, second_index_column] = index

        node_cols = [
            "jetPT" + str(index + 1),
            "jetEta" + str(index + 1),
            "jetPhi" + str(index + 1),
            "jetMinChiHiggsIndex" + str(index + 1),
            "jetBTagDisc" + str(index + 1),
        ]

        rest_cols = [
            "jetHT",  # "bjetHT", "lightjetHT",
            "jetNumber", "jetAverageMass",
            "leptonPT1", "leptonEta1", "leptonPhi1",
            "leptonPT2", "leptonEta2", "leptonPhi2",
            "met",
        ]

        if drop_empty:
            working = working.loc[working[f"jetPT{index + 1}"] >= 0].copy()

        btag_weight = working["btag_weight"].to_numpy()

        node_part = working[node_cols].to_numpy()
        rest_part = working[rest_cols].to_numpy()

        low_partner = (working[low_index_column] + 1).astype(int).astype(str)
        second_partner = (working[second_index_column] + 1).astype(int).astype(str)

        # print("index is : ", index_main)
        # print("the dataframe : ", np.sum(("jetPT" + low_partner) == "jetPT0"))

        label_higgs = working[f"jetHiggsMatched{index + 1}"].to_numpy(dtype=bool)  # .drop(empty_index)
        label_top = working[f"jetTopMatched{index + 1}"].to_numpy(dtype=bool)  # .drop(empty_index)
        # label_sample = working["sample"].drop(empty_index)
        label_others = (~ np.logical_or(label_top, label_higgs)).astype(int)

        low_rows = pd.Series("jetPT" + low_partner, index=working.index)
        neighbour = [
            self._safe_lookup(working, low_rows.index, low_rows),
            # .drop(empty_index)) for the following lines is droped -> use instead padding
            self._safe_lookup(working, low_rows.index, pd.Series("jetEta" + low_partner, index=working.index)),
            self._safe_lookup(working, low_rows.index, pd.Series("jetPhi" + low_partner, index=working.index)),
            self._safe_lookup(working, low_rows.index, pd.Series(
                "jetBTagDisc" + low_partner, index=working.index,
            )),
        ]

        second_rows = pd.Series("jetPT" + second_partner, index=working.index)
        neighbour2 = [
            self._safe_lookup(working, second_rows.index, second_rows),
            # .drop(empty_index)) for the following lines is droped -> use instead padding
            self._safe_lookup(working, second_rows.index, pd.Series("jetEta" + second_partner, index=working.index)),
            self._safe_lookup(working, second_rows.index, pd.Series("jetPhi" + second_partner, index=working.index)),
            self._safe_lookup(working, second_rows.index, pd.Series(
                "jetBTagDisc" + second_partner, index=working.index,
            )),
        ]
        graph_data = np.hstack(
            (btag_weight[:, None], node_part, rest_part, np.array(neighbour).T, np.array(neighbour2).T),
        ).astype(np.float32)
        # graph_data = np.hstack((main.to_numpy(), np.array(neighbour).T, np.array(neighbour2).T))
        labels = np.vstack(
            (
                label_higgs.astype(int),
                label_top.astype(int),
                label_others.astype(int),
            ),
        )
        padding_mask = self.compute_padding_mask(working, index)
        event_weights = working["weights"].to_numpy(dtype=np.float32)

        return graph_data, labels, padding_mask, event_weights

    def create_graphs(
        self,
        df: "pd.DataFrame",
        index: int,
        drop_empty: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._create_graphs_core(df, index=index, drop_empty=drop_empty)

    def prepare_stage_one_graph_tensors(
        self,
        df_sample: "pd.DataFrame",
        max_njets: int = 8,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    ]:
        sample_blocks = []
        label_blocks = []
        btag_weights_blocks = []
        process_blocks = []
        padding_blocks = []
        event_weights_blocks = []

        for index in range(max_njets):
            sample_block, label_block, padding_mask, event_weight_block = self.create_graphs(df_sample, index)
            real = ~padding_mask
            btag_weights_blocks.append(sample_block[real, 0])
            sample_blocks.append(sample_block[real, 1:])
            label_blocks.append(label_block.T[real].astype(np.float32))
            process_blocks.append(df_sample["process"].to_numpy()[real])
            padding_blocks.append(padding_mask[real])
            event_weights_blocks.append(event_weight_block[real])

        sample = np.concatenate(sample_blocks, axis=0)
        label = np.concatenate(label_blocks, axis=0)
        btag_weights = np.concatenate(btag_weights_blocks, axis=0)
        process_names = np.concatenate(process_blocks, axis=0)
        padded = np.concatenate(padding_blocks, axis=0)
        event_weights = np.concatenate(event_weights_blocks, axis=0)

        assert label.shape[1] == len(self.jet_classes), f"label shape {label.shape} != (n, 3)"
        assert len(sample) == len(label) == len(btag_weights) == len(process_names) == len(padded)

        return sample, label, btag_weights, process_names, padded, event_weights

    def _log_class_composition(self, label, process_names, event_weights=None, tag: str = "") -> None:
        logger.info(f"class composition ({tag}):")

        groups = [("ALL", np.ones(len(label), dtype=bool))]
        groups += [(str(p), process_names == p) for p in np.unique(process_names)]

        for name, sel in groups:
            n = int(sel.sum())
            if n == 0:
                continue
            counts = label[sel].sum(axis=0)
            parts = [
                f"{cls}={int(c):,} ({c / n:.2%})"
                for cls, c in zip(self.jet_classes, counts)
            ]
            line = f"  {name:<22} n_jets={n:>10,}   " + "   ".join(parts)

            if event_weights is not None:
                w_counts = (label[sel] * event_weights[sel, None]).sum(axis=0)
                line += "   | weighted: " + " ".join(
                    f"{cls}={w:.1f}" for cls, w in zip(self.jet_classes, w_counts)
                )
            logger.info(line)

    def make_model_gnn(self, input_shape, index_node: int, index_neigh1: int, index_neigh2: int):
        import tensorflow
        from tensorflow import keras
        layers = tensorflow.keras.layers
        from tensorflow.keras.layers import BatchNormalization

        inputs = keras.Input(shape=input_shape)
        input_node_value = inputs[:, :index_node]
        input_neigh1_value = inputs[:, -(index_neigh1 + index_neigh2):-index_neigh2]
        input_neigh2_value = inputs[:, -index_neigh2:]
        input_rest = inputs[:, index_node:-(index_neigh1 + index_neigh2)]

        def dense_layer(values, units: int):
            values = layers.Dense(units)(values)
            values = layers.LeakyReLU()(values)
            return values

        node_value = dense_layer(input_node_value, 256)
        node_value = layers.Concatenate()([node_value, input_node_value])
        node_value = dense_layer(node_value, 128)

        neigh1_value = dense_layer(input_neigh1_value, 256)
        neigh1_value = layers.Concatenate()([neigh1_value, input_neigh1_value])
        neigh1_value = dense_layer(neigh1_value, 128)

        neigh2_value = dense_layer(input_neigh2_value, 256)
        neigh2_value = layers.Concatenate()([neigh2_value, input_neigh2_value])
        neigh2_value = dense_layer(neigh2_value, 128)

        weight_main = layers.Softmax()(keras.ops.matmul(keras.ops.transpose(node_value), node_value))
        weight_neigh1 = layers.Softmax()(keras.ops.matmul(keras.ops.transpose(node_value), neigh1_value))
        weight_neigh2 = layers.Softmax()(keras.ops.matmul(keras.ops.transpose(node_value), neigh2_value))

        rest = dense_layer(input_rest, 256)
        rest = layers.Concatenate()([rest, input_rest])
        rest = dense_layer(rest, 128)

        node = node_value * weight_main[:, 0]
        neigh1 = neigh1_value * weight_neigh1[:, 0]
        neigh2 = neigh2_value * weight_neigh2[:, 0]

        max_embed = layers.Concatenate()([node, layers.Maximum()([neigh1, neigh2])])
        x_dense = layers.Concatenate()([rest, max_embed])
        x_dense = layers.Dropout(0.15)(x_dense)

        def dropout_layer(values, units: int):
            values = layers.Dense(units)(values)
            values = BatchNormalization()(values)
            values = layers.LeakyReLU()(values)
            values = layers.Dropout(0.15)(values)
            values = layers.Concatenate()([values, x_dense])
            return values

        x = dropout_layer(x_dense, 2048)
        x = dropout_layer(x, 2048)
        x = dropout_layer(x, 2048)
        x = dropout_layer(x, 2048)
        x = dropout_layer(x, 1024)
        x = dropout_layer(x, 512)
        x = dropout_layer(x, 128)
        x = dropout_layer(x, 32)
        outputs = layers.Dense(3, activation="softmax")(x)
        return keras.Model(inputs, outputs)

    class_weight_power = 0.5

    def compute_class_weights(self, label: np.ndarray) -> np.ndarray:
        class_sums = label.sum(axis=0).astype(np.float64)
        w_cls = class_sums ** (-self.class_weight_power)
        w_cls /= np.average(w_cls, weights=class_sums)

        logger.info(
            "class weights: " + ", ".join(
                f"{c}={w:.3f} (n={int(n):,})"
                for c, w, n in zip(self.jet_classes, w_cls, class_sums)
            ),
        )
        return w_cls.astype(np.float32)

    def make_optimizer(self, train_data_length: int):
        import tensorflow as tf
        class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, initial_lr, decay_steps, warmup_steps=2, warmup_lr=1e-9, name=None):
                super().__init__()
                self.initial_lr = float(initial_lr)
                self.decay_steps = int(decay_steps)
                self.warmup_steps = int(warmup_steps)
                self.warmup_lr = float(warmup_lr)
                self.name = name
                self.cosine = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=self.initial_lr,
                    decay_steps=self.decay_steps,
                )

            def __call__(self, step):
                import tensorflow as tf
                step = tf.cast(step, tf.float32)
                warmup_steps = tf.cast(self.warmup_steps, tf.float32)

                def warmup():
                    return self.warmup_lr + (self.initial_lr - self.warmup_lr) * (step / warmup_steps)

                def decay():
                    return self.cosine(step - warmup_steps)

                return tf.cond(step < warmup_steps, warmup, decay)

            def get_config(self):
                return {
                    "initial_lr": self.initial_lr,
                    "decay_steps": self.decay_steps,
                    "warmup_steps": self.warmup_steps,
                    "warmup_lr": self.warmup_lr,
                    "name": self.name,
                }

        import math
        steps_per_epoch = math.ceil(train_data_length / self.batchsize)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = int(self.warmup_epochs * steps_per_epoch)

        lr_schedule = WarmupCosineDecay(
            initial_lr=self.initial_lr,
            decay_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_lr=self.warmup_lr,
        )

        return tf.keras.optimizers.Lamb(learning_rate=lr_schedule)

    def compile_stage_one(self, model, optimizer=None):
        import tensorflow as tf
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            # loss=custom_focal_loss,
            metrics=[
                "accuracy",
                "categorical_crossentropy",
                tf.keras.metrics.AUC(
                    name="auc_higgs", multi_label=True, num_labels=len(self.jet_classes),
                    label_weights=[1.0, 0.0, 0.0], num_thresholds=50,
                ),
            ],
        )
        return model

    def prepare_ml_model(self, task: Any):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.seed + task.fold)
        model = self.make_model_gnn(
            input_shape=(self.n_features,),
            index_node=self.stage_one_index_node,
            index_neigh1=self.stage_one_index_neigh1,
            index_neigh2=self.stage_one_index_neigh2,
        )
        # model = self.compile_stage_one(model, self.make_optimizer())
        return model

    _jet_downsample = {
        "tt_custom": (1.0, 0.15, 0.05),
        "tt_bb_custom": (1.0, 0.2, 0.15),
        "tth": (1.0, 0.4, 0.15),
        "tthh_4b": (1.0, 1.0, 1.0),
        "hhh_4b2w_2l2nu_c30_d40": (1.0, 1.0, 1.0),
    }

    def _downsample_jets(self, sample, label, btag, process_names, padded, event_weights, tag=""):
        if not self._jet_downsample:
            return sample, label, btag, process_names, padded, event_weights

        class_ids = label.argmax(axis=1)
        rng = np.random.default_rng(self.seed)
        u = rng.random(len(label))
        keep = np.ones(len(label), dtype=bool)

        for proc, fracs in self._jet_downsample.items():
            for cls_idx, frac in enumerate(fracs):
                if frac >= 1.0:
                    continue
                sel = (process_names == proc) & (class_ids == cls_idx)
                n_before = int(sel.sum())
                if n_before == 0:
                    continue
                keep &= ~(sel & (u >= frac))
                logger.info(
                    f"{tag} jet-downsampling {proc}/{self.jet_classes[cls_idx]}: "
                    f"{n_before} -> {int((sel & keep).sum())} ({frac:.0%})",
                )

        return (sample[keep], label[keep], btag[keep],
                process_names[keep], padded[keep], event_weights[keep])

    def fit_ml_model(
        self,
        task: Any,
        model,
        train: DotDict,
        validation: DotDict,
        output,
    ) -> None:
        import tensorflow as tf
        from sklearn.preprocessing import RobustScaler, QuantileTransformer, MinMaxScaler
        import pandas as pd
        import gc

        df_train = pd.concat(
            [self._proc_data_to_dataframe(proc_data, proc_inst.name)
             for proc_inst, proc_data in train.items()],
            ignore_index=True,
        )
        df_val = pd.concat(
            [self._proc_data_to_dataframe(proc_data, proc_inst.name)
             for proc_inst, proc_data in validation.items()],
            ignore_index=True,
        )
        logger.info(
            f"fold {task.fold}: {len(df_train)} train / {len(df_val)} val events "
            f"({dict(df_train['process'].value_counts())})",
        )
        for proc_data in list(train.values()) + list(validation.values()):
            proc_data.cleanup()

        gc.collect()

        train_sample, train_label, train_btag_weights, train_process, train_padded, train_event_weights = \
            self.prepare_stage_one_graph_tensors(df_train, max_njets=self.n_jets)

        train_sample, train_label, train_btag_weights, train_process, train_padded, train_event_weights = \
            self._downsample_jets(train_sample, train_label, train_btag_weights,
                                train_process, train_padded, train_event_weights, tag="train")

        val_sample, val_label, val_btag_weights, val_process, val_padded, val_event_weights = \
            self.prepare_stage_one_graph_tensors(df_val, max_njets=self.n_jets)

        assert train_sample.shape[1] == self.n_features, (
            f"graph width {train_sample.shape[1]} != n_features {self.n_features}"
        )

        self._log_class_composition(train_label, train_process, train_event_weights, tag=f"fold{task.fold} train")
        self._log_class_composition(val_label, val_process, val_event_weights, tag=f"fold{task.fold} val")

        del df_train, df_val
        gc.collect()

        robust_scaler = RobustScaler()
        quantile_scaler = QuantileTransformer(random_state=42)
        minmax_scaler = MinMaxScaler()

        x_train = minmax_scaler.fit_transform(
            quantile_scaler.fit_transform(
                robust_scaler.fit_transform(train_sample),
            ),
        )
        x_val = minmax_scaler.transform(
            quantile_scaler.transform(
                robust_scaler.transform(val_sample),
            ),
        )

        def _as_f32(name, arr):
            a = np.asarray(arr)
            if a.dtype != np.float32:
                logger.info(f"{name}: casting {a.dtype} -> float32")
                a = a.astype(np.float32)
            return a

        x_train = _as_f32("x_train", x_train)
        x_val = _as_f32("x_val", x_val)
        train_label = _as_f32("train_label", train_label)
        val_label = _as_f32("val_label", val_label)
        train_btag_weights = _as_f32("train_btag_weights", train_btag_weights)
        val_btag_weights = _as_f32("val_btag_weights", val_btag_weights)

        class_weight_vec = self.compute_class_weights(train_label)
        train_weights = train_btag_weights * class_weight_vec[train_label.argmax(axis=1)]
        val_weights = val_btag_weights * class_weight_vec[val_label.argmax(axis=1)]

        optimizer = self.make_optimizer(train_data_length=len(x_train))
        model = self.compile_stage_one(model, optimizer)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=output["checkpoint"].abspath,
                monitor="val_loss", mode="min", save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min",
                min_delta=0, patience=50,
                restore_best_weights=True, verbose=1,
            ),
            # tf.keras.callbacks.EarlyStopping(
            #     monitor="val_loss", mode="min",
            #     min_delta=1e-5, patience=50,
            #     restore_best_weights=True,
            #     verbose=1,
            # ),
            tf.keras.callbacks.CSVLogger(
                output["mlmodel"].child(f"history_fold{task.fold}.csv", type="f").abspath,
                append=False,
            ),
        ]

        output["mlmodel"].child("robust_scaler.pkl", type="f").dump(robust_scaler, formatter="pickle")
        output["mlmodel"].child("quantile_scaler.pkl", type="f").dump(quantile_scaler, formatter="pickle")
        output["mlmodel"].child("minmax_scaler.pkl", type="f").dump(minmax_scaler, formatter="pickle")

        history = model.fit(
            x=x_train,
            y=train_label,
            sample_weight=train_weights,
            validation_data=(x_val, val_label, val_weights),
            # sample_weight=train_btag_weights,
            # validation_data=(x_val, val_label, val_btag_weights),
            epochs=self.epochs,
            batch_size=self.batchsize,
            callbacks=callbacks,
            verbose=2,
        )

        val_pred = model.predict(x_val, batch_size=self.batchsize, verbose=0)

        plot_sets = [("all", np.ones(len(val_label), dtype=bool))]
        plot_sets += [(str(p), val_process == p) for p in np.unique(val_process)]

        for name, sel in plot_sets:
            if not np.any(sel):
                continue
            tag = f"val_fold{task.fold}_{name}"
            self._plot_confusion(val_label[sel], val_pred[sel], val_event_weights[sel], output, tag)
            self._plot_roc(val_label[sel], val_pred[sel], val_event_weights[sel], output, tag)
            self._plot_scores(val_label[sel], val_pred[sel], val_event_weights[sel], output, tag)

        output["mlmodel"].child("history.pkl", type="f").dump(history.history, formatter="pickle")

        del train_sample, val_sample, x_train, x_val
        gc.collect()

    def open_model(self, target) -> dict[str, Any]:
        import tensorflow as tf

        models = {}

        models["input_features"] = tuple(
            target["mlmodel"].child("input_features.pkl", type="f").load(formatter="pickle"),
        )

        with open(target["mlmodel"].child("parameters.yaml", type="f").fn) as f:
            models["parameters"] = yaml.load(f.read(), Loader=yaml.Loader)

        models["model"] = tf.keras.models.load_model(
            target["mlmodel_file"].abspath, compile=False,
        )
        models["best_model"] = tf.keras.models.load_model(
            target["checkpoint"].abspath, compile=False,
        )

        for scaler in ("robust_scaler", "quantile_scaler", "minmax_scaler"):
            models[scaler] = target["mlmodel"].child(f"{scaler}.pkl", type="f").load(formatter="pickle")

        return models

    def _plot_confusion(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray,
        output,
        tag: str,
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(
            y_true.argmax(axis=1),
            y_pred.argmax(axis=1),
            labels=range(len(self.jet_classes)),
            sample_weight=sample_weight,
            normalize="true",
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ticks = range(len(self.jet_classes))
        ax.set_xticks(ticks, self.jet_classes)
        ax.set_yticks(ticks, self.jet_classes)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_title(f"GATJA confusion ({tag})")
        for i in ticks:
            for j in ticks:
                ax.text(
                    j, i, f"{cm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black",
                )
        fig.tight_layout()

        output["plots"].child(f"confusion_{tag}.pdf", type="f").dump(fig, formatter="mpl")
        plt.close(fig)

    def _plot_roc(self, y_true, y_pred, sample_weight, output, tag: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        fig, ax = plt.subplots(figsize=(6, 5))
        for i, cls in enumerate(self.jet_classes):
            # n_pos = int(y_true[:, i].sum())
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            ax.plot(fpr, tpr, label=f"{cls} (AUC = {auc(fpr, tpr):.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title(f"GATJA ROC ({tag})")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        output["plots"].child(f"roc_{tag}.pdf", type="f").dump(fig, formatter="mpl")
        plt.close(fig)

    def _plot_scores(self, y_true, y_pred, sample_weight, output, tag: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(self.jet_classes)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        for ax, (i, cls) in zip(axes[0], enumerate(self.jet_classes)):
            matched = y_true[:, i].astype(bool)
            for sel, label, color in (
                (~matched, "unmatched", "darkseagreen"),
                (matched, "matched", "darkred"),
            ):
                if not np.any(sel):
                    continue
                ax.hist(y_pred[sel, i], bins=40, range=(0, 1), weights=sample_weight[sel],
                        density=True, histtype="step", lw=2, color=color, label=label)
            ax.set_yscale("log")
            ax.set_xlabel(f"{cls} score")
            ax.set_ylabel("normalized")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)

        fig.suptitle(f"GATJA scores ({tag})")
        fig.tight_layout()
        output["plots"].child(f"scores_{tag}.pdf", type="f").dump(fig, formatter="mpl")
        plt.close(fig)

    def evaluate(
        self,
        task,
        events: ak.Array,
        models: list,
        fold_indices: ak.Array,
        events_used_in_training: bool = True,
    ) -> ak.Array:

        import pandas as pd
        from hbw.ml.data_loader import input_features_sanity_checks

        use_best_model = True  # True: Checkpoint (bestes val_loss) statt letzter Epoche
        n_classes = len(self.jet_classes)
        n_out = self.n_jets * n_classes

        for model in models:
            input_features_sanity_checks(self, model["input_features"])

        nets = [model["best_model" if use_best_model else "model"] for model in models]
        scalers = [
            (model["robust_scaler"], model["quantile_scaler"], model["minmax_scaler"])
            for model in models
        ]

        df_all = pd.DataFrame({
            col: np.asarray(ak.to_numpy(events[col]))
            for col in sorted(self.input_features)
        })

        evt_pos = np.arange(len(events), dtype=np.int64)
        fold_np = np.asarray(ak.to_numpy(fold_indices), dtype=np.int64)

        allowed = df_all["jetNumber"].to_numpy() >= 3
        df_f = df_all.loc[allowed].reset_index(drop=True)
        pos_f = evt_pos[allowed]
        folds_f = fold_np[allowed]

        outputs = np.full((len(events), n_out), -10.0, dtype=np.float32)

        for jet_idx in range(self.n_jets):

            keep = df_f[f"jetPT{jet_idx + 1}"].to_numpy() != self.padding_value
            if not np.any(keep):
                continue
            df_kept = df_f.loc[keep].reset_index(drop=True)
            pos_kept = pos_f[keep]
            folds_kept = folds_f[keep]

            sample_block, _, _, event_weights = self.create_graphs(df_kept, jet_idx, drop_empty=False)
            x_raw = sample_block[:, 1:]

            for fold in range(self.folds):
                sel = folds_kept == fold
                if not np.any(sel):
                    continue

                robust_scaler, quantile_scaler, minmax_scaler = scalers[fold]
                x = minmax_scaler.transform(
                    quantile_scaler.transform(
                        robust_scaler.transform(x_raw[sel]),
                    ),
                )
                pred = nets[fold].predict(x, batch_size=self.batchsize, verbose=0)
                outputs[pos_kept[sel], jet_idx * n_classes:(jet_idx + 1) * n_classes] = pred

        for i in range(n_out):
            events = set_ak_column(
                events, f"gatja_output_{i}", np.ascontiguousarray(outputs[:, i]),
            )

        return events


gatja_training = GatjaTraining_MLClassifierBase_small_class_weights_05.derive("gatja_training", cls_dict={})
