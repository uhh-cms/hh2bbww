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

from columnflow.types import Sequence
from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.columnar_util import Route, set_ak_column
from columnflow.config_util import get_datasets_from_process

from hbw.util import log_memory
from hbw.ml.data_loader import MLDatasetLoader, MLProcessData, input_features_sanity_checks
from hbw.config.processes import create_combined_proc_forML

from hbw.tasks.ml import MLPreTraining

np = maybe_import("numpy")
ak = maybe_import("awkward")
pickle = maybe_import("pickle")

logger = law.logger.get_logger(__name__)


class MLClassifierBase(MLModel):
    """
    Provides a base structure to implement Multiclass Classifier in Columnflow
    """

    # set some defaults, can be overwritten by subclasses or via cls_dict
    # NOTE: the order of processes is crucial! Do not change after training
    processes: tuple = ("tt", "st")
    input_features: set = {"mli_ht", "mli_n_jet"}

    # identifier of the PrepareMLEvents and MergeMLEvents outputs. Needs to be changed when producing new input features
    store_name: str = "inputs_base"

    # Class for data loading and it's dependencies.
    data_loader = MLDatasetLoader
    # NOTE: we might want to use the data_loader.hyperparameter_deps instead
    preml_params: set[str] = {"data_loader", "input_features", "train_val_test_split"}

    # NOTE: we split each fold into train, val, test + do k-folding, so we have a 4-way split in total
    # TODO: test whether setting "test" to 0 is working
    train_val_test_split: tuple = (0.75, 0.15, 0.10)
    folds: int = 5

    # training-specific parameters. Only need to re-run training when changing these
    ml_process_weights: dict = {"st": 2, "tt": 1}
    negative_weights: str = "handle"
    epochs: int = 50
    batchsize: int = 2 ** 10

    # parameters to add into the `parameters` attribute to determine the 'parameters_repr' and to store in a yaml file
    bookkeep_params: set[str] = {
        "data_loader", "input_features", "train_val_test_split",
        "processes", "ml_process_weights", "negative_weights", "epochs", "batchsize", "folds",
    }

    # parameters that can be overwritten via command line
    settings_parameters: set[str] = {
        "processes", "ml_process_weights",
        "negative_weights", "epochs", "batchsize",
    }

    def __init__(
            self,
            *args,
            folds: int | None = None,
            **kwargs,
    ):
        """
        Initialization function of the MLModel. We first overwrite the class attributes with what is set
        in the  *self.parameters* attribute via the `--ml-model-settings` parameter. Then we cast the
        parameters to the correct types and store them as individual class attributes. Finally, we store
        the parameters in the `self.parameters` attribute, which is used both to create a hash for the output
        path and to store the parameters in a yaml file.

        Only the parameters in the `settings_parameters` attribute can be overwritten via the command line.
        Only the parameters in the `bookkeep_params` attribute are stored in the `self.parameters` attribute.
        """
        super().__init__(*args, **kwargs)

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
            # overwrite the default value with the value from the parameters
            # TODO: this is quite dangerous, as it overwrites a class attribute instead of an instance attribute
            setattr(self, param, self.parameters.get(param, getattr(self, param)))

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

    def cast_ml_param_values(self):
        """
        Resolve the values of the parameters that are used in the MLModel
        """
        self.processes = tuple(self.processes)
        self.input_features = set(self.input_features)
        self.train_val_test_split = tuple(self.train_val_test_split)
        if not isinstance(self.ml_process_weights, dict):
            # cast tuple to dict
            self.ml_process_weight = {
                proc: weight for proc, weight in [s.split(":") for s in self.ml_process_weight]
            }
        # cast weights to int and remove processes not used in training
        self.ml_model_weights = {
            proc: int(weight)
            for proc, weight in self.ml_process_weights.items()
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

    def setup(self) -> None:
        """ function that is run as part of the setup phase. Most likely overwritten by subclasses """
        logger.debug(
            f"Setting up MLModel {self.cls_name} (parameter hash: {self.parameters_repr}), "
            f"parameters: \n{self.parameters}",
        )
        # dynamically add processes and variables for the quantities produced by this model
        # NOTE: this function might not be called for all configs when the requested configs
        # between MLTraining and the requested task are different
        for proc in self.combine_processes:
            for config_inst in self.config_insts:
                if proc not in config_inst.processes:
                    proc_name = str(proc)
                    proc_dict = DotDict(self.combine_processes[proc])
                    create_combined_proc_forML(config_inst, proc_name, proc_dict)

        for proc in self.processes:
            for config_inst in self.config_insts:
                if f"mlscore.{proc}" not in config_inst.variables:
                    config_inst.add_variable(
                        name=f"mlscore.{proc}",
                        expression=f"mlscore.{proc}",
                        null_value=-1,
                        binning=(1000, 0., 1.),
                        x_title=f"DNN output score {config_inst.get_process(proc).x('ml_label', proc)}",
                        aux={
                            "rebin": 25,
                            "rebin_config": {
                                "processes": [proc],
                                "n_bins": 4,
                            },
                        },  # automatically rebin to 40 bins for plotting tasks
                    )

    def preparation_producer(self: MLModel, analysis_inst: od.Analysis):
        """ producer that is run as part of PrepareMLEvents and MLEvaluation (before `evaluate`) """
        return "ml_preparation"

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
            proc_inst.x.ml_id = i
            proc_inst.x.ml_process_weight = self.ml_process_weights.get(proc, 1)

            # get datasets corresponding to this process
            dataset_insts = [
                dataset_inst for dataset_inst in
                get_datasets_from_process(config_inst, proc, strategy="all")
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
        if not all(var.startswith("mli_") for var in self.input_features):
            raise Exception(
                "We currently expect all input_features to start with 'mli_', which is not the case"
                f"for one of the variables in the 'input_features' {self.input_features}",
            )
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
        for proc in self.processes:
            produced.add(f"mlscore.{proc}")

        return produced

    def output(self, task: law.Task) -> dict[str, law.FileSystemTarget]:

        # declare the main target
        target = task.target(f"mlmodel_f{task.branch}of{self.folds}", dir=True)

        outp = {
            "mlmodel": target,
            "plots": target.child("plots", type="d", optional=True),
            "checkpoint": target.child("checkpoint", type="d", optional=True),
        }

        # define all files that need to be present
        outp["required_files"] = [
            target.child(fname, type="f") for fname in
            ("saved_model.pb", "keras_metadata.pb", "fingerprint.pb", "parameters.yaml", "input_features.pkl")
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
        from hbw.ml.tf_util import cumulated_crossentropy

        models["model"] = tf.keras.models.load_model(
            target["mlmodel"].path, custom_objects={cumulated_crossentropy.__name__: cumulated_crossentropy},
        )
        models["best_model"] = tf.keras.models.load_model(
            target["checkpoint"].path, custom_objects={cumulated_crossentropy.__name__: cumulated_crossentropy},
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

        validation = MLProcessData(
            self, input_files, "val", self.processes, task.fold,
        )
        # load into memory
        validation.load_all
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
        self.process_insts = [self.config_inst.get_process(proc) for proc in self.processes]
        # np.random.seed(1337)  # for reproducibility

        # input preparation
        train, validation = self.load_data(task, input, output)
        physical_devices = tf.config.list_physical_devices("GPU")
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

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
        model.save(output["mlmodel"].path)

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

        # check that the input features are the same for all models
        for model in models:
            input_features_sanity_checks(self, model["input_features"])

        process = task.dataset_inst.x("ml_process", task.dataset_inst.processes.get_first().name)
        process_inst = task.config_inst.get_process(process)

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
            if len(pred[0]) != len(self.processes):
                raise Exception("Number of output nodes should be equal to number of processes")
            predictions.append(pred)
            # store predictions for each model
            for j, proc in enumerate(self.processes):
                events = set_ak_column(
                    events, f"fold{i}_mlscore.{proc}", pred[:, j],
                )

        # combine all models into 1 output score, using the model that has not yet seen the test set
        outputs = ak.where(ak.ones_like(predictions[0]), -1, -1)
        for i in range(self.folds):
            logger.info(f"Evaluation fold {i}")
            # reshape mask from N*bool to N*k*bool (TODO: simpler way?)
            idx = ak.to_regular(ak.concatenate([ak.singletons(fold_indices == i)] * len(self.processes), axis=1))
            outputs = ak.where(idx, predictions[i], outputs)

        # sanity check of the number of output nodes
        if len(outputs[0]) != len(self.processes):
            raise Exception(
                f"The number of output nodes {len(outputs[0])} should be equal to "
                f"the number of processes {len(self.processes)}",
            )

        for i, proc in enumerate(self.processes):
            events = set_ak_column(
                events, f"mlscore.{proc}", outputs[:, i],
            )

        return events


class ExampleDNN(MLClassifierBase):
    """ Example class how to implement a DNN from the MLClassifierBase """

    # optionally overwrite input parameters
    epochs: int = 10

    def prepare_ml_model(
        self,
        task: law.Task,
    ):
        """
        Minimal implementation of a ML model
        """
        import tesorflow.keras as keras

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

        with tf.device("CPU"):
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
