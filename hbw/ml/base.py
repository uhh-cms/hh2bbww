# coding: utf-8

"""
First implementation of DNN for HH analysis, generalized (TODO)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any
import gc
import time
import yaml

import law
import order as od

from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.columnar_util import Route, set_ak_column, remove_ak_column

from hbw.util import log_memory
from hbw.ml.helper import assign_dataset_to_process, predict_numpy_on_batch
from hbw.ml.plotting import (
    plot_history, plot_confusion, plot_roc_ovr,  # plot_roc_ovo,
    plot_output_nodes, get_input_weights,
)


np = maybe_import("numpy")
ak = maybe_import("awkward")
tf = maybe_import("tensorflow")
pickle = maybe_import("pickle")
keras = maybe_import("tensorflow.keras")


logger = law.logger.get_logger(__name__)


class MLClassifierBase(MLModel):
    """
    Provides a base structure to implement Multiclass Classifier in Columnflow
    """

    # set some defaults, can be overwritten by subclasses or via cls_dict
    processes: list = ["tt", "st"]
    dataset_names: set = {"tt_sl_powheg", "tt_dl_powheg", "st_tchannel_t_powheg"}
    input_features: list = ["mli_ht", "mli_n_jet"]
    validation_fraction: float = 0.20  # percentage of the non-test events
    store_name: str = "inputs_base"
    ml_process_weights: dict = {"st": 2, "tt": 1}
    negative_weights: str = "handle"
    epochs: int = 50
    batchsize: int = 2 ** 10
    folds: int = 5

    dump_arrays: bool = False

    # parameters to add into the `parameters` attribute and store in a yaml file
    bookkeep_params: int = [
        "processes", "dataset_names", "input_features", "validation_fraction", "ml_process_weights",
        "negative_weights", "epochs", "batchsize", "folds",
    ]

    def __init__(
            self,
            *args,
            folds: int | None = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # TODO: find some appropriate names for the negative_weights modes
        assert self.negative_weights in ("ignore", "abs", "handle")

        for param in self.bookkeep_params:
            self.parameters[param] = getattr(self, param, None)

    def setup(self):
        # dynamically add variables for the quantities produced by this model
        # NOTE: since these variables are only used in ConfigTasks,
        #       we do not need to add these variables to all configs
        for proc in self.processes:
            if f"mlscore.{proc}" not in self.config_inst.variables:
                self.config_inst.add_variable(
                    name=f"mlscore.{proc}",
                    null_value=-1,
                    binning=(40, 0., 1.),
                    x_title=f"DNN output score {self.config_inst.get_process(proc).x.ml_label}",
                )

    def preparation_producer(self: MLModel, config_inst: od.Config):
        """ producer that is run as part of PrepareMLEvents and MLEvaluation (before `evaluate`) """
        return "ml_preparation"

    def requires(self, task: law.Task) -> str:
        # Custom requirements (none currently)
        return {}

    def sandbox(self, task: law.Task) -> str:
        # venv_ml_tf sandbox but with scikit-learn and restricted to tf 2.11.0
        return dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_plotting.sh")

    def datasets(self, config_inst: od.Config) -> set[od.Dataset]:
        return {config_inst.get_dataset(dataset_name) for dataset_name in self.dataset_names}

    def uses(self, config_inst: od.Config) -> set[Route | str]:
        return set(self.input_features) | {"normalization_weight"}

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        produced = set()
        for proc in self.processes:
            produced.add(f"mlscore.{proc}")

        return produced

    def output(self, task: law.Task) -> dict[law.FileSystemTarget]:

        # declare the main target
        target = task.target(f"mlmodel_f{task.branch}of{self.folds}", dir=True)

        outp = {
            "mlmodel": target,
            "checkpoint": target.child("checkpoint", type="d", optional=True),
            "plots": target.child("plots", type="d", optional=True),
            "stats": target.child("stats.yaml", type="f", optional=True),
            "arrays": target.child("arrays", type="d", optional=True),
        }

        # define all files that need to be present
        outp["required_files"] = [
            target.child(fname, type="f") for fname in
            ("saved_model.pb", "keras_metadata.pb", "fingerprint.pb", "parameters.yaml", "input_features.pkl")
        ]

        return outp

    def open_model(self, target: law.LocalDirectoryTarget) -> dict[str, Any]:
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

    def prepare_inputs(
        self,
        task,
        input,
        output: law.LocalDirectoryTarget,
    ) -> dict[str, np.array]:
        # get process instances and assign relevant information to the process_insts
        # from the first config_inst (NOTE: this assumes that all config_insts use the same processes)
        self.process_insts = []
        for i, proc in enumerate(self.processes):
            proc_inst = self.config_insts[0].get_process(proc)
            proc_inst.x.ml_id = i
            proc_inst.x.ml_process_weight = self.ml_process_weights.get(proc, 1)

            self.process_insts.append(proc_inst)

        # assign datasets to proceses and calculate some stats per process
        # TODO: testing with working multi-config analysis
        for config_inst in self.config_insts:
            for dataset, files in input["events"][config_inst.name].items():
                t0 = time.perf_counter()

                dataset_inst = config_inst.get_dataset(dataset)
                if len(dataset_inst.processes) != 1:
                    raise Exception("only 1 process inst is expected for each dataset")

                # do the dataset-process assignment and raise Execption when unmatched
                if not assign_dataset_to_process(dataset_inst, self.process_insts):
                    raise Exception(
                        f"The dataset {dataset_inst.name} is not matched to any"
                        f"of the given processes {self.process_insts}",
                    )
                proc_inst = dataset_inst.x.ml_process

                # calculate some stats per dataset
                filenames = [inp["mlevents"].path for inp in files]

                N_events = sum([len(ak.from_parquet(fn)) for fn in filenames])
                if N_events == 0:
                    # skip empty datasets
                    logger.warning(f"Dataset {dataset_inst.name} is empty and will be ignored")
                    continue

                weights = [ak.from_parquet(inp["mlevents"].fn).normalization_weight for inp in files]
                sum_weights = sum([ak.sum(w) for w in weights])
                sum_abs_weights = sum([ak.sum(np.abs(w)) for w in weights])
                sum_pos_weights = sum([ak.sum(w[w > 0]) for w in weights])

                # bookkeep filenames and stats per process
                proc_inst.x.filenames = proc_inst.x("filenames", []) + filenames
                proc_inst.x.N_events = proc_inst.x("N_events", 0) + N_events
                proc_inst.x.sum_weights = proc_inst.x("sum_weights", 0) + sum_weights
                proc_inst.x.sum_abs_weights = proc_inst.x("sum_abs_weights", 0) + sum_abs_weights
                proc_inst.x.sum_pos_weights = proc_inst.x("sum_pos_weights", 0) + sum_pos_weights

                logger.info(
                    f"Dataset {dataset} was assigned to process {proc_inst.name}; "
                    f"took {(time.perf_counter() - t0):.3f}s {chr(10)}"
                    f"----- Number of events: {N_events} {chr(10)}"
                    f"----- Sum of weights:   {sum_weights}",
                )

        #
        # set inputs, weights and targets for each datset and fold
        #

        train = DotDict()
        validation = DotDict()

        # bookkeep that input features are always the same
        input_features = None

        for proc_inst in self.process_insts:
            _train = DotDict()
            _validation = DotDict()
            logger.info(
                f"Preparing inputs for process {proc_inst.name} {chr(10)}"
                f"----- Number of files:  {len(proc_inst.x.filenames)} {chr(10)}"
                f"----- Number of events: {proc_inst.x.N_events} {chr(10)}"
                f"----- Sum of weights:   {proc_inst.x.sum_weights}",
            )
            t0 = time.perf_counter()
            for fn in proc_inst.x.filenames:
                events = ak.from_parquet(fn)
                if len(events) == 0:
                    logger.warning("File {fn} of process {proc_inst.name} is empty and will be skipped")
                    continue
                # check that all relevant input features are present
                if not set(self.input_features).issubset(set(events.fields)):
                    raise Exception(
                        f"The columns {set(self.input_features).difference(events.fields)} "
                        "are not present in the ML input events",
                    )

                # create truth values
                label = np.ones(len(events)) * proc_inst.x.ml_id
                target = np.zeros((len(events), len(self.processes))).astype(np.float32)
                target[:, proc_inst.x.ml_id] = 1

                # event weights, normalized to the sum of events per process
                weights = ak.to_numpy(events.normalization_weight).astype(np.float32)
                ml_weights = weights / proc_inst.x.sum_abs_weights * proc_inst.x.N_events

                # transform ml weights to handle negative weights
                m_negative_weights = ml_weights < 0
                if self.negative_weights == "ignore":
                    ml_weights[m_negative_weights] = 0
                elif self.negative_weights == "abs":
                    ml_weights = np.abs(ml_weights)
                elif self.negative_weights == "handle":
                    ml_weights[m_negative_weights] = (
                        np.abs(ml_weights[m_negative_weights]) / (len(self.process_insts) - 1)
                    )
                    # transform target layer for events with negative weights (1 -> 0 and 0 -> 1)
                    target[m_negative_weights] = 1 - target[m_negative_weights]

                ml_weights = ml_weights.astype(np.float32)
                proc_inst.x.sum_ml_weights = proc_inst.x("sum_ml_weights", 0) + ak.sum(ml_weights)

                # remove columns that are not used as training features
                for var in events.fields:
                    if var not in self.input_features:
                        events = remove_ak_column(events, var)

                # bookkeep order of input features and check that it is the same for all datasets
                if input_features is None:
                    input_features = tuple(events.fields)
                elif input_features != tuple(events.fields):
                    raise Exception("The order of input features is not the same for all datasets")

                # transform events into numpy npdarray
                events = ak.to_numpy(events)
                events = events.astype(
                    [(name, np.float32) for name in events.dtype.names], copy=False,
                ).view(np.float32).reshape((-1, len(events.dtype)))

                # split into train and validation dataset
                N_events_validation = int(self.validation_fraction * len(events))

                def add_arrays(array: np.array, key: str):
                    """
                    Small helper to add arrays split into train and validation
                    """
                    if key not in _train.keys():
                        # initialize list on first call
                        _train[key] = [array[N_events_validation:]]
                        _validation[key] = [array[:N_events_validation]]
                    else:
                        # append on all following calls
                        _train[key].append(array[N_events_validation:])
                        _validation[key].append(array[:N_events_validation])

                # shuffle arrays in-place and add them to train and validation dictionaries
                np.random.shuffle(shuffle_indices := np.array(range(len(events))))
                for array, key in (
                    (events, "inputs"),
                    (target, "target"),
                    (label, "label"),
                    (weights, "weights"),
                    (ml_weights, "ml_weights"),
                ):
                    array[...] = array[shuffle_indices]
                    add_arrays(array, key)

            # concatenate arrays per process
            for inp in (_train, _validation):
                for key, arrays in inp.items():
                    inp[key] = np.concatenate(arrays)

            train[proc_inst] = _train
            validation[proc_inst] = _validation

            logger.info(f"Input preparation done for process {proc_inst.name}; took {(time.perf_counter() - t0):.1f}s")

        # check that weights are set as expected
        for proc_inst in self.process_insts:
            logger.info(f"{proc_inst.name} sum of ml weights: {proc_inst.x.sum_ml_weights:.1f}")

        # save tuple of input feature names for sanity checks in MLEvaluation
        output["mlmodel"].child("input_features.pkl", type="f").dump(input_features, formatter="pickle")

        # shuffle per process
        for inp in (train, validation):
            for proc_inst, _inp in inp.items():
                np.random.shuffle(shuffle_indices := np.array(range(len(_inp.inputs))))
                for key in _inp.keys():
                    inp[proc_inst][key] = inp[proc_inst][key][shuffle_indices]

        # reweight validation events to match the number of events used in the training multi_dataset
        weights_scaler = (
            min([proc_inst.x.N_events / proc_inst.x.ml_process_weight for proc in self.process_insts]) *
            sum([proc_inst.x.ml_process_weight for proc_inst in self.process_insts])
        )
        for proc_inst in validation.keys():
            validation[proc_inst].ml_weights = (
                validation[proc_inst].ml_weights * weights_scaler / proc_inst.x.N_events *
                proc_inst.x.ml_process_weight
            )

        return train, validation

    def merge_processes(self, inputs: DotDict[any, DotDict[any: np.array]]):
        """ Helper function to concatenate arrays in double-dict structure """
        return DotDict({
            key: np.concatenate([inputs[proc][key] for proc in inputs.keys()])
            for key in list(inputs.values())[0].keys()
        })

    def merge_and_shuffle(self, train, validation):
        """ Helper function to merge and shuffle training and validation inputs """
        train = self.merge_processes(train)
        validation = self.merge_processes(validation)

        # shuffle all events
        for inp in (train, validation):
            np.random.shuffle(shuffle_indices := np.array(range(len(inp.inputs))))
            for key in inp.keys():
                inp[key] = inp[key][shuffle_indices]

        return train, validation

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

    def create_train_val_plots(
        self,
        task: law.Task,
        model,
        train: tf.data.Dataset,
        validation: tf.data.Dataset,
        output: law.LocalDirectoryTarget,
    ) -> None:
        output_stats = output["stats"]
        stats = {}

        # store all outputs from this function in the 'plots' directory
        output = output["plots"]

        # store the model history
        output.child("model_history.pkl", type="f").dump(model.history.history)

        def call_func_safe(func, *args, **kwargs) -> Any:
            """
            Small helper to make sure that our training does not fail due to plotting
            """

            # get the function name without the possibility of raising an error
            try:
                func_name = func.__name__
            except Exception:
                # default to empty name
                func_name = ""

            t0 = time.perf_counter()

            try:
                outp = func(*args, **kwargs)
                logger.info(f"Function '{func_name}' done; took {(time.perf_counter() - t0):.2f} seconds")
            except Exception as e:
                logger.warning(f"Function '{func_name}' failed due to {type(e)}: {e}")
                outp = None

            return outp

        # get a simple ranking of input variables
        call_func_safe(get_input_weights, model, output)

        log_memory("start plotting")
        # make some plots of the history
        for metric, ylabel in (
            ("loss", "Loss"),
            ("categorical_accuracy", "Accuracy"),
            ("weighted_categorical_accuracy", "Weighted Accuracy"),
        ):
            call_func_safe(plot_history, model.history.history, output, metric, ylabel)

        # evaluate training and validation sets
        train.prediction = call_func_safe(predict_numpy_on_batch, model, train.inputs)
        validation.prediction = call_func_safe(predict_numpy_on_batch, model, validation.inputs)

        # create some confusion matrices
        call_func_safe(plot_confusion, model, train, output, "train", self.process_insts, stats=stats)
        call_func_safe(plot_confusion, model, validation, output, "validation", self.process_insts, stats=stats)
        gc.collect()

        # create some ROC curves
        call_func_safe(plot_roc_ovr, model, train, output, "train", self.process_insts, stats=stats)
        call_func_safe(plot_roc_ovr, model, validation, output, "validation", self.process_insts, stats=stats)
        # call_func_safe(plot_roc_ovo, model, train, output, "train", self.process_insts)
        # call_func_safe(plot_roc_ovo, model, validation, output, "validation", self.process_insts)
        gc.collect()

        # create plots for all output nodes
        call_func_safe(plot_output_nodes, model, train, validation, output, self.process_insts)

        # dump all stats into yaml file
        output_stats.dump(stats, formatter="yaml")

        return

    def train(
        self,
        task: law.Task,
        input: Any,
        output: law.LocalDirectoryTarget,
    ) -> ak.Array:
        """ Training function that is called during the MLTraining task """
        # np.random.seed(1337)  # for reproducibility

        physical_devices = tf.config.list_physical_devices("GPU")
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

        # hyperparameter bookkeeping
        output["mlmodel"].child("parameters.yaml", type="f").dump(dict(self.parameters), formatter="yaml")

        #
        # input preparation
        #
        log_memory("start")
        train, validation = self.prepare_inputs(task, input, output)

        log_memory("prepare_inputs")
        # check for infinite values
        for proc_inst in train.keys():
            for key in train[proc_inst].keys():
                if np.any(~np.isfinite(train[proc_inst][key])):
                    raise Exception(f"Infinite values found in training {key}, process {proc_inst.name}")
                if np.any(~np.isfinite(validation[proc_inst][key])):
                    raise Exception(f"Infinite values found in validation {key}, process {proc_inst.name}")

        gc.collect()
        log_memory("garbage collected")
        #
        # model preparation
        #

        if self.dump_arrays:
            def dump_arrays(inputs: DotDict[any, DotDict[any, np.array]], output, type: str):
                for proc_inst, arrays in inputs.items():
                    outp = output.child(f"{type}_{proc_inst.name}.npz", type="f")
                    outp.touch()
                    np.savez(outp.fn, **arrays)

            dump_arrays(train, output["arrays"], "train")
            dump_arrays(validation, output["arrays"], "validation")

            # return without training
            return

        model = self.prepare_ml_model(task)
        logger.info(model.summary())
        log_memory("prepare-model")

        #
        # training
        #

        # merge validation data
        validation = self.merge_processes(validation)
        log_memory("val merged")

        # train the model
        self.fit_ml_model(task, model, train, validation, output)
        log_memory("training")
        # save the model and history; TODO: use formatter
        # output.dump(model, formatter="tf_keras_model")
        model.save(output["mlmodel"].path)

        # merge train data
        train = self.merge_processes(train)
        log_memory("train merged")

        #
        # direct evaluation as part of MLTraining
        #

        self.create_train_val_plots(task, model, train, validation, output)

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
        use_best_model = False

        if len(events) == 0:
            logger.warning(f"Dataset {task.dataset} is empty. No columns are produced.")
            return events

        logger.info(f"Evaluation of dataset {task.dataset}")

        # determine truth label of the dataset (-1 if not used in training)
        ml_truth_label = -1
        if events_used_in_training:
            process_insts = []
            for i, proc in enumerate(self.processes):
                proc_inst = self.config_inst.get_process(proc)
                proc_inst.x.ml_id = i
                process_insts.append(proc_inst)

            assign_dataset_to_process(task.dataset_inst, process_insts)
            ml_truth_label = task.dataset_inst.x.ml_process.x.ml_id

        # store the ml truth label in the events
        events = set_ak_column(
            events, f"{self.cls_name}.ml_truth_label",
            ak.Array(np.ones(len(events)) * ml_truth_label),
        )

        # separate the outputs generated by the MLTraining task
        parameters = [model["parameters"] for model in models]
        input_features = [model["input_features"] for model in models]

        if use_best_model:
            models = [model["best_model"] for model in models]
        else:
            models = [model["model"] for model in models]

        # check that all MLTrainings were started with the same set of parameters
        from hbw.util import dict_diff
        for i, params in enumerate(parameters[1:]):
            if params != parameters[0]:
                diff = dict_diff(params, parameters[0])
                raise Exception(
                    "The MLTraining parameters (see 'parameters.yaml') from "
                    f"fold {i} differ from fold 0; diff: {diff}",
                )

        # check that the correct input features were used during training
        if any(map(lambda x: x != input_features[0], input_features)):
            raise Exception(f"The input_features are not equal for all 5 ML models: {input_features}")

        input_features = input_features[0]
        if set(input_features) != set(self.input_features):
            raise Exception(
                f"The input features used in training {input_features} are not the "
                f"same as defined by the ML model {self.input_features}",
            )

        # create a copy of the inputs to use for evaluation
        inputs = ak.copy(events)

        # check that all relevant input features are present
        if not set(self.input_features).issubset(set(inputs.fields)):
            raise Exception(
                f"The columns {set(self.input_features).difference(set(inputs.fields))} "
                "are not present in the ML input events",
            )

        # remove columns not used in training
        for var in inputs.fields:
            if var not in input_features:
                inputs = remove_ak_column(inputs, var)

        # check that all input features are present and reorder them if necessary

        if diff := set(inputs.fields).difference(set(self.input_features)):
            raise Exception(f"Columns {diff} are not present in the ML input events")
        if tuple(inputs.fields) != input_features:
            inputs = ak.Array({var: inputs[var] for var in input_features})

        # transform inputs into numpy ndarray
        inputs = ak.to_numpy(inputs)
        inputs = inputs.astype(
            [(name, np.float32) for name in inputs.dtype.names], copy=False,
        ).view(np.float32).reshape((-1, len(inputs.dtype)))

        # do prediction for all models and all inputs
        predictions = []
        for i, model in enumerate(models):
            # NOTE: the next line triggers some warning concering tf.function retracing
            pred = ak.from_numpy(model.predict_on_batch(inputs))
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
        from hbw.ml.tf_util import MultiDataset

        with tf.device("CPU"):
            tf_train = MultiDataset(data=train, batch_size=self.batchsize, kind="train")
            tf_validation = tf.data.Dataset.from_tensor_slices(
                (validation.inputs, validation.target, validation.ml_weights),
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
