# coding: utf-8

"""
First implementation of DNN for HH analysis, generalized (TODO)
"""

from __future__ import annotations

from typing import Any
import gc
import time

import law
import order as od

from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.columnar_util import Route, set_ak_column, remove_ak_column
from columnflow.tasks.selection import MergeSelectionStatsWrapper

from hbw.util import log_memory
from hbw.ml.helper import assign_dataset_to_process, predict_numpy_on_batch
from hbw.ml.plotting import (
    plot_history, plot_confusion, plot_roc_ovr, plot_roc_ovo, plot_output_nodes,
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
    processes = ["tt", "st"]
    dataset_names = ["tt_sl_powheg", "tt_dl_powheg", "st_tchannel_t_powheg"]
    input_features = ["mli_ht", "mli_n_jet"]
    validation_fraction = 0.20  # percentage of the non-test events
    store_name = "inputs_base"
    ml_process_weights = {"st": 2, "tt": 1}
    eqweight = True
    epochs = 50
    batchsize = 2 ** 10
    folds = 5

    def __init__(
            self,
            *args,
            folds: int | None = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert self.eqweight in (True, False)

    def setup(self):
        # dynamically add variables for the quantities produced by this model
        # NOTE: since these variables are only used in ConfigTasks,
        #       we do not need to add these variables to all configs
        for proc in self.processes:
            if f"{self.cls_name}.score_{proc}" not in self.config_inst.variables:
                self.config_inst.add_variable(
                    name=f"{self.cls_name}.score_{proc}",
                    null_value=-1,
                    binning=(40, 0., 1.),
                    x_title=f"DNN output score {self.config_inst.get_process(proc).x.ml_label}",
                )

    def requires(self, task: law.Task) -> str:
        # TODO: either remove or include some MLStats task
        # add selection stats to requires; NOTE: not really used at the moment
        return MergeSelectionStatsWrapper.req(
            task,
            shifts="nominal",
            configs=self.config_inst.name,
            datasets=self.dataset_names,
        )

    def sandbox(self, task: law.Task) -> str:
        # venv_ml_tf sandbox but with scikit-learn and restrictet to tf 2.11.0
        return dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_plotting.sh")
        # return dev_sandbox("bash::$CF_BASE/sandboxes/venv_ml_tf.sh")

    def datasets(self, config_inst: od.Config) -> set[od.Dataset]:
        return {config_inst.get_dataset(dataset_name) for dataset_name in self.dataset_names}

    def uses(self, config_inst: od.Config) -> set[Route | str]:
        return set(self.input_features) | {"normalization_weight"}

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        produced = set()
        for proc in self.processes:
            produced.add(f"{self.cls_name}.score_{proc}")

        return produced

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"mlmodel_f{task.branch}of{self.folds}", dir=True)

    def open_model(self, target: law.LocalDirectoryTarget) -> tf.keras.models.Model:
        input_features = tuple(target.child(
            "input_features.pkl", type="f",
        ).load(formatter="pickle"))

        model = tf.keras.models.load_model(target.path)
        return model, input_features

    def prepare_inputs(
        self,
        task,
        input,
        output: law.LocalDirectoryTarget,
        per_process: bool = True,
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

                weights = [ak.from_parquet(inp["mlevents"].fn).normalization_weight for inp in files]
                sum_weights = sum([ak.sum(w) for w in weights])
                sum_abs_weights = sum([ak.sum(np.abs(w)) for w in weights])

                # bookkeep filenames and stats per process
                proc_inst.x.filenames = proc_inst.x("filenames", []) + filenames
                proc_inst.x.N_events = proc_inst.x("N_events", 0) + N_events
                proc_inst.x.sum_weights = proc_inst.x("sum_weights", 0) + sum_weights
                proc_inst.x.sum_abs_weights = proc_inst.x("sum_abs_weights", 0) + sum_abs_weights

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

                # event weights, normalized to the sum of events per process
                weights = ak.to_numpy(events.normalization_weight).astype(np.float32)
                ml_weights = weights / proc_inst.x.sum_abs_weights * proc_inst.x.N_events

                # transform ml weights to handle negative weights
                m_negative_weights = ml_weights < 0
                ml_weights[m_negative_weights] = (
                    np.abs(ml_weights[m_negative_weights]) / (len(self.process_insts) - 1)
                )
                ml_weights = ml_weights.astype(np.float32)
                proc_inst.x.sum_ml_weights = proc_inst.x("sum_ml_weights", 0) + ak.sum(ml_weights)

                # check that all relevant input features are present
                if not set(self.input_features).issubset(set(events.fields)):
                    raise Exception(
                        f"The columns {set(events.fields).difference(set(self.input_features))} "
                        "are not present in the ML input events",
                    )

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

                # create truth values
                label = np.ones(len(events)) * proc_inst.x.ml_id
                target = np.zeros((len(events), len(self.processes))).astype(np.float32)
                target[:, proc_inst.x.ml_id] = 1

                # transform target layer for events with negative weights (1 -> 0 and 0 -> 1)
                # TODO: network classifies everything into 1 process when doing this. Needs to be fixed.
                # target[m_negative_weights] = 1 - target[m_negative_weights]

                # shuffle arrays
                np.random.shuffle(shuffle_indices := np.array(range(len(events))))
                events = events[shuffle_indices]
                ml_weights = ml_weights[shuffle_indices]
                weights = weights[shuffle_indices]
                target = target[shuffle_indices]

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

                for array, key in (
                    (events, "inputs"),
                    (target, "target"),
                    (label, "label"),
                    (weights, "weights"),
                    (ml_weights, "ml_weights"),
                ):
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
        output.child("input_features.pkl", type="f").dump(input_features, formatter="pickle")

        # shuffle per process
        for inp in (train, validation):
            for proc_inst, _inp in inp.items():
                np.random.shuffle(shuffle_indices := np.array(range(len(_inp.inputs))))
                for key in _inp.keys():
                    inp[proc_inst][key] = inp[proc_inst][key][shuffle_indices]

        # if requested, merge per process
        if not per_process:
            def merge_processes(inputs: DotDict[any, DotDict[any: np.array]]):
                return DotDict({
                    key: np.concatenate([inputs[proc][key] for proc in inputs.keys()])
                    for key in list(inputs.values())[0].keys()
                })

            validation = merge_processes(validation)
            train = merge_processes(train)

            # shuffle all events
            for inp in (train, validation):
                np.random.shuffle(shuffle_indices := np.array(range(len(inp.inputs))))
                for key in _inp.keys():
                    inp[key] = inp[key][shuffle_indices]

        return train, validation

    def prepare_ml_model(
        self,
        task: law.Task,
    ):
        """
        Minimal implementation of a ML model
        """

        from keras.models import Sequential
        from keras.layers import Dense, BatchNormalization

        n_inputs = len(self.input_features)
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
        optimizer = keras.optimizers.Adam(learning_rate=0.00050)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            weighted_metrics=["categorical_accuracy", "memory_GB"],
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
        from hbw.ml.multi_dataset import MultiDataset

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

    def create_train_val_plots(
        self,
        task: law.Task,
        model,
        train: tf.data.Dataset,
        validation: tf.data.Dataset,
        output: law.LocalDirectoryTarget,
    ) -> None:
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
        call_func_safe(plot_confusion, model, train, output, "train", self.process_insts)
        call_func_safe(plot_confusion, model, validation, output, "validation", self.process_insts)

        # create some ROC curves
        call_func_safe(plot_roc_ovr, model, train, output, "train", self.process_insts)
        call_func_safe(plot_roc_ovr, model, validation, output, "validation", self.process_insts)
        call_func_safe(plot_roc_ovo, model, train, output, "train", self.process_insts)
        call_func_safe(plot_roc_ovo, model, validation, output, "validation", self.process_insts)
        gc.collect()
        # create plots for all output nodes
        call_func_safe(plot_output_nodes, model, train, validation, output, self.process_insts)

        return

    def train(
        self,
        task: law.Task,
        input: Any,
        output: law.LocalDirectoryTarget,
    ) -> ak.Array:
        # np.random.seed(1337)  # for reproducibility

        physical_devices = tf.config.list_physical_devices("GPU")
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

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

        model = self.prepare_ml_model(task)
        logger.info(model.summary())
        log_memory("prepare-model")
        #
        # training
        #

        # merge validation data
        validation = DotDict({
            key: np.concatenate([validation[proc][key] for proc in validation.keys()])
            for key in list(validation.values())[0].keys()
        })
        log_memory("val merged")

        # train the model
        self.fit_ml_model(task, model, train, validation, output)
        log_memory("training")
        # save the model and history; TODO: use formatter
        # output.dump(model, formatter="tf_keras_model")
        model.save(output.path)

        # merge train data
        train = DotDict({
            key: np.concatenate([train[proc][key] for proc in train.keys()])
            for key in list(train.values())[0].keys()
        })
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
        models, input_features = zip(*models)

        # check that the correct input features were used during training
        if any(map(lambda x: x != input_features[0], input_features)):
            raise Exception(f"the input_features are not equal for all 5 ML models: \n{input_features}")

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
                f"The columns {set(inputs.fields).difference(set(self.input_features))} "
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
            pred = ak.from_numpy(model.predict_on_batch(inputs))
            if len(pred[0]) != len(self.processes):
                raise Exception("Number of output nodes should be equal to number of processes")
            predictions.append(pred)

            # store predictions for each model
            for j, proc in enumerate(self.processes):
                events = set_ak_column(
                    events, f"{self.cls_name}.fold{i}_score_{proc}", pred[:, j],
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
                events, f"{self.cls_name}.score_{proc}", outputs[:, i],
            )

        return events


base_test = MLClassifierBase.derive("base_test", cls_dict={"folds": 5})
