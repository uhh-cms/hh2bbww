# coding: utf-8

"""
First implementation of DNN for HH analysis, generalized (TODO)
"""

from __future__ import annotations

from typing import Any
import gc
from time import time

import law
import order as od

from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.columnar_util import Route, set_ak_column, remove_ak_column
from columnflow.tasks.selection import MergeSelectionStatsWrapper

from hbw.ml.helper import assign_dataset_to_process

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
    def __init__(
            self,
            *args,
            folds: int | None = None,
            **kwargs,
    ):

        # set some defaults, can be overwritten (TODO) via cls_dict
        self.processes = ["tt", "st"]
        self.ml_weights = {"st": 2}
        self.dataset_names = ["tt_sl_powheg", "tt_dl_powheg", "st_tchannel_t_powheg"]
        self.input_features = ["mli_ht", "mli_n_jet"]
        self.validation_fraction = 0.20  # percentage of the non-test events
        self.store_name = "inputs_base"

        super().__init__(*args, **kwargs)

        # class- to instance-level attributes
        # (before being set, self.folds refers to a class-level attribute)
        self.folds = folds or self.folds

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
        return dev_sandbox("bash::$CF_BASE/sandboxes/venv_ml_tf.sh")

    def datasets(self, config_inst: od.Config) -> set[od.Dataset]:
        return {config_inst.get_dataset(dataset_name) for dataset_name in self.dataset_names}

    def uses(self, config_inst: od.Config) -> set[Route | str]:
        # for now: start with only input features, ignore event weights
        return set(self.input_features)  # | {"event_weight"}

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        produced = set()
        for proc in self.processes:
            produced.add(f"{self.cls_name}.score_{proc}")

        return produced

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"mlmodel_f{task.branch}of{self.folds}", dir=True)

    def open_model(self, target: law.LocalDirectoryTarget) -> tf.keras.models.Model:
        # return target.load(formatter="keras_model")

        # with open(f"{target.path}/model_history.pkl", "rb") as f:
        #     history = pickle.load(f)
        history = target.child("model_history.pkl", type="f").load(formatter="pickle")

        input_features = tuple(target.child(
            "input_features.pickle", type="f",
        ).load(formatter="pickle"))

        model = tf.keras.models.load_model(target.path)
        return model, history, input_features

    def prepare_inputs(
        self,
        task,
        input,
        output: law.LocalDirectoryTarget,
    ) -> dict[str, np.array]:

        # get process instances and assign relevant information to the process_insts
        # from the first config_inst (NOTE: this assumes that all config_insts use the same processes)
        process_insts = []
        for i, proc in enumerate(self.processes):
            proc_inst = self.config_insts[0].get_process(proc)
            proc_inst.x.ml_id = i
            proc_inst.x.ml_weight = self.ml_weights.get(proc, 1)

            process_insts.append(proc_inst)

        # assign datasets to proceses and calculate some stats per process
        # TODO: testing with working multi-config analysis
        for config_inst in self.config_insts:
            for dataset, files in input["events"][config_inst.name].items():
                t0 = time()

                dataset_inst = config_inst.get_dataset(dataset)
                if len(dataset_inst.processes) != 1:
                    raise Exception("only 1 process inst is expected for each dataset")

                # calculate some stats per dataset
                filenames = [inp["mlevents"].path for inp in files]
                proc_inst.x.filenames = proc_inst.x("filenames", []) + filenames
                N_events = sum([len(ak.from_parquet(fn)) for fn in filenames])
                sum_weights = 0  # TODO

                # do the dataset-process assignment
                assign_dataset_to_process(dataset_inst, process_insts)
                proc_inst = dataset_inst.x.ml_process

                # bookkeep stats per process
                proc_inst.x.N_events = proc_inst.x("N_events", 0) + N_events
                proc_inst.x.sum_weights = proc_inst.x("sum_weights", 0) + sum_weights

                logger.info(f"Dataset {dataset} was assigned to process {proc_inst.name}; took {(time() - t0):.3f}s")

        # Number to scale weights such that the largest weights are at the order of 1
        # TODO: implement as part of the event weights
        weights_scaler = min([(proc_inst.x.N_events / proc_inst.x.ml_weight) for proc_inst in process_insts])  # noqa

        #
        # set inputs, weights and targets for each datset and fold
        #

        train = DotDict()
        validation = DotDict()

        # bookkeep that input features are always the same
        input_features = None

        for proc_inst in process_insts:
            for fn in proc_inst.x.filenames:
                events = ak.from_parquet(fn)

                # TODO: everything concerning weights
                # weights = events.ml_event_weights

                # check that all relevant input features are present
                if not set(self.input_features).issubset(set(events.fields)):
                    raise Exception(
                        f"The columns {set(events.fields).difference(set(self.input_features)) "
                        "are not present in the ML input events"
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

                # shuffle events and weights
                np.random.shuffle(shuffle_indices := np.array(range(len(events))))
                events = events[shuffle_indices]
                # weights = weights[shuffle_indices]

                # create truth values (no need to shuffle them)
                target = np.zeros((len(events), len(self.processes)))
                target[:, proc_inst.x.ml_id] = 1

                # split into train and validation dataset
                N_events_validation = int(self.validation_fraction * len(events))

                if not train.keys():
                    # initialize arrays on the first call
                    train.inputs = events[N_events_validation:]
                    train.target = target[N_events_validation:]
                    validation.inputs = events[:N_events_validation]
                    validation.target = target[:N_events_validation]
                else:
                    # concatenate arrays on all following calls
                    train.inputs = np.concatenate([train.inputs, events[N_events_validation:]])
                    train.target = np.concatenate([train.target, target[N_events_validation:]])
                    validation.inputs = np.concatenate([validation.inputs, events[:N_events_validation]])
                    validation.target = np.concatenate([validation.target, target[:N_events_validation]])

        output.child("input_features.pickle", type="f").dump(input_features, formatter="pickle")
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
            metrics=["categorical_accuracy"],
        )

        return model

    def fit_ml_model(
        self,
        task: law.Task,
        model,
        tf_train: tf.data.Dataset,
        tf_validation: tf.data.Dataset,
    ) -> None:
        """
        Minimal implementation of training loop.
        """

        # tf_train = tf_train.batch(2 ** 14)
        # tf_train  = tf_validation.batch(2 ** 14)

        logger.info("Starting training...")
        model.fit(
            tf_train,
            validation_data=tf_validation,
            epochs=5,
            verbose=2,
        )

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

        train, validation = self.prepare_inputs(task, input, output)

        # check for infinite values
        for key in train.keys():
            if np.any(~np.isfinite(train[key])):
                raise Exception(f"Infinite values found in training {key}")
            if np.any(~np.isfinite(validation[key])):
                raise Exception(f"Infinite values found in validation {key}")

        gc.collect()
        logger.info("garbage collected")

        with tf.device("CPU"):
            tf_train = tf.data.Dataset.from_tensor_slices(
                (train.inputs, train.target),
            ).batch(2 ** 14)
            tf_validation = tf.data.Dataset.from_tensor_slices(
                (validation.inputs, validation.target),
            ).batch(2 ** 14)

        #
        # model preparation
        #

        model = self.prepare_ml_model(task)

        #
        # training
        #
        self.fit_ml_model(task, model, tf_train, tf_validation)

        # save the model and history; TODO: use formatter
        # output.dump(model, formatter="tf_keras_model")
        model.save(output.path)
        output.child("model_history.pkl", type="f").dump(model.history.history)

    def evaluate(
        self,
        task: law.Task,
        events: ak.Array,
        models: list(Any),
        fold_indices: ak.Array,
        events_used_in_training: bool = True,
    ) -> None:
        logger.info(f"Evaluation of dataset {task.dataset}")

        # separate the outputs generated by the MLTraining task
        models, history, input_features = zip(*models)
        # TODO: use history for loss+acc plotting

        if any(map(lambda x: x != input_features[0], input_features)):
            raise Exception(f"the input_features are not equal for all 5 ML models: \n{input_features}")

        input_features = input_features[0]
        if set(input_features) != set(self.input_features):
            raise Exception(
                f"The input features used in training {input_features} are not the "
                f"same as defined by the ML model {self.input_features}"
            )

        # check that all relevant input features are present
        if not set(self.input_features).issubset(set(events.fields)):
            raise Exception(
                f"The columns {set(events.fields).difference(set(self.input_features)) "
                "are not present in the ML input events"
            )

        # create a copy of the inputs to use for evaluation
        inputs = ak.copy(events)

        # remove columns not used in training
        for var in inputs.fields:
            if var not in self.input_features:
                inputs = remove_ak_column(inputs, var)

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

            # Save predictions for each model
            # TODO: create train/val/test plots (confusion, ROC, nodes) using these predictions?
            """
            for j, proc in enumerate(self.processes):
                events = set_ak_column(
                    events, f"{self.cls_name}.fold{i}_score_{proc}", pred[:, j],
                )
            """
        # combine all models into 1 output score, using the model that has not seen test set yet
        outputs = ak.where(ak.ones_like(predictions[0]), -1, -1)
        for i in range(self.folds):
            logger.info(f"Evaluation fold {i}")
            # reshape mask from N*bool to N*k*bool (TODO: simpler way?)
            idx = ak.to_regular(ak.concatenate([ak.singletons(fold_indices == i)] * len(self.processes), axis=1))
            outputs = ak.where(idx, predictions[i], outputs)

        if len(outputs[0]) != len(self.processes):
            raise Exception("Number of output nodes should be equal to number of processes")

        for i, proc in enumerate(self.processes):
            events = set_ak_column(
                events, f"{self.cls_name}.score_{proc}", outputs[:, i],
            )

        # ML categorization on top of existing categories
        ml_categories = [cat for cat in self.config_inst.categories if "ml_" in cat.name]
        ml_proc_to_id = {cat.name.replace("ml_", ""): cat.id for cat in ml_categories}

        scores = ak.Array({
            f.replace("score_", ""): events[self.cls_name, f]
            for f in events[self.cls_name].fields if f.startswith("score_")
        })

        ml_category_ids = max_score = ak.Array(np.zeros(len(events)))
        for proc in scores.fields:
            ml_category_ids = ak.where(scores[proc] > max_score, ml_proc_to_id[proc], ml_category_ids)
            max_score = ak.where(scores[proc] > max_score, scores[proc], max_score)

        category_ids = ak.where(
            events.category_ids != 1,  # Do not split Inclusive category into DNN sub-categories
            events.category_ids + ak.values_astype(ml_category_ids, np.int32),
            events.category_ids,
        )
        events = set_ak_column(events, "category_ids", category_ids)
        return events


base_test = MLClassifierBase.derive("base_test", cls_dict={"folds": 5})
