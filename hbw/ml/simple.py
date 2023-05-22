# coding: utf-8

"""
First implementation of DNN for HH analysis, generalized (TODO)
"""

from __future__ import annotations

from typing import Any, Sequence
import gc
from time import time

import law
import order as od

from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import Route, set_ak_column, remove_ak_column
from columnflow.tasks.selection import MergeSelectionStatsWrapper
from hbw.config.categories import add_categories_ml

np = maybe_import("numpy")
ak = maybe_import("awkward")
tf = maybe_import("tensorflow")
pickle = maybe_import("pickle")
keras = maybe_import("tensorflow.keras")

logger = law.logger.get_logger(__name__)


class SimpleDNN(MLModel):

    def __init__(
            self,
            *args,
            folds: int | None = None,
            **kwargs,
    ):
        """
        Parameters that need to be set by derived model:
        folds, layers, learningrate, batchsize, epochs, eqweight, dropout,
        processes, custom_procweights, dataset_names, input_features, store_name,
        """

        single_config = True  # noqa

        super().__init__(*args, **kwargs)

        # class- to instance-level attributes
        # (before being set, self.folds refers to a class-level attribute)
        self.folds = folds or self.folds

        # DNN model parameters
        """
        self.layers = [512, 512, 512]
        self.learningrate = 0.00050
        self.batchsize = 2048
        self.epochs = 6  # 200
        self.eqweight = 0.50

        # Dropout: either False (disable) or a value between 0 and 1 (dropout_rate)
        self.dropout = False
        """

    def setup(self):
        # dynamically add variables for the quantities produced by this model
        for proc in self.processes:
            if f"{self.cls_name}.score_{proc}" not in self.config_inst.variables:
                self.config_inst.add_variable(
                    name=f"{self.cls_name}.score_{proc}",
                    null_value=-1,
                    binning=(40, 0., 1.),
                    x_title=f"DNN output score {self.config_inst.get_process(proc).x.ml_label}",
                )
                hh_bins = [0.0, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .92, 1.0]
                bkg_bins = [0.0, 0.4, 0.7, 1.0]
                self.config_inst.add_variable(
                    name=f"{self.cls_name}.score_{proc}_rebin1",
                    expression=f"{self.cls_name}.score_{proc}",
                    null_value=-1,
                    binning=hh_bins if "HH" in proc else bkg_bins,
                    x_title=f"DNN output score {self.config_inst.get_process(proc).x.ml_label}",
                )

        # one variable to bookkeep truth labels
        # TODO: still needs implementation
        if f"{self.cls_name}.ml_label" not in self.config_inst.variables:
            self.config_inst.add_variable(
                name=f"{self.cls_name}.ml_label",
                null_value=-1,
                binning=(len(self.processes) + 1, -1.5, len(self.processes) - 0.5),
                x_title="DNN truth score",
            )

        # dynamically add ml categories (but only if production categories have been added)
        if (
                self.config_inst.x("add_categories_ml", True) and
                not self.config_inst.x("add_categories_production", True)
        ):
            add_categories_ml(self.config_inst, ml_model_inst=self)
            self.config_inst.x.add_categories_ml = False

    def requires(self, task: law.Task) -> str:
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
        return {"normalization_weight", "category_ids"} | set(self.input_features)

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        produced = set()
        for proc in self.processes:
            produced.add(f"{self.cls_name}.score_{proc}")

        produced.add("category_ids")

        return produced

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"mlmodel_f{task.branch}of{self.folds}", dir=True)

    def open_model(self, target: law.LocalDirectoryTarget) -> tf.keras.models.Model:
        # return target.load(formatter="keras_model")

        with open(f"{target.path}/model_history.pkl", "rb") as f:
            history = pickle.load(f)
        model = tf.keras.models.load_model(target.path)
        return model, history

    def training_configs(self, requested_configs: Sequence[str]) -> list[str]:
        # default config
        print(requested_configs)
        if len(requested_configs) == 1:
            return list(requested_configs)
        else:
            # TODO: change to "config_2017" when finished with testing phase
            return ["config_2017_limited"]

    def training_calibrators(self, config_inst: od.Config, requested_calibrators: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        return ["skip_jecunc"]

    def training_selector(self, config_inst: od.Config, requested_selector: str) -> str:
        # fix MLTraining Phase Space
        return "default"

    def training_producers(self, config_inst: od.Config, requested_producers: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        return ["ml_inputs"]

    def prepare_inputs(
        self,
        task,
        input,
    ) -> dict[str, np.array]:

        # max_events_per_fold = int(self.max_events / (self.folds - 1))

        process_insts = [self.config_inst.get_process(proc) for proc in self.processes]
        N_events_processes = np.array(len(self.processes) * [0])
        custom_procweights = np.array(len(self.processes) * [0])
        sum_eventweights_processes = np.array(len(self.processes) * [0])
        dataset_proc_idx = {}  # bookkeeping which process each dataset belongs to

        #
        # determine process of each dataset and count number of events & sum of eventweights for this process
        #

        for dataset, files in input["events"][self.config_inst.name].items():
            t0 = time()

            dataset_inst = self.config_inst.get_dataset(dataset)
            if len(dataset_inst.processes) != 1:
                raise Exception("only 1 process inst is expected for each dataset")

            # TODO: use stats here instead
            N_events = sum([len(ak.from_parquet(inp["mlevents"].fn)) for inp in files])
            # NOTE: this only works as long as each dataset only contains one process
            sum_eventweights = sum([
                ak.sum(ak.from_parquet(inp["mlevents"].fn).normalization_weight)
                for inp in files],
            )
            for i, proc in enumerate(process_insts):
                custom_procweights[i] = self.custom_procweights[proc.name]
                leaf_procs = [p for p, _, _ in self.config_inst.get_process(proc).walk_processes(include_self=True)]
                if dataset_inst.processes.get_first() in leaf_procs:
                    logger.info(f"the dataset *{dataset}* is used for training the *{proc.name}* output node")
                    dataset_proc_idx[dataset] = i
                    N_events_processes[i] += N_events
                    sum_eventweights_processes[i] += sum_eventweights
                    continue

            if dataset_proc_idx.get(dataset, -1) == -1:
                raise Exception(f"dataset {dataset} is not matched to any of the given processes")

            logger.info(f"Weights done for {dataset} in {(time() - t0):.3f}s")

        # Number to scale weights such that the largest weights are at the order of 1
        # (only implemented for eqweight = True)
        weights_scaler = min(N_events_processes / custom_procweights)

        #
        # set inputs, weights and targets for each datset and fold
        #

        DNN_inputs = {
            "weights": None,
            "inputs": None,
            "target": None,
        }

        sum_nnweights_processes = {}

        for dataset, files in input["events"][self.config_inst.name].items():
            t0 = time()
            this_proc_idx = dataset_proc_idx[dataset]
            proc_name = self.processes[this_proc_idx]
            N_events_proc = N_events_processes[this_proc_idx]
            sum_eventweights_proc = sum_eventweights_processes[this_proc_idx]

            logger.info(
                f"dataset: {dataset}, \n  #Events: {N_events_proc}, "
                f"\n  Sum Eventweights: {sum_eventweights_proc}",
            )
            sum_nnweights = 0

            for inp in files:
                events = ak.from_parquet(inp["mlevents"].path)
                weights = events.normalization_weight
                if self.eqweight:
                    weights = weights * weights_scaler / sum_eventweights_proc
                    custom_procweight = self.custom_procweights[proc_name]
                    weights = weights * custom_procweight

                weights = ak.to_numpy(weights)

                if np.any(~np.isfinite(weights)):
                    raise Exception(f"Infinite values found in weights from dataset {dataset}")

                sum_nnweights += sum(weights)
                sum_nnweights_processes.setdefault(proc_name, 0)
                sum_nnweights_processes[proc_name] += sum(weights)

                # remove columns not used in training
                for var in events.fields:
                    if var not in self.input_features:
                        events = remove_ak_column(events, var)

                # transform events into numpy ndarray
                # TODO: at this point we should save the order of our input variables
                #       to ensure that they will be loaded in the correct order when
                #       doing the evaluation
                events = ak.to_numpy(events)
                events = events.astype(
                    [(name, np.float32) for name in events.dtype.names], copy=False,
                ).view(np.float32).reshape((-1, len(events.dtype)))

                if np.any(~np.isfinite(events)):
                    raise Exception(f"Infinite values found in inputs from dataset {dataset}")

                # create the truth values for the output layer
                target = np.zeros((len(events), len(self.processes)))
                target[:, this_proc_idx] = 1

                if np.any(~np.isfinite(target)):
                    raise Exception(f"Infinite values found in target from dataset {dataset}")
                if DNN_inputs["weights"] is None:
                    DNN_inputs["weights"] = weights
                    DNN_inputs["inputs"] = events
                    DNN_inputs["target"] = target
                else:
                    DNN_inputs["weights"] = np.concatenate([DNN_inputs["weights"], weights])
                    DNN_inputs["inputs"] = np.concatenate([DNN_inputs["inputs"], events])
                    DNN_inputs["target"] = np.concatenate([DNN_inputs["target"], target])

            logger.debug(f"   weights: {weights[:5]}")
            logger.debug(f"   Sum NN weights: {sum_nnweights}")

            logger.info(f"Inputs done for {dataset} in {(time() - t0):.3f}s")

        logger.info(f"Sum of weights per process: {sum_nnweights_processes}")

        #
        # shuffle events and split into train and validation fold
        #

        inputs_size = sum([arr.size * arr.itemsize for arr in DNN_inputs.values()])
        logger.info(f"inputs size is {inputs_size / 1024**3} GB")

        shuffle_indices = np.array(range(len(DNN_inputs["weights"])))
        np.random.shuffle(shuffle_indices)

        validation_fraction = 0.25
        N_validation_events = int(validation_fraction * len(DNN_inputs["weights"]))

        train, validation = {}, {}
        for k in DNN_inputs.keys():
            DNN_inputs[k] = DNN_inputs[k][shuffle_indices]

            validation[k] = DNN_inputs[k][:N_validation_events]
            train[k] = DNN_inputs[k][N_validation_events:]

        return train, validation

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

        train, validation = self.prepare_inputs(task, input)

        # check for infinite values
        for key in train.keys():
            if np.any(~np.isfinite(train[key])):
                raise Exception(f"Infinite values found in training {key}")
            if np.any(~np.isfinite(validation[key])):
                raise Exception(f"Infinite values found in validation {key}")

        gc.collect()
        logger.info("garbage collected")

        #
        # model preparation
        #

        n_inputs = len(self.input_features)
        n_outputs = len(self.processes)

        from keras.models import Sequential
        from keras.layers import Dense, BatchNormalization

        # define the DNN model
        # TODO: do this Funcional instead of Sequential
        model = Sequential()

        # BatchNormalization layer with input shape
        model.add(BatchNormalization(input_shape=(n_inputs,)))

        activation_settings = {
            "elu": ("ELU", "he_uniform", "Dropout"),
            "relu": ("ReLU", "he_uniform", "Dropout"),
            "prelu": ("PReLU", "he_normal", "Dropout"),
            "selu": ("selu", "lecun_normal", "AlphaDropout"),
            "tanh": ("tanh", "glorot_normal", "Dropout"),
            "softmax": ("softmax", "glorot_normal", "Dropout"),
        }
        keras_act_name, init_name, dropout_layer = activation_settings[self.activation]

        # following hidden layers
        for n_nodes in self.layers:
            model.add(Dense(
                units=n_nodes,
                activation=keras_act_name,
            ))

            # Potentially add dropout layer after each hidden layer
            if self.dropout:
                Dropout = getattr(keras.layers, dropout_layer)
                model.add(Dropout(self.dropout))

        # output layer
        model.add(Dense(n_outputs, activation="softmax"))

        # compile the network
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learningrate, beta_1=0.9, beta_2=0.999,
            epsilon=1e-6, amsgrad=False,
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            weighted_metrics=["categorical_accuracy"],
        )

        #
        # training
        #

        # early stopping to determine the 'best' model
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=int(self.epochs / 4),
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0,
        )

        logger.info("input to tf Dataset")
        with tf.device("CPU"):
            tf_train = tf.data.Dataset.from_tensor_slices(
                (train["inputs"], train["target"], train["weights"]),
            ).batch(self.batchsize)
            tf_validate = tf.data.Dataset.from_tensor_slices(
                (validation["inputs"], validation["target"], validation["weights"]),
            ).batch(self.batchsize)

        fit_kwargs = {
            "epochs": self.epochs,
            "callbacks": [early_stopping],
            "verbose": 2,
        }

        # train the model
        logger.info("Start training...")
        model.fit(
            tf_train,
            validation_data=tf_validate,
            **fit_kwargs,
        )

        # save the model and history; TODO: use formatter
        # output.dump(model, formatter="tf_keras_model")
        output.parent.touch()
        model.save(output.path)
        with open(f"{output.path}/model_history.pkl", "wb") as f:
            pickle.dump(model.history.history, f)

    def evaluate(
        self,
        task: law.Task,
        events: ak.Array,
        models: list(Any),
        fold_indices: ak.Array,
        events_used_in_training: bool = True,
    ) -> None:
        logger.info(f"Evaluation of dataset {task.dataset}")

        models, history = zip(*models)
        # TODO: use history for loss+acc plotting

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
