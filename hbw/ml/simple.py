# coding: utf-8

"""
First implementation of DNN for HH analysis, generalized (TODO)
"""

from __future__ import annotations

from typing import Any
import gc

import law
import order as od

from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import Route, set_ak_column  #, ChunkedIOHandler
from columnflow.tasks.selection import MergeSelectionStatsWrapper


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
            # layers: list[int] | None = None,
            # learningrate: float | None = None,
            # batchsize: int | None = None,
            # epochs: int | None = None,
            # eqweight: bool | None = None,
            # dropout: bool | None = None,
            # processes: list[str] | None = None,  # TODO: processes might be needed to declare output variables
            # custom_procweights: dict[str, float] | None = None,
            # dataset_names: set[str] | None = None,  # TODO: might not work for input preparation
            # input_features: tuple[str] | None = None,  # TODO: might not work for input preparation
            **kwargs,
    ):
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
        # dynamically add variables for the quantities produced by this model
        for proc in self.processes:
            if f"{self.cls_name}.score_{proc}" not in self.config_inst.variables:
                self.config_inst.add_variable(
                    name=f"{self.cls_name}.score_{proc}",
                    null_value=-1,
                    binning=(40, 0., 1.),
                    x_title=f"DNN output score {self.config_inst.get_process(proc).label}",
                )

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

    def datasets(self) -> set[od.Dataset]:
        return {self.config_inst.get_dataset(dataset_name) for dataset_name in self.dataset_names}

    def uses(self) -> set[Route | str]:
        return {"normalization_weight"} | set(self.input_features)

    def produces(self) -> set[Route | str]:
        produced = set()
        for proc in self.processes:
            produced.add(f"{self.cls_name}.score_{proc}")
        return produced

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"mlmodel_f{task.branch}of{self.folds}", dir=True)

    def open_model(self, target: law.LocalDirectoryTarget) -> tf.keras.models.Model:
        # return target.load(formatter="keras_model")

        with open(f"{target.path}/model_history.pkl", "rb") as f:
            history = pickle.load(f)
        model = tf.keras.models.load_model(target.path)
        return model, history

    def prepare_inputs(
        self,
        task,
        input,
    ) -> dict[str, np.array]:

        max_events_per_fold = int(self.max_events / (self.folds - 1))

        process_insts = [self.config_inst.get_process(proc) for proc in self.processes]
        N_events_processes = np.array(len(self.processes) * [0])
        custom_procweights = np.array(len(self.processes) * [0])
        sum_eventweights_processes = np.array(len(self.processes) * [0])
        dataset_proc_idx = {}  # bookkeeping which process each dataset belongs to

        #
        # determine process of each dataset and count number of events & sum of eventweights for this process
        #

        for dataset, infiletargets in input["events"].items():
            dataset_inst = self.config_inst.get_dataset(dataset)
            if len(dataset_inst.processes) != 1:
                raise Exception("only 1 process inst is expected for each dataset")

            N_events = sum([len(ak.from_parquet(inp.fn)[:max_events_per_fold]) for inp in infiletargets])
            # NOTE: this only works as long as each dataset only contains one process
            sum_eventweights = sum([
                ak.sum(ak.from_parquet(inp.fn).normalization_weight[:max_events_per_fold])
                for inp in infiletargets],
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

        for dataset, infiletargets in input["events"].items():
            this_proc_idx = dataset_proc_idx[dataset]
            proc_name = self.processes[this_proc_idx]
            N_events_proc = N_events_processes[this_proc_idx]
            sum_eventweights_proc = sum_eventweights_processes[this_proc_idx]

            logger.info(
                f"dataset: {dataset}, \n  #Events: {N_events_proc}, "
                f"\n  Sum Eventweights: {sum_eventweights_proc}",
            )
            sum_nnweights = 0

            # for inp in infiletargets:
            #     with ChunkedIOHandler(
            #             inp.path,
            #             source_type="awkward_parquet",
            #             # chunk_size=200000,
            #     ) as handler:
            #         for events, position in handler:
            for inp in infiletargets:
                if True:
                    if True:
                        events = ak.from_parquet(inp.path)[:max_events_per_fold]
                        weights = events.normalization_weight
                        if self.eqweight:
                            weights = weights * weights_scaler / sum_eventweights_proc
                            custom_procweight = self.custom_procweights[proc_name]
                            weights = weights * custom_procweight

                        weights = ak.to_numpy(weights)

                        sum_nnweights += sum(weights)
                        sum_nnweights_processes.setdefault(proc_name, 0)
                        sum_nnweights_processes[proc_name] += sum(weights)

                        # logger.info("weights, min, max:", weights[:5], ak.min(weights), ak.max(weights))

                        # transform events into numpy array and transpose
                        inputs = np.transpose(ak.to_numpy(ak.Array(
                            [events[var] for var in self.input_features],
                        ))).astype("float32")
                        # create the truth values for the output layer
                        target = np.zeros((len(events), len(self.processes)))
                        target[:, this_proc_idx] = 1

                        if DNN_inputs["weights"] is None:
                            DNN_inputs["weights"] = weights
                            DNN_inputs["inputs"] = inputs
                            DNN_inputs["target"] = target
                        else:
                            DNN_inputs["weights"] = np.concatenate([DNN_inputs["weights"], weights])
                            DNN_inputs["inputs"] = np.concatenate([DNN_inputs["inputs"], inputs])
                            DNN_inputs["target"] = np.concatenate([DNN_inputs["target"], target])

            logger.info("   weights:", weights[:5])
            logger.info(f"  Sum NN weights: {sum_nnweights}")

        logger.info(sum_nnweights_processes)

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
        for k, vals in DNN_inputs.items():
            # shuffle inputs and split them into validation and train
            vals = vals[shuffle_indices]

            validation[k] = vals[:N_validation_events]
            train[k] = vals[N_validation_events:]

        return train, validation

    def train(
        self,
        task: law.Task,
        input: Any,
        output: law.LocalDirectoryTarget,
    ) -> ak.Array:
        # np.random.seed(1337)  # for reproducibility

        #
        # input preparation
        #

        train, validation = self.prepare_inputs(task, input)
        logger.info("garbage collected")
        gc.collect()

        #
        # model preparation
        #

        N_inputs = len(self.input_features)
        N_outputs = len(self.processes)

        from keras.models import Sequential
        from keras.layers import Dense, BatchNormalization, Dropout

        # define the DNN model
        # TODO: do this Funcional instead of Sequential
        model = Sequential()

        # BatchNormalization layer with input shape
        model.add(BatchNormalization(input_shape=(N_inputs,)))

        # first layer with input shape
        # model.add(Dense(self.layers[0], activation="relu", input_shape=(N_inputs,)))

        # following hidden layers
        # for layer in self.layers[1:]:
        for layer in self.layers:
            model.add(Dense(layer, activation="relu"))
            # Potentially add dropout layer after each hidden layer
            if self.dropout:
                model.add(Dropout(self.dropout))

        # output layer
        model.add(Dense(N_outputs, activation="softmax"))

        # compile the network
        # optimizer = keras.optimizers.SGD(learning_rate=self.learningrate)
        # NOTE: decay is deprecated, therefore use legacy.Adam for now
        optimizer = keras.optimizers.legacy.Adam(
            lr=self.learningrate, beta_1=0.9, beta_2=0.999,
            epsilon=1e-6, decay=0.0, amsgrad=False,
        )
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["categorical_accuracy"])

        #
        # training
        #

        # TODO: early stopping to determine the 'best' model

        # train the model
        logger.info("Start training...")
        model.fit(
            x=train["inputs"], y=train["target"], epochs=self.epochs, sample_weight=train["weights"],
            validation_data=(validation["inputs"], validation["target"], validation["weights"]),
            shuffle=True, verbose=1,
            batch_size=self.batchsize,
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

        inputs = np.transpose(ak.to_numpy(ak.Array([events[var] for var in self.input_features])))

        # do prediction for all models and all events
        predictions = []
        for i, model in enumerate(models):
            pred = ak.from_numpy(model.predict_on_batch(inputs))
            if len(pred[0]) != len(self.processes):
                raise Exception("Number of output nodes should be equal to number of processes")
            predictions.append(pred)

            # Save predictions for each model
            # TODO: create train/val/test plots (confusion, ROC, nodes) using these predictions?
            for j, proc in enumerate(self.processes):
                events = set_ak_column(
                    events, f"{self.cls_name}.fold{i}_score_{proc}", pred[:, j],
                )

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

        return events
