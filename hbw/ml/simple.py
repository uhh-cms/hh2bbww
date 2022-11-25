# coding: utf-8

"""
First implementation of DNN for HH analysis
"""

from typing import List, Any, Set, Union, Optional

import law
import order as od

from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import Route, set_ak_column, remove_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")
tf = maybe_import("tensorflow")
pickle = maybe_import("pickle")
keras = maybe_import("tensorflow.keras")


class SimpleDNN(MLModel):

    def __init__(self, *args, folds: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)

        # class- to instance-level attributes
        # (before being set, self.folds refers to a class-level attribute)
        self.folds = folds or self.folds

        # define output classes (processes)
        self.processes = ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "st"]

        # the custom process weight should be chosen such that individual eventweights are close to 1
        # but can also be used to optimize relative weights between processes
        self.custom_procweights = {
            "ggHH_kl_1_kt_1_sl_hbbhww": 1 / 1000,
            "tt": 1 / 1000,
            "st": 1 / 1000,
        }

        # DNN model parameters
        self.layers = [512, 512, 512]
        self.learningrate = 0.00050
        self.batchsize = 2048
        self.epochs = 6  # 200
        self.eqweight = 0.50

        # Dropout: either False (disable) or a value between 0 and 1 (dropout_rate)
        self.dropout = False

        # dynamically add variables for the quantities produced by this model
        for proc in self.processes:
            if f"{self.cls_name}.score_{proc}" not in self.config_inst.variables:
                self.config_inst.add_variable(
                    name=f"{self.cls_name}.score_{proc}",
                    null_value=-1,
                    binning=(40, 0., 1.),
                    x_title=f"DNN output score {self.config_inst.get_process(proc).label}",
                )

    def sandbox(self, task: law.Task) -> str:
        return dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_tf.sh")

    def datasets(self) -> Set[od.Dataset]:
        return {
            self.config_inst.get_dataset("ggHH_kl_1_kt_1_sl_hbbhww_powheg"),
            self.config_inst.get_dataset("st_tchannel_t_powheg"),
            self.config_inst.get_dataset("tt_sl_powheg"),
        }

    def uses(self) -> Set[Union[Route, str]]:
        return {"ht", "m_bb", "deltaR_bb", "normalization_weight"}

    def produces(self) -> Set[Union[Route, str]]:
        produced = set({})
        for proc in self.processes:
            produced.add(f"{self.cls_name}.score_{proc}")
        return produced

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"mlmodel_f{task.fold}of{self.folds}", dir=True)

    def open_model(self, target: law.LocalDirectoryTarget) -> tf.keras.models.Model:
        with open(f"{target.path}/model_history.pkl", "rb") as f:
            history = pickle.load(f)
        model = tf.keras.models.load_model(target.path)
        return model, history

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

        N_inputs = len(self.used_columns) - 1  # don't count the weight columns
        N_outputs = len(self.processes)

        process_insts = [self.config_inst.get_process(proc) for proc in self.processes]
        N_events_proc = np.array(len(self.processes) * [(self.folds - 1) * [0]])
        sum_eventweights_proc = np.array(len(self.processes) * [(self.folds - 1) * [0]])
        dataset_proc_idx = {}  # bookkeeping which process each dataset belongs to

        # determine process of each dataset and count number of events & sum of eventweights for this process
        for dataset, infiletargets in input.items():
            dataset_inst = self.config_inst.get_dataset(dataset)
            if len(dataset_inst.processes) != 1:
                raise Exception("only 1 process inst is expected for each dataset")

            N_events = [len(ak.from_parquet(inp.fn)) for inp in infiletargets]
            sum_eventweights = [ak.sum(ak.from_parquet(inp.fn).normalization_weight) for inp in infiletargets]

            for i, proc in enumerate(process_insts):
                # NOTE: here are some assumptions made, should check if they hold true for each relevant process
                leaf_procs = [proc] if proc.is_leaf_process else proc.get_leaf_processes()
                if dataset_inst.processes.get_first() in leaf_procs:
                    print(f"the dataset {dataset} counts as process {proc.name}")
                    dataset_proc_idx[dataset] = i
                    for j, N_evt in enumerate(N_events):
                        N_events_proc[i][j] += N_evt

                    for j, sumw in enumerate(sum_eventweights):
                        sum_eventweights_proc[i][j] += sumw
                    continue
            if dataset_proc_idx.get(dataset, -1) == -1:
                raise Exception(f"dataset {dataset} is not matched to any of the given processes")

        # set inputs, weights and targets for each datset and fold
        NN_inputs = {
            "weights": [],
            "inputs": [],
            "target": [],
        }

        for dataset, infiletargets in input.items():
            this_proc_idx = dataset_proc_idx[dataset]
            print(f"dataset: {dataset}, \n  #Events: {sum(N_events_proc[this_proc_idx])}, "
                  f"\n  Sum Normweights: {sum(sum_eventweights_proc[this_proc_idx])}")

            event_folds = [ak.from_parquet(inp.fn) for inp in infiletargets]

            sum_nnweights = 0

            for i, events in enumerate(event_folds):  # i is in [0, self.folds-2]
                weights = events.normalization_weight
                if self.eqweight:
                    weights = weights * sum(sum_eventweights_proc)[i] / sum_eventweights_proc[this_proc_idx][i]
                    custom_procweight = self.custom_procweights[self.processes[this_proc_idx]]
                    weights = weights * custom_procweight

                weights = ak.to_numpy(weights)
                sum_nnweights += sum(weights)

                # print("weights, min, max:", weights[:5], ak.min(weights), ak.max(weights))
                events = remove_ak_column(events, "normalization_weight")

                # bookkeep input feature names (order corresponds to order in the NN input)
                # features = events.fields
                # print("features:", features)

                # transform events into numpy array and transpose
                events = np.transpose(ak.to_numpy(ak.Array(ak.unzip(events))))

                # create the truth values for the output layer
                target = np.zeros((len(events), len(self.processes)))
                target[:, this_proc_idx] = 1

                # add relevant collections to the NN inputs
                if len(NN_inputs["weights"]) <= i:
                    NN_inputs["weights"].append(weights)
                    NN_inputs["inputs"].append(events)
                    NN_inputs["target"].append(target)
                else:
                    NN_inputs["weights"][i] = np.concatenate([NN_inputs["weights"][i], weights])
                    NN_inputs["inputs"][i] = np.concatenate([NN_inputs["inputs"][i], events])
                    NN_inputs["target"][i] = np.concatenate([NN_inputs["target"][i], target])

            print(f"  Sum NN weights: {sum_nnweights}")
        train, validation = {}, {}  # combine all except one input as train input, use last one for validation
        for k, vals in NN_inputs.items():
            # validation set always corresponds to (eval_fold+1) in that way
            validation[k] = vals.pop((task.fold) % (self.folds - 1))
            # print("Number of training folds:", len(vals))
            train[k] = np.concatenate(vals)

        #
        # model preparation
        #
        from keras.models import Sequential
        from keras.layers import Dense, BatchNormalization, Dropout

        # define the DNN model
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
        optimizer = keras.optimizers.Adam(
            lr=self.learningrate, beta_1=0.9, beta_2=0.999,
            epsilon=1e-6, decay=0.0, amsgrad=False,
        )
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["categorical_accuracy"])

        #
        # training
        #

        # TODO: early stopping to determine the 'best' model

        # train the model
        print("Start training...")
        model.fit(
            x=train["inputs"], y=train["target"], epochs=self.epochs, sample_weight=train["weights"],
            validation_data=(validation["inputs"], validation["target"], validation["weights"]),
            shuffle=True, verbose=1,
            batch_size=self.batchsize,
        )
        # save the model and history
        output.parent.touch()
        model.save(output.path)
        with open(f"{output.path}/model_history.pkl", "wb") as f:
            pickle.dump(model.history.history, f)

    def evaluate(
        self,
        task: law.Task,
        events: ak.Array,
        models: List[Any],
        fold_indices: ak.Array,
        events_used_in_training: bool = True,
    ) -> None:
        print(f"Evaluation of dataset {task.dataset}")
        models, history = zip(*models)
        # TODO: use history for loss+acc plotting

        features = self.used_columns
        features.discard("normalization_weight")
        inputs = np.transpose(ak.to_numpy(ak.Array([events[var] for var in features])))

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
            print(f"Evaluation fold {i}")
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


# usable derivations
simple_dnn = SimpleDNN.derive("simple", cls_dict={"folds": 5})
