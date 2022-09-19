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

keras = maybe_import("tensorflow.keras")
Dense = maybe_import("tensorflow.keras.layers.Dense")


class SimpleDNN(MLModel):

    def __init__(self, *args, folds: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)

        # class- to instance-level attributes
        # (before being set, self.folds refers to a class-level attribute)
        self.folds = folds or self.folds

        # define output classes (processes)
        self.processes = ["hh_ggf_kt_1_kl_1_bbww_sl", "tt", "st"]

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
            self.config_inst.get_dataset("hh_ggf_kt_1_kl_1_bbww_sl_powheg"),
            self.config_inst.get_dataset("st_tchannel_t_powheg"),
            self.config_inst.get_dataset("tt_sl_powheg"),
        }

    def uses(self) -> Set[Union[Route, str]]:
        return {"ht", "m_bb", "deltaR_bb", "normalization_weight", "mc_weight"}

    def produces(self) -> Set[Union[Route, str]]:
        return {f"{self.cls_name}.score"}

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"mlmodel_f{task.fold}of{self.folds}", dir=True)

    def open_model(self, target: law.LocalDirectoryTarget) -> tf.keras.models.Model:
        return tf.keras.models.load_model(target.path)

    def train(
        self,
        task: law.Task,
        input: Any,
        output: law.LocalDirectoryTarget,
    ) -> ak.Array:
        # np.random.seed(1337)  # for reproducibility
        from keras.layers import Dense

        # parameters (TODO: as input parameters)
        N_inputs = len(self.used_columns) - 2
        layers = [32, 32, 32]  # [512, 512, 512]
        N_outputs = 3  # number of processes
        eqweight = False
        learningrate = 0.001
        # batchsize = -1
        epochs = 5  # 200

        process_insts = [self.config_inst.get_process(proc) for proc in self.processes]
        N_events_proc = len(self.processes) * [(self.folds - 1) * [0]]
        dataset_proc_idx = {}  # bookkeeping which process each dataset belongs to

        # determine process of each dataset and count number of events for this process
        for dataset, infiletargets in input.items():
            # print("dataset:", dataset)
            dataset_inst = self.config_inst.get_dataset(dataset)
            if len(dataset_inst.processes) != 1:
                raise Exception("only 1 process inst is expected for each dataset")

            N_events = [len(ak.from_parquet(inp.fn)) for inp in infiletargets]

            for i, proc in enumerate(process_insts):
                # print("process:", proc.name)
                # NOTE: here are some assumptions made, should check if they hold true for each relevant process
                leaf_procs = [proc] if proc.is_leaf_process else proc.get_leaf_processes()
                if dataset_inst.processes.get_first() in leaf_procs:
                    dataset_proc_idx[dataset] = i
                    for j, N_evt in enumerate(N_events):
                        N_events_proc[i][j] += N_evt
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
            # print("dataset:", dataset)
            this_proc_idx = dataset_proc_idx[dataset]

            event_folds = [ak.from_parquet(inp.fn) for inp in infiletargets]

            for i, events in enumerate(event_folds):  # i is in [0, self.folds-2]

                weights = events.normalization_weight
                if eqweight:
                    weights *= N_events_proc[this_proc_idx] / sum(N_events_proc)

                weights = ak.to_numpy(weights)

                events = remove_ak_column(events, "mc_weight")
                events = remove_ak_column(events, "normalization_weight")

                # bookkeep input feature names (order corresponds to order in the NN input)
                features = events.fields
                print("features:", features)

                # transform events into numpy array and transpose
                events = np.transpose(ak.to_numpy(ak.Array(ak.unzip(events))))

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

        train, validation = {}, {}  # combine all except one input as train input, use last one for validation
        for k, vals in NN_inputs.items():
            train[k] = np.concatenate(vals[:-1])
            validation[k] = vals[-1]

        # define the DNN model
        model = keras.models.Sequential()

        # first layer with input shape
        model.add(Dense(layers[0], activation="relu", input_shape=(N_inputs,)))

        # following hidden layers
        for layer in layers[1:]:
            model.add(Dense(layer, activation="relu"))

        # output layer
        model.add(Dense(N_outputs, activation="softmax"))

        # compile the network
        optimizer = keras.optimizers.SGD(learning_rate=learningrate)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["categorical_accuracy"])

        # train the model (output history here maybe?)
        model.fit(
            x=train["inputs"], y=train["target"], epochs=epochs, sample_weight=train["weights"],
            validation_data=(validation["inputs"], validation["target"], validation["weights"]),
            shuffle=True, verbose=0,
        )

        output.parent.touch()
        model.save(output.path)

    def evaluate(
        self,
        task: law.Task,
        events: ak.Array,
        models: List[Any],
        fold_indices: ak.Array,
        events_used_in_training: bool = True,
    ) -> None:
        print(f"Evaluation of dataset {task.dataset}")
        features = self.used_columns
        features.remove("mc_weight")
        features.remove("normalization_weight")

        inputs = np.transpose(ak.to_numpy(ak.Array([events[var] for var in features])))

        outputs = -1
        for i in range(self.folds):
            print(f"Evaluation fold {i}")
            outputs = np.where(fold_indices == i, models[i].predict_on_batch(inputs), outputs)

        if len(outputs, 0) != len(self.processes):
            raise Exception("number of output nodes should be equal to number of processes")

        for i, proc in enumerate(self.processes):
            events = set_ak_column(events, f"{self.cls_name}.score", outputs[:, i])

        return events


# usable derivations
simple_dnn = SimpleDNN.derive("simple", cls_dict={"folds": 5})
