"""
Collection of classes to load and prepare data for machine learning.
"""

from __future__ import annotations

import law
import order as od

from columnflow.util import maybe_import
from columnflow.columnar_util import remove_ak_column
from columnflow.ml import MLModel
from hbw.ml.helper import predict_numpy_on_batch
from hbw.util import timeit

ak = maybe_import("awkward")
np = maybe_import("numpy")

logger = law.logger.get_logger(__name__)


def get_proc_mask(
    events: ak.Array,
    proc: str | od.Process,
    config_inst: od.Config | None = None,
) -> np.ndarray:
    """
    Creates a list of the Ids of all subprocesses and teh corresponding mask for all events.

    :param events: Event array
    :param config_inst: An instance of the Config, can be None if Porcess instance is given.
    :param proc: Either string or process instance.
    """
    # get process instance
    if config_inst:
        proc_inst = config_inst.get_process(proc)
    elif isinstance(proc, od.Process):
        proc_inst = proc

    proc_id = events.process_id
    unique_proc_ids = set(proc_id)

    # get list of Ids that are belonging to the process and are present in the event array
    sub_id = [
        proc_inst.id
        for proc_inst, _, _ in proc_inst.walk_processes(include_self=True)
        if proc_inst.id in unique_proc_ids
    ]

    # Create process mask
    proc_mask = np.isin(proc_id, sub_id)
    return proc_mask, sub_id


def del_sub_proc_stats(
    stats: dict,
    proc: str,
    sub_id: list,
) -> np.ndarray:
    """
    Function deletes dict keys which are not part of the requested process

    :param stats: Dictionaire containing ML stats for each process.
    :param proc: String of the process.
    :param sub_id: List of ids of sub processes that should be reatined (!).
    """
    id_list = list(stats[proc]["num_events_per_process"].keys())
    item_list = list(stats[proc].keys())
    for id in id_list:
        if int(id) not in sub_id:
            for item in item_list:
                if "per_process" in item:
                    del stats[proc][item][id]


def input_features_sanity_checks(ml_model_inst: MLModel, input_features: list[str]):
    """
    Perform sanity checks on the input features.

    :param ml_model_inst: An instance of the MLModel class.
    :param input_features: A list of strings representing the input features.
    :raises Exception: If input features are not ordered in the same way for all datasets.
    """
    # check if input features are ordered in the same way for all datasets
    if getattr(ml_model_inst, "input_features_ordered", None):
        if ml_model_inst.input_features_ordered != input_features:
            raise Exception(
                f"Input features are not ordered in the sme way for all datasets. "
                f"Expected: {ml_model_inst.input_features_ordered}, "
                f"got: {input_features}",
            )
    else:
        # if not already set, bookkeep input features in the ml_model_inst aswell
        ml_model_inst.input_features_ordered = input_features

    # check that the input features contain exactly what was requested by the MLModel
    if set(input_features) != set(ml_model_inst.input_features):
        raise Exception(
            f"Input features do not match the input features requested by the MLModel. "
            f"Expected: {ml_model_inst.input_features}, got: {input_features}",
        )


class MLDatasetLoader:
    """
    Helper class to conveniently load ML training data from an awkward array.

    Depends on following parameters of the ml_model_inst:
    - input_features: A set of strings representing the input features we want to keep.
    - train_val_test_split: A tuple of floats representing the split of the data into training, validation, and testing.
    - processes: A tuple of strings representing the processes. Can be parallelized over.
    """

    # shuffle the data in *load_split_data* method
    shuffle: bool = True

    input_arrays: tuple = ("features", "weights", "train_weights", "equal_weights")
    # input_arrays: tuple = ("features", "weights", _
    #         _ "equal_train_weights", "xsec_train_weights", "train_weights", "equal_weights")
    evaluation_arrays: tuple = ("prediction",)

    def __init__(
        self,
        ml_model_inst: MLModel,
        process: "str",
        events: ak.Array,
        stats: dict | None = None,
        skip_mask=False,
    ):
        """
        Initializes the MLDatasetLoader with the given parameters.

        :param ml_model_inst: An instance of the MLModel class.
        :param process: A string representing the process.
        :param events: An awkward array representing the events.
        :param stats: A dictionary containing merged stats per training process.
        :raises Exception: If input features are not ordered in the same way for all datasets.

        .. note:: The method prepares the weights, bookkeeps the order of input features,
        removes columns that are not used as training features, and transforms events into a numpy array.
        """
        self._ml_model_inst = ml_model_inst
        self._process = process

        proc_mask, _ = get_proc_mask(events, process, ml_model_inst.config_inst)
        self._stats = stats
        # del_sub_proc_stats(process, sub_id)
        if not skip_mask:
            self._events = events[proc_mask]
            self._events = events[events.event_weight >= 0.0]
        else:
            self._events = events

    def __repr__(self):
        return f"{self.__class__.__name__}({self.ml_model_inst.cls_name}, {self.process})"

    @property
    def hyperparameter_deps(self) -> set:
        """
        Hyperparameters that are required to be set in the MLModel class. If they are changed,
        then tasks using this class need to be re-run.
        """
        # TODO: store values of hyperparameters as task output
        return {"input_features", "train_val_test_split", "input_features_ordered"}

    @property
    def parameters(self):
        """
        Values of the MLModel parameters that the MLDatasetLoader depends on.
        """
        if hasattr(self, "_parameters"):
            return self._parameters

        self._parameters = {
            param: getattr(self.ml_model_inst, param, None)
            for param in self.hyperparameter_deps
        }
        return self._parameters

    @property
    def ml_model_inst(self):
        return self._ml_model_inst

    @property
    def process(self):
        return self._process

    @property
    def process_inst(self):
        return self.ml_model_inst.config_inst.get_process(self.process)

    @property
    def input_features(self) -> tuple:
        if not hasattr(self, "_input_features"):
            # input features are initialized with the features propery
            self.features

        return self._input_features

    @property
    def stats(self) -> dict:
        return self._stats

    @property
    def weights(self) -> np.ndarray:
        if not hasattr(self, "_weights"):
            self._weights = ak.to_numpy(self._events.event_weight).astype(np.float32)

        return self._weights

    @property
    def features(self) -> np.ndarray:
        if hasattr(self, "_features"):
            return self._features

        # work with a copy of the events
        features = self._events

        # remove columns that are not used as training features
        for var in features.fields:
            if var not in self.ml_model_inst.input_features:
                features = remove_ak_column(features, var)

        # bookkeep order of input features and perform sanity checks
        self._input_features = tuple(features.fields)
        input_features_sanity_checks(self.ml_model_inst, self._input_features)

        # transform features into numpy npdarray
        # NOTE: when converting to numpy, the awkward array seems to stay in memory...
        features = ak.to_numpy(features)
        features = features.astype(
            [(name, np.float32) for name in features.dtype.names], copy=False,
        ).view(np.float32).reshape((-1, len(features.dtype)))

        # check for infinite values
        if np.any(~np.isfinite(features)):
            raise Exception(f"Found non-finite values in input features for process {self.process}.")

        self._features = features

        return self._features

    @property
    def n_events(self) -> int:
        if hasattr(self, "_n_events"):
            return self._n_events
        self._n_events = len(self.weights)
        return self._n_events

    @property
    def shuffle_indices(self) -> np.ndarray:
        if hasattr(self, "_shuffle_indices"):
            return self._shuffle_indices

        self._shuffle_indices = np.random.permutation(self.n_events)
        return self._shuffle_indices

    def get_xsec_train_weights(self) -> np.ndarray:
        """
        Weighting such that each event has roughly the same weight,
        sub processes are weighted accoridng to their cross section
        """
        if hasattr(self, "_xsec_train_weights"):
            return self._xsec_train_weights

        if not self.stats:
            raise Exception("cannot determine train weights without stats")

        _, sub_id = get_proc_mask(self._events, self.process, self.ml_model_inst.config_inst)
        sum_abs_weights = np.sum([self.stats[self.process]["sum_abs_weights_per_process"][str(id)] for id in sub_id])
        num_events = np.sum([self.stats[self.process]["num_events_per_process"][str(id)] for id in sub_id])

        xsec_train_weights = self.weights / sum_abs_weights * num_events

        return xsec_train_weights

    def get_equal_train_weights(self) -> np.ndarray:
        """
        Weighting such that events of each sub processes are weighted equally
        """
        if hasattr(self, "_equally_train_weights"):
            return self._equal_train_weights

        if not self.stats:
            raise Exception("cannot determine train weights without stats")

        combined_proc_inst = self.ml_model_inst.config_inst.get_process(self.process)
        _, sub_id_proc = get_proc_mask(self._events, self.process, self.ml_model_inst.config_inst)
        num_events = np.sum([self.stats[self.process]["num_events_per_process"][str(id)] for id in sub_id_proc])
        targeted_sum_of_weights_per_process = (
            num_events / len(combined_proc_inst.x.ml_config.sub_processes)
        )
        equal_train_weights = ak.full_like(self.weights, 1.)
        sub_class_factors = {}

        for proc in combined_proc_inst.x.ml_config.sub_processes:
            proc_mask, sub_id = get_proc_mask(self._events, proc, self.ml_model_inst.config_inst)
            sum_pos_weights_per_sub_proc = 0.
            sum_pos_weights_per_proc = self.stats[self.process]["sum_pos_weights_per_process"]

            for id in sub_id:
                id = str(id)
                if id in self.stats[self.process]["num_events_per_process"]:
                    sum_pos_weights_per_sub_proc += sum_pos_weights_per_proc[id]

            if sum_pos_weights_per_sub_proc == 0:
                norm_const_per_proc = 1.
                logger.info(
                    f"No weight sum found in stats for sub process {proc}."
                    f"Normalization constant set to 1 but results are probably not correct.")
            else:
                norm_const_per_proc = targeted_sum_of_weights_per_process / sum_pos_weights_per_sub_proc
                logger.info(f"Normalizing constant for {proc} is {norm_const_per_proc}")

            sub_class_factors[proc] = norm_const_per_proc
            equal_train_weights = np.where(proc_mask, self.weights * norm_const_per_proc, equal_train_weights)

        return equal_train_weights

    @property
    def train_weights(self) -> np.ndarray:
        """
        Weighting according to the parameters set in the ML model config
        """
        if hasattr(self, "_train_weights"):
            return self._train_weights

        if not self.stats:
            raise Exception("cannot determine train weights without stats")

        # TODO: hier muss np.float gemacht werden
        proc = self.process
        proc_inst = self.ml_model_inst.config_inst.get_process(proc)
        if proc_inst.x("ml_config", None) and proc_inst.x.ml_config.weighting == "equal":
            train_weights = self.get_equal_train_weights()
        else:
            train_weights = self.get_xsec_train_weights()
        #     self._train_weights = self.get_equal_train_weights()
        # else:
        #     self._train_weights = self.get_xsec_train_weights()

        self._train_weights = ak.to_numpy(train_weights).astype(np.float32)

        return self._train_weights

    @property
    def equal_weights(self) -> np.ndarray:
        """
        Weighting such that each process has roughly the same sum of weights
        """
        if hasattr(self, "_validation_weights"):
            return self._validation_weights

        if not self.stats:
            raise Exception("cannot determine val weights without stats")

        # TODO: per process pls [done] and now please tidy up
        processes = self.ml_model_inst.processes
        num_events_per_process = {}
        for proc in processes:
            id_list = list(self.stats[proc]["num_events_per_process"].keys())
            proc_inst = self.ml_model_inst.config_inst.get_process(proc)
            sub_id = [
                p_inst.id
                for p_inst, _, _ in proc_inst.walk_processes(include_self=True)
                if str(p_inst.id) in id_list
            ]
            if proc == self.process:
                sum_abs_weights = np.sum([
                    self.stats[self.process]["sum_abs_weights_per_process"][str(id)] for id in sub_id
                ])
            num_events_per_proc = np.sum([self.stats[proc]["num_events_per_process"][str(id)] for id in sub_id])
            num_events_per_process[proc] = num_events_per_proc

        # sum_abs_weights = self.stats[self.process]["sum_abs_weights"]
        # num_events_per_process = {proc: self.stats[proc]["num_events"] for proc in processes}
        validation_weights = self.weights / sum_abs_weights * max(num_events_per_process.values())
        self._validation_weights = ak.to_numpy(validation_weights).astype(np.float32)

        return self._validation_weights

    @property
    def labels(self) -> np.ndarray:
        raise Exception(
            "This should not be used anymore since we now create the labels during training/evaluation"
            "to allow sharing these outputs between ML models with different sets of processes.",
        )
        if hasattr(self, "_labels"):
            return self._labels

        if not self.process_inst.has_aux("ml_id"):
            logger.warning(
                f"Process {self.process} does not have an ml_id. Label will be set to -1.",
            )
            self._labels = np.ones(self.n_events) * -1
            return self._labels
        elif self.process_inst.x.ml_id not in range(len(self.ml_model_inst.processes)):
            raise Exception(
                f"ml_id {self.process_inst.x.ml_id} of process {self.process} not in range of processes "
                f"{self.ml_model_inst.processes}. Cannot create target array.",
            )

        self._labels = np.ones(self.n_events) * self.process_inst.x.ml_id
        return self._labels

    @property
    def target(self) -> np.ndarray:
        raise Exception(
            "This should not be used anymore since we now create the labels during training/evaluation"
            "to allow sharing these outputs between ML models with different sets of processes.",
        )
        if hasattr(self, "_target"):
            return self._target

        self._target = np.zeros((self.n_events, len(self.ml_model_inst.processes))).astype(np.float32)

        if not self.process_inst.has_aux("ml_id"):
            logger.warning(
                f"Process {self.process} does not have an ml_id. Target will be set to 0 for all classes.",
            )
            return self._target
        elif self.process_inst.x.ml_id not in range(len(self.ml_model_inst.processes)):
            raise Exception(
                f"ml_id {self.process_inst.x.ml_id} of process {self.process} not in range of processes "
                f"{self.ml_model_inst.processes}. Cannot create target array.",
            )

        self._target[:, self.process_inst.x.ml_id] = 1
        return self._target

    @property
    def get_data_split(self) -> tuple[int, int]:
        """
        Get the data split for training, validation and testing.

        :param data: The data to be split.
        :return: The end indices for the training and validation data.
        """
        if hasattr(self, "_train_end") and hasattr(self, "_val_end"):
            return self._train_end, self._val_end

        data_split = np.array(self.ml_model_inst.train_val_test_split)
        data_split = data_split / np.sum(data_split)

        self._train_end = int(data_split[0] * self.n_events)
        self._val_end = int((data_split[0] + data_split[1]) * self.n_events)

        return self._train_end, self._val_end

    def load_split_data(self, data: np.array | str) -> tuple[np.ndarray]:
        """
        Function to split data into training, validation, and test sets.

        :param data: The data to be split. If a string is provided, it is treated as an attribute name.
        :return: The training, validation, and test data.
        """
        if isinstance(data, str):
            data = getattr(self, data)
        train_end, val_end = self.get_data_split

        if self.shuffle:
            data = data[self.shuffle_indices]

        return data[:train_end], data[train_end:val_end], data[val_end:]


class MLProcessData:
    """
    Helper class to conveniently load ML training data from the MLPreTraining task outputs.

    Data is merged for all folds except the evaluation_fold.

    Implements the following parameters of the ml_model_inst:
    - negative_weights: A string representing the handling of negative weights.
    """

    shuffle = False

    input_arrays: tuple = ("features", "weights", "train_weights", "equal_weights", "target", "labels")
    evaluation_arrays: tuple = ("prediction",)

    def __init__(
        self,
        ml_model_inst: MLModel,
        inputs,
        data_split: str,
        processes: str,
        evaluation_fold: int,
        fold_modus: str = "all_except_evaluation_fold",
    ):
        self._ml_model_inst = ml_model_inst

        self._input = inputs
        self._data_split = data_split
        self._processes = law.util.make_list(processes)
        self._evaluation_fold = evaluation_fold

        assert fold_modus in ("all_except_evaluation_fold", "evaluation_only", "all")
        self._fold_modus = fold_modus

        # initialize input features
        self.input_features

    def __del__(self):
        """
        Destructor for the MLDatasetLoader class.

        This method is called when the object is about to be destroyed.
        It deletes the attributes that are numpy arrays to free up memory.
        """
        for attr in ("features", "weights", "train_weights", "equal_weights", "target", "labels"):
            if hasattr(self, f"_{attr}"):
                delattr(self, f"_{attr}")

        del self

    def __repr__(self):
        return f"{self.__class__.__name__}({self._ml_model_inst.cls_name}, {self._data_split}, {self._processes})"

    @property
    def process_insts(self) -> list[od.process]:
        return [self._ml_model_inst.config_inst.get_process(proc) for proc in self._processes]

    @property
    def shuffle_indices(self) -> np.ndarray:
        if hasattr(self, "_shuffle_indices"):
            return self._shuffle_indices

        self._shuffle_indices = np.random.permutation(self.n_events)
        return self._shuffle_indices

    @property
    def input_features(self) -> tuple[str]:
        if hasattr(self, "_input_features"):
            return self._input_features

        # load input features for all folds and check consistency between them and with the ml_model_inst
        for process in self._processes:
            for i in range(self._ml_model_inst.folds):
                self._input_features = self._input["input_features"][process][i].load(formatter="pickle")
                input_features_sanity_checks(self._ml_model_inst, self._input_features)

        return self._input_features

    @property
    def n_events(self) -> int:
        if hasattr(self, "_n_events"):
            return self._n_events

        # NOTE: this requires us to load labels. Might not be the optimal choice
        self._n_events = len(self.labels)
        return self._n_events

    @property
    def folds(self) -> tuple[int]:
        """ Property to set the folds for which to merge the data """
        if hasattr(self, "_folds"):
            return self._folds

        if self._fold_modus == "all_except_evaluation_fold":
            self._folds = list(range(self._ml_model_inst.folds))
            self._folds.remove(self._evaluation_fold)
        elif self._fold_modus == "evaluation_only":
            self._folds = [self._evaluation_fold]
        elif self._fold_modus == "all":
            self._folds = list(range(self._ml_model_inst.folds))
        else:
            raise Exception(f"unknown fold modus {self._fold_modus} for MLProcessData")
        return self._folds

    @timeit
    def load_all(self):
        """
        Convenience function to load all data into memory.
        """
        logger.info(f"Loading all data for processes {self._processes} in {self._data_split} set in memory.")
        self.features
        self.weights
        self.train_weights
        self.equal_weights
        self.target
        self.labels
        # do not load prediction because it can only be loaded after training
        # self.prediction

    def load_file(self, data_str, data_split, process, fold):
        """
        Load a file from the input dictionary.
        """
        return self._input[data_str][data_split][process][fold].load(formatter="numpy")

    def load_labels(self, data_split, process, fold):
        """
        Load the labels for a given process and fold.
        """
        proc_inst = self._ml_model_inst.config_inst.get_process(process)
        if not proc_inst.has_aux("ml_id"):
            logger.warning(
                f"Process {process} does not have an ml_id. Label will be set to -1.",
            )
            ml_id = -1
        else:
            ml_id = proc_inst.x.ml_id

        # load any column to get the array length
        weights = self.load_file("weights", data_split, process, fold)

        labels = np.ones(len(weights), dtype=np.int32) * ml_id
        return labels

    def load_data(self, data_str: str) -> np.ndarray:
        """
        Load data from the input dictionary. Options for data_str are "features", "weights", "train_weights",
        "equal_weights", "labels", and "prediction".
        When the data is loaded, it is concatenated over all processes and folds.
        When the *shuffle* attribute is set to True, the data is shuffled using the *shuffle_indices* attribute.
        """
        if data_str not in ("features", "weights", "train_weights", "equal_weights", "labels", "prediction"):
            logger.warning(f"Unknown data string {data_str} for MLProcessData.")
        data = []
        for process in self._processes:
            for fold in self.folds:
                if data_str == "labels":
                    fold_data = self.load_labels(self._data_split, process, fold)
                else:
                    fold_data = self.load_file(data_str, self._data_split, process, fold)
                if np.any(~np.isfinite(fold_data)):
                    raise Exception(f"Found non-finite values in {data_str} for {process} in fold {fold}.")
                data.append(fold_data)

        data = np.concatenate(data)
        if self.shuffle:
            data = data[self.shuffle_indices]
        return data

    @property
    def features(self) -> np.ndarray:
        if hasattr(self, "_features"):
            return self._features

        self._features = self.load_data("features")
        return self._features

    @property
    def weights(self) -> np.ndarray:
        if hasattr(self, "_weights"):
            return self._weights

        self._weights = self.load_data("weights")
        return self._weights

    @property
    def m_negative_weights(self) -> np.ndarray:
        if hasattr(self, "_m_negative_weights"):
            return self._m_negative_weights

        # if not already done, run the *train_weights* method that also initializes the m_negative_weights
        self.train_weights
        return self._m_negative_weights

    @property
    def train_weights(self) -> np.ndarray:
        if hasattr(self, "_train_weights"):
            return self._train_weights

        train_weights = self.load_data("train_weights")
        self._m_negative_weights = (train_weights < 0)

        # handling of negative weights based on the ml_model_inst.negative_weights parameter
        if self._ml_model_inst.negative_weights == "ignore":
            train_weights[self._m_negative_weights] = 0
        elif self._ml_model_inst.negative_weights == "abs":
            train_weights = np.abs(train_weights)
        elif self._ml_model_inst.negative_weights == "handle":
            train_weights[self._m_negative_weights] = (
                np.abs(train_weights[self._m_negative_weights]) / (len(self._ml_model_inst.processes) - 1)
            )
        elif self._ml_model_inst.negative_weights == "nothing":
            train_weights = train_weights

        self._train_weights = train_weights
        return self._train_weights

    @property
    def equal_weights(self) -> np.ndarray:
        if hasattr(self, "_equal_weights"):
            return self._equal_weights

        self._equal_weights = self.load_data("equal_weights")
        return self._equal_weights

    @property
    def target(self) -> np.ndarray:
        if hasattr(self, "_target"):
            return self._target

        # use the labels to create the target array
        labels = self.labels
        target = np.eye(len(self._ml_model_inst.processes))[labels]

        # handling of negative weights based on the ml_model_inst.negative_weights parameter
        if self._ml_model_inst.negative_weights == "handle":
            target[self.m_negative_weights] = 1 - target[self.m_negative_weights]

        # NOTE: I think here the targets are somehow 64floats... Maybe check that
        self._target = target
        return self._target

    @property
    def labels(self) -> np.ndarray:
        if hasattr(self, "_labels"):
            return self._labels

        self._labels = self.load_data("labels")
        return self._labels

    @property
    def prediction(self) -> np.ndarray:
        if hasattr(self, "_prediction"):
            return self._prediction

        if "prediction" in self._input.keys():
            # load prediction if possible
            self._prediction = self.load_data("prediction")
        else:
            # calcluate prediction if needed
            if not hasattr(self._ml_model_inst, "trained_model"):
                raise Exception("No trained model found in the MLModel instance. Cannot calculate prediction.")
            self._prediction = predict_numpy_on_batch(self._ml_model_inst.trained_model, self.features)

        return self._prediction  # TODO ML best model
