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
    - processes: A tuple of strings representing the processes.
    """

    input_arrays: tuple = ("features", "weights", "train_weights", "val_weights", "target", "labels")
    evaluation_arrays: tuple = ("prediction",)

    def __init__(self, ml_model_inst: MLModel, process: "str", events: ak.Array, stats: dict | None = None):
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
        self._stats = stats
        self._events = events

    def __repr__(self):
        return f"{self.__class__.__name__}({self.ml_model_inst.cls_name}, {self.process})"

    @property
    def hyperparameter_deps(self):
        """
        Hyperparameters that are required to be set in the MLModel class. If they are changed,
        then tasks using this class need to be re-run.
        """
        # TODO: store values of hyperparameters as task output
        # TODO: we could also reuse task outputs for multiple MLModels with same hyperparameters
        return ("input_features", "train_val_test_split", "processes", "ml_process_weights")

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
            self._weights = ak.to_numpy(self._events.normalization_weight).astype(np.float32)

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

    @property
    def train_weights(self) -> np.ndarray:
        """
        Weighting such that each event has roughly the same weight
        """
        if hasattr(self, "_train_weights"):
            return self._train_weights

        if not self.stats:
            raise Exception("cannot determine train weights without stats")

        sum_abs_weights = self.stats[self.process]["sum_abs_weights"]
        num_events = self.stats[self.process]["num_events"]

        self._train_weights = self.weights / sum_abs_weights * num_events
        return self._train_weights

    @property
    def val_weights(self) -> np.ndarray:
        """
        Weighting such that each process has roughly the same sum of weights
        """
        if hasattr(self, "_validation_weights"):
            return self._validation_weights

        if not self.stats:
            raise Exception("cannot determine val weights without stats")

        processes = self.ml_model_inst.processes
        sum_abs_weights = self.stats[self.process]["sum_abs_weights"]
        num_events_per_process = {proc: self.stats[proc]["num_events"] for proc in processes}

        self._validation_weights = self.weights / sum_abs_weights * max(num_events_per_process.values())

        return self._validation_weights

    @property
    def labels(self) -> np.ndarray:
        if hasattr(self, "_labels"):
            return self._labels

        self._labels = np.ones(self.n_events) * self.process_inst.x.ml_id
        return self._labels

    @property
    def target(self) -> np.ndarray:
        if hasattr(self, "_target"):
            return self._target

        self._target = np.zeros((self.n_events, len(self.ml_model_inst.processes))).astype(np.float32)
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
        return data[:train_end], data[train_end:val_end], data[val_end:]


class MLProcessData:
    """
    Helper class to conveniently load ML training data from the MLPreTraining task outputs.

    Data is merged for all folds except the evaluation_fold.

    Implements the following parameters of the ml_model_inst:
    - negative_weights: A string representing the handling of negative weights.
    """

    shuffle = False

    input_arrays: tuple = ("features", "weights", "train_weights", "val_weights", "target", "labels")
    evaluation_arrays: tuple = ("prediction",)

    def __init__(
        self,
        ml_model_inst: MLModel,
        inputs,
        test_val_train: str,
        processes: str,
        evaluation_fold: int,
        fold_modus: str = "all_except_evaluation_fold",
    ):
        self._ml_model_inst = ml_model_inst

        self._input = inputs
        self._test_val_train = test_val_train
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
        for attr in ("features", "weights", "train_weights", "val_weights", "target", "labels"):
            if hasattr(self, f"_{attr}"):
                delattr(self, f"_{attr}")

        del self

    def __repr__(self):
        return f"{self.__class__.__name__}({self._ml_model_inst.cls_name}, {self._test_val_train}, {self._processes})"

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
        logger.info(f"Loading all data for processes {self._processes} in {self._test_val_train} set in memory.")
        self.features
        self.weights
        self.train_weights
        self.val_weights
        self.target
        self.labels
        # do not load prediction because it can only be loaded after training
        # self.prediction

    def load_data(self, data_str: str) -> np.ndarray:
        data = []
        for process in self._processes:
            for fold in self.folds:
                fold_data = self._input[data_str][self._test_val_train][process][fold].load(formatter="numpy")
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
    def val_weights(self) -> np.ndarray:
        if hasattr(self, "_val_weights"):
            return self._val_weights

        self._val_weights = self.load_data("val_weights")
        return self._val_weights

    @property
    def target(self) -> np.ndarray:
        if hasattr(self, "_target"):
            return self._target

        target = self.load_data("target")

        # handling of negative weights based on the ml_model_inst.negative_weights parameter
        if self._ml_model_inst.negative_weights == "handle":
            target[self.m_negative_weights] = 1 - target[self.m_negative_weights]

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

        return self._prediction
