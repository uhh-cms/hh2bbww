"""
Collection of classes to load and prepare data for machine learning.
"""

from __future__ import annotations

from columnflow.util import maybe_import
from columnflow.columnar_util import remove_ak_column
from columnflow.ml import MLModel


ak = maybe_import("awkward")
np = maybe_import("numpy")


class MLDatasetLoader:
    """
    Helper class to conveniently load ML training data from an awkward array.
    """

    def __init__(self, ml_model_inst: MLModel, process: "str", events: ak.Array, stats: dict):
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

        # prepare the weights
        self._weights = ak.to_numpy(events.normalization_weight).astype(np.float32)

        # bookkeep order of input features
        self._input_features = tuple(events.fields)
        if getattr(self.ml_model_inst, "input_features_ordered", None):
            # check if input features are ordered in the same way for all datasets
            if self.ml_model_inst.input_features_ordered != self._input_features:
                raise Exception(
                    f"Input features are not ordered in the same way for all datasets. "
                    f"Expected: {self.ml_model_inst.input_features_ordered}, "
                    f"got: {self._input_features}",
                )
        else:
            # if not already set, bookkeep input features in the ml_model_inst aswell
            self.ml_model_inst.input_features_ordered = self._input_features

        # remove columns that are not used as training features
        for var in events.fields:
            if var not in self.input_features:
                events = remove_ak_column(events, var)

        # transform events into numpy npdarray
        events = ak.to_numpy(events)
        events = events.astype(
            [(name, np.float32) for name in events.dtype.names], copy=False,
        ).view(np.float32).reshape((-1, len(events.dtype)))
        self._features = events

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
        return self._input_features

    @property
    def stats(self) -> dict:
        return self._stats

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def features(self) -> np.ndarray:
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

        self._shuffle_indices = np.random.permutation(len(self.features))
        return self._shuffle_indices

    @property
    def train_weights(self) -> np.ndarray:
        if hasattr(self, "_train_weights"):
            return self._train_weights

        sum_abs_weights = self.stats[self.process]["sum_abs_weights"]
        num_events = self.stats[self.process]["num_events"]

        self._train_weights = self.weights / sum_abs_weights * num_events
        return self._train_weights

    @property
    def val_weights(self) -> np.ndarray:
        processes = self.ml_model_inst.processes
        # sum_abs_weights = self.stats[proc_inst.name]["sum_abs_weights"]
        num_events = self.stats[self.process]["num_events"]
        num_events_per_process = {proc: self.stats[proc]["num_events"] for proc in processes}
        ml_process_weights = self.ml_model_inst.ml_process_weights

        # reweight validation events to match the number of events used in the training multi_dataset
        weights_scaler = (
            min([num_events_per_process[proc] / ml_process_weights[proc] for proc in processes]) *
            sum([ml_process_weights[proc] for proc in processes])
        )
        self._validation_weights = (
            self.train_weights * weights_scaler / num_events * ml_process_weights[self.process]
        )
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
