# coding: utf-8

"""
Mixin classes to build ML models
"""

from __future__ import annotations

import law
# import order as od

from columnflow.types import Union
from columnflow.util import maybe_import, DotDict

from hbw.util import log_memory, call_func_safe

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


def loop_dataset(data, max_count=10000):
    for i, x in enumerate(data):
        if i % int(max_count / 100) == 0:
            print(i)
        if i == max_count:
            break


class DenseModelMixin(object):
    """
    Mixin that provides an implementation for `prepare_ml_model`
    """

    activation: str = "relu"
    layers: tuple[int] = (64, 64, 64)
    dropout: float = 0.50
    learningrate: float = 0.00050

    def cast_ml_param_values(self):
        """
        Cast the values of the parameters to the correct types
        """
        super().cast_ml_param_values()
        self.activation = str(self.activation)
        self.layers = tuple(int(n_nodes) for n_nodes in self.layers)
        self.dropout = float(self.dropout)
        self.learningrate = float(self.learningrate)

    def prepare_ml_model(
        self,
        task: law.Task,
    ):
        import tensorflow.keras as keras
        from keras.models import Sequential
        from keras.layers import Dense, BatchNormalization
        from hbw.ml.tf_util import cumulated_crossentropy

        n_inputs = len(set(self.input_features))
        n_outputs = len(self.processes)

        # define the DNN model
        model = Sequential()

        # BatchNormalization layer with input shape
        model.add(BatchNormalization(input_shape=(n_inputs,)))

        activation_settings = DotDict({
            "elu": ("ELU", "he_uniform", "Dropout"),
            "relu": ("ReLU", "he_uniform", "Dropout"),
            "prelu": ("PReLU", "he_normal", "Dropout"),
            "selu": ("selu", "lecun_normal", "AlphaDropout"),
            "tanh": ("tanh", "glorot_normal", "Dropout"),
            "softmax": ("softmax", "glorot_normal", "Dropout"),
        })
        keras_act_name, init_name, dropout_layer = activation_settings[self.activation]

        # hidden layers
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
            loss=cumulated_crossentropy,
            # NOTE: we'd preferrably use the Keras CCE, but it does not work when assigning one event
            #       to multiple classes (target with multiple entries != 0)
            # loss ="categorical_crossentropy",
            # loss=tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE),
            optimizer=optimizer,
            metrics=["categorical_accuracy"],
            weighted_metrics=["categorical_accuracy"],
        )

        return model


class CallbacksBase(object):
    """ Base class that handles parametrization of callbacks """
    callbacks: set[str] = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True

    # NOTE: we could remove these parameters since they can be implemented via reduce_lr_kwargs
    reduce_lr_factor: float = 0.8
    reduce_lr_patience: int = 3

    # custom callback kwargs
    checkpoint_kwargs: dict = {}
    backup_kwargs: dict = {}
    early_stopping_kwargs: dict = {}
    reduce_lr_kwargs: dict = {}

    def cast_ml_param_values(self):
        """
        Cast the values of the parameters to the correct types
        """
        super().cast_ml_param_values()
        self.callbacks = set(self.callbacks)
        self.remove_backup = bool(self.remove_backup)
        self.reduce_lr_factor = float(self.reduce_lr_factor)
        self.reduce_lr_patience = int(self.reduce_lr_patience)

    def get_callbacks(self, output):
        import tensorflow.keras as keras
        # check that only valid options have been requested
        callback_options = {"backup", "checkpoint", "reduce_lr", "early_stopping"}
        if diff := self.callbacks.difference(callback_options):
            logger.warning(f"Callbacks '{diff}' have been requested but are not properly implemented")

        # list of callbacks to be returned at the end
        callbacks = []

        # output used for BackupAndRestore callback (not deleted by --remove-output)
        # NOTE: does that work when running remote?
        # TODO: we should also save the parameters + input_features in the backup to ensure that they
        #       are equivalent (delete backup if not)
        backup_output = output["mlmodel"].sibling(f"backup_{output['mlmodel'].basename}", type="d")
        if self.remove_backup:
            backup_output.remove()

        #
        # for each requested callback, merge default kwargs with custom callback kwargs
        #

        if "backup" in self.callbacks:
            backup_kwargs = dict(
                backup_dir=backup_output.path,
            )
            backup_kwargs.update(self.backup_kwargs)
            callbacks.append(keras.callbacks.BackupAndRestore(**backup_kwargs))

        if "checkpoint" in self.callbacks:
            checkpoint_kwargs = dict(
                filepath=output["checkpoint"].path,
                save_weights_only=False,
                monitor="val_loss",
                mode="auto",
                save_best_only=True,
            )
            checkpoint_kwargs.update(self.checkpoint_kwargs)
            callbacks.append(keras.callbacks.ModelCheckpoint(**checkpoint_kwargs))

        if "early_stopping" in self.callbacks:
            early_stopping_kwargs = dict(
                monitor="val_loss",
                min_delta=0,
                patience=max(min(50, int(self.epochs / 5)), 10),
                verbose=1,
                restore_best_weights=True,
                start_from_epoch=max(min(50, int(self.epochs / 5)), 10),
            )
            early_stopping_kwargs.update(self.early_stopping_kwargs)
            callbacks.append(keras.callbacks.EarlyStopping(**early_stopping_kwargs))

        if "reduce_lr" in self.callbacks:
            reduce_lr_kwargs = dict(
                monitor="val_loss",
                factor=self.reduce_lr_factor,
                patience=self.reduce_lr_patience,
                verbose=1,
                mode="auto",
                min_delta=0,
                min_lr=0,
            )
            reduce_lr_kwargs.update(self.reduce_lr_kwargs)
            callbacks.append(keras.callbacks.ReduceLROnPlateau(**reduce_lr_kwargs))

        if len(callbacks) != len(self.callbacks):
            logger.warning(
                f"{len(self.callbacks)} callbacks have been requested but only {len(callbacks)} are returned",
            )

        return callbacks


class ClassicModelFitMixin(CallbacksBase):
    """
    Mixin to run ML Training with "classic" training loop.
    TODO: this will require a different reweighting
    """

    callbacks: set = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True
    reduce_lr_factor: float = 0.8
    reduce_lr_patience: int = 3
    epochs: int = 200
    batchsize: int = 2 ** 12

    def cast_ml_param_values(self):
        """
        Cast the values of the parameters to the correct types
        """
        super().cast_ml_param_values()
        self.epochs = int(self.epochs)
        self.batchsize = int(self.batchsize)

    def fit_ml_model(
        self,
        task: law.Task,
        model,
        train: DotDict[np.array],
        validation: DotDict[np.array],
        output,
    ) -> None:
        """
        Training loop with normal tf dataset
        """
        import tensorflow as tf

        log_memory("start")

        with tf.device("CPU"):
            tf_train = tf.data.Dataset.from_tensor_slices(
                (train["inputs"], train["target"], train["weights"]),
            ).batch(self.batchsize)
            tf_validation = tf.data.Dataset.from_tensor_slices(
                (validation["inputs"], validation["target"], validation["weights"]),
            ).batch(self.batchsize)

        log_memory("init")

        # set the kwargs used for training
        model_fit_kwargs = {
            "validation_data": tf_validation,
            "epochs": self.epochs,
            "verbose": 2,
            "callbacks": self.get_callbacks(output),
        }

        logger.info("Starting training...")
        model.fit(
            tf_train,
            **model_fit_kwargs,
        )
        log_memory("loop")

        # delete tf datasets to clear memory
        del tf_train
        del tf_validation
        log_memory("del")


class ModelFitMixin(CallbacksBase):
    # parameters related to callbacks
    callbacks: set = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True
    reduce_lr_factor: float = 0.8
    reduce_lr_patience: int = 3

    epochs: int = 200
    batchsize: int = 2 ** 12
    # either set steps directly or use attribute from the MultiDataset
    steps_per_epoch: Union[int, str] = "iter_smallest_process"

    def cast_ml_param_values(self):
        """
        Cast the values of the parameters to the correct types
        """
        super().cast_ml_param_values()
        self.epochs = int(self.epochs)
        self.batchsize = int(self.batchsize)
        if isinstance(self.steps_per_epoch, float):
            self.steps_per_epoch = int(self.steps_per_epoch)
        else:
            self.steps_per_epoch = str(self.steps_per_epoch)

    def fit_ml_model(
        self,
        task: law.Task,
        model,
        train: DotDict[np.array],
        validation: DotDict[np.array],
        output,
    ) -> None:
        """
        Training loop but with custom dataset
        """
        import tensorflow as tf
        from hbw.ml.tf_util import MultiDataset
        from hbw.ml.plotting import plot_history

        log_memory("start")

        with tf.device("CPU"):
            tf_train = MultiDataset(data=train, batch_size=self.batchsize, kind="train", buffersize=0)
            tf_validation = tf.data.Dataset.from_tensor_slices(
                (validation.features, validation.target, validation.train_weights),
            ).batch(self.batchsize)

        log_memory("init")

        # determine the requested steps_per_epoch
        if isinstance(self.steps_per_epoch, str):
            steps_per_epoch = getattr(tf_train, self.steps_per_epoch)
        else:
            steps_per_epoch = int(self.steps_per_epoch)
        if not isinstance(steps_per_epoch, int):
            raise Exception(
                f"steps_per_epoch is {self.steps_per_epoch} but has to be either an integer or"
                "a string corresponding to an integer attribute of the MultiDataset",
            )
        logger.info(f"Training will be done with {steps_per_epoch} steps per epoch")

        # set the kwargs used for training
        model_fit_kwargs = {
            "validation_data": tf_validation,
            "epochs": self.epochs,
            "verbose": 2,
            "steps_per_epoch": steps_per_epoch,
            "callbacks": self.get_callbacks(output),
        }

        # start training by iterating over the MultiDataset
        iterator = (x for x in tf_train)
        logger.info("Starting training...")
        model.fit(
            iterator,
            **model_fit_kwargs,
        )

        # create history plots
        for metric, ylabel in (
            ("loss", "Loss"),
            ("categorical_accuracy", "Accuracy"),
            ("weighted_categorical_accuracy", "Weighted Accuracy"),
        ):
            call_func_safe(plot_history, model.history.history, output["plots"], metric, ylabel)
