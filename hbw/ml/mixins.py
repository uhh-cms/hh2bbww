# coding: utf-8

"""
Mixin classes to build ML models
"""

import law
# import order as od

from columnflow.util import maybe_import, DotDict

np = maybe_import("numpy")
ak = maybe_import("awkward")
tf = maybe_import("tensorflow")
keras = maybe_import("tensorflow.keras")

logger = law.logger.get_logger(__name__)


class DenseModelMixin():
    """
    Mixin that provides an implementation for `prepare_ml_model`
    """

    activation = "relu"
    layers = (64, 64, 64)
    dropout = 0.50
    learningrate = 2 ** 14

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def prepare_ml_model(
        self,
        task: law.Task,
    ):
        from keras.models import Sequential
        from keras.layers import Dense, BatchNormalization

        n_inputs = len(self.input_features)
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
            loss="categorical_crossentropy",
            optimizer=optimizer,
            weighted_metrics=["categorical_accuracy"],
        )

        return model


class ModelFitMixin():

    callbacks = [
        "backup", "checkpoint", "early_stopping", "reduce_lr",
    ]
    remove_backup = False
    epochs = 200
    batchsize = 2 ** 14

    def __init__(
            self,
            *args,
            **kwargs,
    ):

        super().__init__(*args, **kwargs)

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
        with tf.device("CPU"):
            tf_train = tf.data.Dataset.from_tensor_slices(
                (train.inputs, train.target, train.weights),
            ).shuffle(buffer_size=5 * len(train.inputs)).batch(self.batchsize)
            tf_validation = tf.data.Dataset.from_tensor_slices(
                (validation.inputs, validation.target, validation.weights),
            ).shuffle(buffer_size=5 * len(validation.inputs)).batch(self.batchsize)

        # output used for BackupAndRestore callback (not deleted by --remove-output)
        # NOTE: does that work when running remote?
        backup_output=output.parent.child(f"backup_{output.basename}", type="d")
        if self.remove_backup:
            backup_output.remove()

        callback_options = {
            "backup": tf.keras.callbacks.BackupAndRestore(
                backup_dir=backup_output.path,
            ),
            "checkpoint": tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{output.path}/checkpoint",
                save_weights_only=False,
                monitor="val_loss",
                mode="auto",
                save_best_only=True,
            ),
            "early_stopping": tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=min(50, int(self.epochs / 4)),
                verbose=1,
                restore_best_weights=True,
                start_from_epoch=min(50, int(self.epochs / 4)),
            ),
            "reduce_lr": tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=min(10, int(self.epochs / 20)),
                verbose=1,
                mode="auto",
                min_delta=0,
                min_lr=0,
            )
        }
        # allow the user to choose which callbacks to use
        callbacks = [callback_options[key] for key in self.callbacks]

        logger.info("Starting training...")
        model.fit(
            tf_train,
            validation_data=tf_validation,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=2,
        )
