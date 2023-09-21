# coding: utf-8


from hbw.ml.dense_classifier import DenseClassifier, dense_test  # noqa
from hbw.tasks.ml import MLModelOptimizer


# the derived models from here can only be used when started from this script
for i in range(10):
    dense_test_copy = dense_test.derive(f"v{i}")

# TODO: define a useful set of MLModels for a grid search

# TODO: this only stats the training for one workflow (so 1 MLTraining) at the time
#       would be nice, if we could cram the N MLTraining workflows into one workflow
task = MLModelOptimizer(
    version="prod1",
    ml_folds=0,
    ml_models=["v1", "v2"],
)
task.law_run()
