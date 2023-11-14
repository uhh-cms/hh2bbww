# coding: utf-8

from hbw.tasks.ml import MLOptimizer
from hbw.ml.derived import grid_search_models

# TODO: this only stats the training for one workflow (so 1 MLTraining) at the time
#       would be nice, if we could cram the N MLTraining workflows into one workflow
task = MLOptimizer(
    version="prod1",
    ml_models=grid_search_models,
)
task.law_run()
