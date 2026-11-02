# coding: utf-8

"""
Checks that all registered ML models can be set up for the configs they are trained on.
"""

from __future__ import annotations

import pytest

from columnflow.ml import MLModel
from columnflow.production import Producer
from columnflow.tasks.framework.mixins import MLModelMixinBase

from hbw.ml.base import MLClassifierBase

from tests.analyses import isolated_analysis
from tests.helpers import models_per_module, load_analysis
from tests.known_issues import BROKEN_CONFIGS, BROKEN_ML_MODELS, maybe_xfail

pytestmark = pytest.mark.config

# modules defining ML models, mapped to the (analysis, config) used to request them
ML_MODULES = {
    "hbw.ml.derived.ml_dl_dih": ("hbw_dl", "c24v15"),
    "hbw.ml.derived.ml_dl_trih": ("hbw_hhh_dl", "c24v15"),
    "hbw.ml.derived.sl": ("hbw_sl", "c24v15"),
}

# configs that are maintained; models trained on other configs are skipped
SUPPORTED_CONFIGS = {"c24v15", "l24v15", "c25v15", "l25v15"}

ML_MODELS = models_per_module(MLClassifierBase, ML_MODULES, exclude={"DenseClassifierDL", "DenseClassifierSL"})

BROKEN = {
    **BROKEN_ML_MODELS,
    **{
        model: f"config broken: {BROKEN_CONFIGS[case]}"
        for model, case in ML_MODELS.items()
        if case in BROKEN_CONFIGS and model not in BROKEN_ML_MODELS
    },
}


@pytest.fixture
def ml_model_inst(request, get_config):
    model_name = request.param
    analysis_name, config_name = ML_MODELS[model_name]
    # each model gets its own configs, since models with the same train nodes cannot share them
    analysis_name = isolated_analysis(analysis_name, model_name)
    get_config(analysis_name, config_name)
    analysis_inst = load_analysis(analysis_name)

    # resolve the training configs the same way as tasks do
    training_configs = MLModel.get_cls(model_name)(analysis_inst).training_configs([config_name])
    unsupported = set(training_configs) - SUPPORTED_CONFIGS
    if unsupported:
        pytest.skip(f"trains on unmaintained configs {sorted(unsupported)}")

    for name in training_configs:
        get_config(analysis_name, name)
    return MLModelMixinBase.get_ml_model_inst(model_name, analysis_inst, requested_configs=[config_name])


def test_ml_models_found():
    assert "multiclass_2425_v32l2pt15" in ML_MODELS


@pytest.mark.parametrize("ml_model_inst", [maybe_xfail(m, BROKEN) for m in ML_MODELS], indirect=True)
def test_ml_model_setup(ml_model_inst):
    assert ml_model_inst.config_insts, "ML model has no training configs"

    for config_inst in ml_model_inst.config_insts:
        # processes and train nodes
        for proc in ml_model_inst.processes:
            assert config_inst.has_process(proc), f"unknown process {proc} in {config_inst.name}"
        for node in ml_model_inst.train_nodes:
            assert config_inst.has_variable(f"mlscore.{node}"), f"missing variable mlscore.{node}"

        # datasets and input columns
        datasets = ml_model_inst.datasets(config_inst)
        assert datasets, f"no training datasets in {config_inst.name}"
        assert all(config_inst.has_dataset(d) for d in datasets)
        assert ml_model_inst.uses(config_inst), "no input columns"

    # the categorizer producer used for its evaluation must exist
    assert Producer.has_cls(f"cats_ml_{ml_model_inst.cls_name}")
