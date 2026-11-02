# coding: utf-8

"""
Checks that all registered inference models can be built for the config of their analysis.
"""

from __future__ import annotations

import law
import pytest

from columnflow.ml import MLModel
from columnflow.production import Producer
from columnflow.tasks.cms.inference import CreateDatacards

from hbw.config.defaults_and_groups import default_producers
from hbw.inference.base import HBWInferenceModelBase

from tests.analyses import isolated_analysis
from tests.helpers import models_per_module, build_taf, load_analysis
from tests.known_issues import BROKEN_CONFIGS, BROKEN_INFERENCE_MODELS, UNREGISTERED_ML_MODELS, maybe_xfail

pytestmark = pytest.mark.config

# modules defining inference models, mapped to the (analysis, config) used to build them
INFERENCE_MODULES = {
    "hbw.inference.inf_dl_dih": ("hbw_dl", "c24v15"),
    "hbw.inference.inf_dl_trih": ("hbw_hhh_dl", "c24v15"),
    "hbw.inference.sl": ("hbw_sl", "c24v15"),
}

INFERENCE_MODELS = models_per_module(HBWInferenceModelBase, INFERENCE_MODULES)

BROKEN = {
    **BROKEN_INFERENCE_MODELS,
    **{
        model: f"config broken: {BROKEN_CONFIGS[case]}"
        for model, case in INFERENCE_MODELS.items()
        if case in BROKEN_CONFIGS and model not in BROKEN_INFERENCE_MODELS
    },
}


def test_inference_models_found():
    assert "default_unblind" in INFERENCE_MODELS


def ml_model_names(model_name: str) -> list[str]:
    return law.util.make_list(HBWInferenceModelBase.get_cls(model_name).ml_model_name)


@pytest.mark.parametrize("model_name", [maybe_xfail(m, UNREGISTERED_ML_MODELS) for m in INFERENCE_MODELS])
def test_ml_models_registered(model_name):
    missing = [name for name in ml_model_names(model_name) if not MLModel.has_cls(name)]
    assert not missing, f"ml_model_name refers to unregistered MLModels: {missing}"


@pytest.mark.parametrize("model_name", [maybe_xfail(m, BROKEN) for m in INFERENCE_MODELS])
def test_inference_model_builds(get_config, model_name):
    inference_model_cls = HBWInferenceModelBase.get_cls(model_name)
    analysis_name, config_name = INFERENCE_MODELS[model_name]
    names = ml_model_names(model_name)
    # inference models with the same ML models share configs (others might clash, see tests/analyses.py)
    analysis_name = isolated_analysis(analysis_name, ",".join(names))
    config_inst = get_config(analysis_name, config_name)

    # set up the ML models for this config (adding their processes and score variables)
    for name in names:
        MLModel.get_cls(name)(load_analysis(analysis_name))._setup([config_name])

    # build the producers that CreateDatacards resolves by default, which add the categories and
    # input variables to the config
    producers = default_producers(CreateDatacards, config_inst, {"inference_model": model_name})
    unknown = [name for name in producers if not Producer.has_cls(name)]
    assert not unknown, f"default producers do not exist: {unknown}"
    for name in producers:
        build_taf(Producer.get_cls(name), config_inst, config_inst.x.default_dataset)

    inference_model_inst = inference_model_cls([config_inst])

    assert inference_model_inst.categories, "inference model has no categories"
    for cat in inference_model_inst.categories:
        assert cat.processes, f"category {cat.name} has no processes"
        config_data = cat.config_data[config_inst.name]
        assert config_inst.has_category(config_data.category)
        assert config_inst.has_variable(config_data.variable)
