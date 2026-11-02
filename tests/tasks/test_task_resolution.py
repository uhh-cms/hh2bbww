# coding: utf-8

"""
Checks that the main law tasks resolve their parameters and requirements with the analysis defaults
(see hbw/config/defaults_and_groups*.py) without running anything.
"""

from __future__ import annotations

import law
import pytest

from columnflow.tasks.selection import SelectEvents
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.ml import MLTraining
from columnflow.tasks.plotting import PlotVariables1D
from columnflow.tasks.cms.inference import CreateDatacards

from tests.known_issues import BROKEN_TASK_DEFAULTS

pytestmark = pytest.mark.config

ANALYSIS = "hbw.analysis.hbw_dl.hbw_dl"
# tasks using the ML categorization need a separate config instance, see tests/analyses.py
ML_ANALYSIS = "tests.analyses.hbw_dl_ml"
CONFIG = "c24v15"
VERSION = "test_task_resolution"


@pytest.fixture(autouse=True)
def cmdline_values(monkeypatch):
    # tasks look up command line values when requiring other tasks, which are not set outside of "law run"
    monkeypatch.setattr(law.parser, "_global_cmdline_args", {})
    monkeypatch.setattr(law.parser, "_global_cmdline_values", {})


@pytest.fixture(scope="module")
def config_inst(get_config):
    # make sure that the config (and its campaign summary) exists before tasks are created
    return get_config("hbw_dl", CONFIG)


@pytest.fixture(scope="module")
def ml_config_inst(get_config):
    return get_config(ML_ANALYSIS, CONFIG)


def task_params(analysis: str = ANALYSIS, **kwargs) -> dict:
    return {"analysis": analysis, "version": VERSION, **kwargs}


def walk_families(task: law.Task, max_tasks: int = 100) -> dict[str, law.Task]:
    """
    Walks the requirements of *task* and returns the first task found per task family. Only one
    task per family is expanded further to keep the number of created tasks small.
    """
    families = {}
    queue = [task]
    while queue and len(families) < max_tasks:
        current = queue.pop(0)
        if current.task_family in families:
            continue
        families[current.task_family] = current
        queue.extend(
            dep for dep in law.util.flatten(current.requires())
            if isinstance(dep, law.Task) and dep.task_family not in families
        )
    return families


def test_select_events(config_inst):
    task = SelectEvents(**task_params(config=CONFIG, dataset=config_inst.x.default_dataset))
    assert task.selector == config_inst.x.default_selector
    assert tuple(task.calibrators) == tuple(config_inst.x.default_calibrator)

    families = walk_families(task)
    assert {"cf.CalibrateEvents", "cf.GetDatasetLFNs"} <= set(families)


@pytest.mark.parametrize("config_name", [CONFIG, CONFIG.replace("c", "l", 1)])
def test_hbw_store_parts(get_config, config_name):
    config_inst = get_config("hbw_dl", config_name)
    task = SelectEvents(**task_params(config=config_name, dataset=config_inst.x.default_dataset))

    # the modifier is only applied for the outputs that request it in the law config, so call it here
    hbw_parts = task.analysis_inst.x.store_parts_modifiers["hbw_parts"]
    parts = hbw_parts(task, task.store_parts())

    # selection outputs are shared between sub-analyses and between limited and full configs
    assert parts["analysis"] == "hbw_merged"
    assert parts["config"] == CONFIG
    # the version is encoded in the selector part instead
    assert "version" not in parts
    assert list(parts).index("selector") < list(parts).index("task_family")


def test_produce_columns_defaults(config_inst):
    task = ProduceColumns(**task_params(config=CONFIG, dataset=config_inst.x.default_dataset))
    # ProduceColumns has a single producer, the default is taken from the first entry
    assert task.producer


def test_ml_training_defaults(config_inst):
    task = MLTraining(**task_params(configs=(CONFIG,), ml_model="multiclass_2425_v32l2pt15"))
    assert task.ml_model == "multiclass_2425_v32l2pt15"
    # the ML pipeline does not use the ML categorizer, see default_producers
    assert not any(p.startswith("cats_ml") for p in law.util.flatten(task.producers))

    families = walk_families(task)
    assert "cf.PrepareMLEvents" in families


def test_plot_variables_defaults(config_inst):
    task = PlotVariables1D(**task_params(
        configs=(CONFIG,),
        processes=(("tt", "dy"),),
        variables=("n_jet",),
        categories=("incl",),
    ))
    assert task.datasets and all(task.datasets), "no datasets resolved from processes"
    assert task.shift == "nominal"

    producers = law.util.flatten(task.producers)
    assert config_inst.x.ml_inputs_producer in producers
    # with a default ML model, the ML categorizer replaces the pre-ML one
    assert any(p.startswith("cats_ml_") for p in producers) != ("pre_ml_cats" in producers)


def test_plot_variables_shift_resolution(config_inst):
    task = PlotVariables1D(**task_params(
        configs=(CONFIG,),
        processes=(("tt",),),
        variables=("n_jet",),
        categories=("incl",),
        shift="nominal",
    ))
    assert task.local_shift == "nominal"


DEFAULT_DATACARDS_XFAIL = pytest.mark.xfail(reason=BROKEN_TASK_DEFAULTS["CreateDatacards-default"], strict=True)


@pytest.mark.parametrize("inference_model", [
    pytest.param("default", marks=DEFAULT_DATACARDS_XFAIL),
    "rate_boostedbkg32",
], ids=lambda name: f"CreateDatacards-{name}")
def test_create_datacards(ml_config_inst, inference_model):
    params = task_params(analysis=ML_ANALYSIS, configs=(CONFIG,))
    if inference_model != "default":
        params["inference_model"] = inference_model
    task = CreateDatacards(**params)

    inference_model = params.get("inference_model", ml_config_inst.x.default_inference_model)
    assert task.inference_model == inference_model

    # the ML models and the ML categorizer (of the first model) are derived from the inference model
    ml_models = law.util.make_list(task.inference_model_inst.ml_model_name)
    assert tuple(law.util.flatten(task.ml_models)) == tuple(ml_models)
    assert f"cats_ml_{ml_models[0]}" in law.util.flatten(task.producers)

    families = walk_families(task)
    expected = {
        "cf.MergeShiftedHistograms", "cf.MergeHistograms", "cf.CreateHistograms", "cf.ProduceColumns",
        "cf.MLEvaluation", "cf.MLTraining",
    }
    assert expected <= set(families), f"missing upstream tasks: {expected - set(families)}"
