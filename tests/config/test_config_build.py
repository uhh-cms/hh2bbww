# coding: utf-8

"""
Checks that the analysis configs can be built and that their defaults point to existing objects.
"""

from __future__ import annotations

import pytest

from columnflow.calibration import Calibrator
from columnflow.selection import Selector
from columnflow.reduction import Reducer
from columnflow.histogramming import HistProducer
from columnflow.inference import InferenceModel

from hbw.config.processes import unique_process_sanity_check

from tests.known_issues import BROKEN_CONFIGS, xfail_params

pytestmark = pytest.mark.config

# (analysis, config) pairs that are expected to build (broken ones are listed in known_issues.py)
CONFIGS = [
    ("hbw_dl", "c24v15"),
    ("hbw_dl", "l24v15"),
    ("hbw_dl", "c25v15"),
]

# config aux defaults that name registered TaskArrayFunctions or models
REGISTERED_DEFAULTS = {
    "default_calibrator": Calibrator,
    "default_selector": Selector,
    "default_reducer": Reducer,
    "default_hist_producer": HistProducer,
    "default_inference_model": InferenceModel,
}


def config_id(case):
    return "-".join(case)


@pytest.fixture(params=CONFIGS, ids=config_id)
def config_inst(request, get_config):
    return get_config(*request.param)


@pytest.mark.parametrize("case", [
    *CONFIGS,
    *xfail_params(BROKEN_CONFIGS),
], ids=config_id)
def test_config_builds(get_config, case):
    config_inst = get_config(*case)
    assert config_inst.name == case[1]
    assert config_inst.has_tag("is_sl") or config_inst.has_tag("is_dl")


@pytest.mark.parametrize("aux_key", REGISTERED_DEFAULTS)
def test_default_is_registered(config_inst, aux_key):
    value = config_inst.x(aux_key)
    if callable(value):
        pytest.skip(f"{aux_key} is resolved per task, covered by tests/tasks")

    base_cls = REGISTERED_DEFAULTS[aux_key]
    names = [value] if isinstance(value, str) else list(value)
    missing = [name for name in names if not base_cls.has_cls(name)]
    assert not missing, f"{aux_key} of {config_inst.name} refers to unknown {base_cls.__name__}s: {missing}"


def test_default_categories_and_variables(config_inst):
    categories = {cat.name for cat, _, _ in config_inst.walk_categories()}
    categories |= set(config_inst.x("category_groups", {}))
    missing_cats = set(config_inst.x.default_categories) - categories
    assert not missing_cats, f"unknown default_categories: {missing_cats}"

    variables = set(config_inst.variables.names()) | set(config_inst.x("variable_groups", {}))
    missing_vars = set(config_inst.x.default_variables) - variables
    assert not missing_vars, f"unknown default_variables: {missing_vars}"


def test_default_dataset(config_inst):
    default_dataset = config_inst.x.default_dataset
    assert config_inst.has_dataset(default_dataset), f"unknown default_dataset {default_dataset}"


def test_datasets_have_known_processes(config_inst):
    no_process = [dataset.name for dataset in config_inst.datasets if not dataset.processes]
    assert not no_process, f"datasets without processes: {no_process}"

    unknown = {
        dataset.name: process.name
        for dataset in config_inst.datasets
        for process in dataset.processes
        if not config_inst.has_process(process.name)
    }
    assert not unknown, f"datasets with processes missing in the config: {unknown}"


def test_dataset_groups_resolve(config_inst):
    datasets = set(config_inst.datasets.names())
    empty = [
        group for group, members in config_inst.x("dataset_groups", {}).items()
        if not any(m in datasets or "*" in m for m in members)
    ]
    assert not empty, f"dataset groups without any existing dataset: {empty}"


def test_unique_ids(config_inst):
    unique_process_sanity_check(config_inst)

    seen = {}
    duplicates = set()
    for cat, _, _ in config_inst.walk_categories():
        if seen.setdefault(cat.id, cat.name) != cat.name:
            duplicates.add((cat.id, seen[cat.id], cat.name))
    assert not duplicates, f"categories with duplicate ids: {duplicates}"


def test_shifts(config_inst):
    assert config_inst.has_shift("nominal")
    # every up shift needs a down shift and vice versa
    sources = {shift.source for shift in config_inst.shifts if not shift.is_nominal}
    incomplete = [
        source for source in sources
        if not (config_inst.has_shift(f"{source}_up") and config_inst.has_shift(f"{source}_down"))
    ]
    assert not incomplete, f"shift sources without up/down variations: {incomplete}"
