# coding: utf-8

"""
Central list of known problems found by the test suite.

Tests for these cases are marked as strict xfails: they do not break the suite while the problem
persists, but they fail (XPASS) as soon as it is fixed, so that the entry here can be removed.
Entries map the affected object to a short reason.
"""

import pytest


def xfail_params(issues: dict) -> list:
    """
    Converts a dict of known *issues* into a list of strict xfail parameters for ``parametrize``.
    """
    return [pytest.param(case, marks=pytest.mark.xfail(reason=reason, strict=True)) for case, reason in issues.items()]


def maybe_xfail(case, *issues: dict):
    """
    Returns *case* as a strict xfail parameter when it is listed in any of the *issues* dicts.
    """
    for _issues in issues:
        if case in _issues:
            return pytest.param(case, marks=pytest.mark.xfail(reason=_issues[case], strict=True))
    return case


# law config files with module lists outside of [analysis] (ignored by columnflow)
MISPLACED_LAW_CFG_MODULES = {
    "law.dl.cfg": "production/calibration/selection_modules listed in [outputs] instead of [analysis]",
    "law.balduin.cfg": "*_modules listed in [outputs] instead of [analysis]",
}

# (analysis, config) pairs whose config cannot be built
BROKEN_CONFIGS = {
    ("hbw_hhh_dl", "c24v15"): "defaults_and_groups_hhh.py: process group 'dl1' does not exist (KeyError)",
    ("hbw_sl", "c24v15"): "analysis has neither the 'is_hh' nor the 'is_hhh' tag required by add_config",
    ("hbw_sl_res", "c24v15"): "analysis has neither the 'is_hh' nor the 'is_hhh' tag required by add_config",
    ("hbw_merged", "c24v15"): "analysis has neither the 'is_hh' nor the 'is_hhh' tag required by add_config",
}

# ML models that cannot be set up, besides those of broken configs
BROKEN_ML_MODELS = {
    "dctr_dy_24_v6": "negative_weights 'nothing' is not one of ('ignore', 'abs', 'handle')",
}

# inference models whose ml_model_name refers to MLModels that are not registered (e.g. commented out)
_unregistered = "ml_model_name refers to MLModels that are not registered: {}"
UNREGISTERED_ML_MODELS = {
    "dl_kl1_dnn": _unregistered.format("multiclass_kl1, ggf_kl1, vbf_kl1"),
    "full24": _unregistered.format("multiclass24, ggf24, vbf24"),
    "rate24": _unregistered.format("multiclass24, ggf24, vbf24"),
    "full24_closure": _unregistered.format("multiclass24_closure, ggf24_closure, vbf24_closure"),
    "rate24_closure": _unregistered.format("multiclass24_closure, ggf24_closure, vbf24_closure"),
    "rate24_v3_lep2pt15": _unregistered.format("multiclass24_v3l2pt15, ggf24_v3l2pt15, vbf24_v3l2pt15"),
    "dl_trih": _unregistered.format("hhh_v0 models"),
    "Cat_eq2b_Bin_V5_shape": _unregistered.format("Bin_V1 models"),
    "Cat_eq2b_Bin_V5_shape_unblind": _unregistered.format("Bin_V1 models"),
    "Cat_eq3b_Bin_V5_shape": _unregistered.format("Bin_V1 models"),
    "Cat_eq3b_Bin_V5_shape_unblind": _unregistered.format("Bin_V1 models"),
    "Cat_geq4b_Bin_V5_shape": _unregistered.format("Bin_V1 models"),
    "Cat_geq4b_Bin_V5_shape_unblind": _unregistered.format("Bin_V1 models"),
}

# inference models that cannot be built, besides those of broken configs and unregistered ML models
_no_ml = (
    "ml_model_name [] makes default_producers (hbw/config/defaults_and_groups.py) resolve the "
    "non-existent producer 'cats_ml_[]', since [] is not caught by the 'not in (..., tuple())' check"
)
BROKEN_INFERENCE_MODELS = {
    **UNREGISTERED_ML_MODELS,
    "bjet_incl": "config_categories.bjet_incl contains 'sr__ml_dy', but its ML models have no 'dy' node",
    "ll_pt_dycr": _no_ml,
    "ll_pt_dycr_before": _no_ml,
    "n_jet_dycr": _no_ml,
    "n_jet_dycr_before": _no_ml,
}

# tasks that do not resolve with the analysis defaults, mapped by test id
BROKEN_TASK_DEFAULTS = {
    "CreateDatacards-default": (
        "default_inference_model 'default_unblind' uses ML models trained on the unmaintained 2022/23 configs, "
        "so resolving it requires their campaigns"
    ),
}
