# coding: utf-8

"""
ML models using the MLClassifierBase and Mixins
"""

from __future__ import annotations

import law

from columnflow.util import maybe_import, DotDict

# from hbw.ml.base import MLClassifierBase
# from hbw.ml.mixins import DenseModelMixin, ModelFitMixin
# from hbw.config.styling import color_palette
from hbw.ml.derived.dl import DenseClassifierDL


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)

#
# configs
#

hhh_train_procs = [
    "hhh_4b2w_2l2nu_c30_d40",
    "hhh_4b2w_2l2nu_c30_d499",
    "hhh_4b2w_2l2nu_c319_d419",
    "hhh_4b2w_2l2nu_c31_d40",
    "hhh_4b2w_2l2nu_c31_d42",
    "hhh_4b2w_2l2nu_c32_d4m1",
    "hhh_4b2w_2l2nu_c34_d49",
    "hhh_4b2w_2l2nu_c3m1_d40",
    "hhh_4b2w_2l2nu_c3m1_d4m1",
    "hhh_4b2w_2l2nu_c3m1p5_d4m0p5",
]

processes = DotDict({
    "backgrounds_v0": [
        "st", "dy", "h", "tt_custom", "ttbb_custom", # "tthh"  # "tt_bb_custom", "tt2b_custom", "ttb_custom" # "ttbb_custom",  # "tthh_4b",   # "vhh_4b",  # noqa E501
    ],
    "hhh": [
        "hhh_4b2w_2l2nu_c30_d40",
        "hhh_4b2w_2l2nu_c30_d499",
        # "hhh_4b2w_2l2nu_c30_d4m1",
        "hhh_4b2w_2l2nu_c319_d419",
        "hhh_4b2w_2l2nu_c31_d40",
        "hhh_4b2w_2l2nu_c31_d42",
        "hhh_4b2w_2l2nu_c32_d4m1",
        "hhh_4b2w_2l2nu_c34_d49",
        "hhh_4b2w_2l2nu_c3m1_d40",
        "hhh_4b2w_2l2nu_c3m1_d4m1",
        "hhh_4b2w_2l2nu_c3m1p5_d4m0p5",
    ],
})
input_features = DotDict({
    "default": DenseClassifierDL.input_features,
    "expanded_hhh_inputs": [
        "mli_n_jet",
        "mli_n_btag",
        "mli_ht",
        "mli_met_pt",
        "mli_mll",
        "mli_ll_pt",
        "mli_mllMET",
        "mli_dr_ll",
        "mli_dphi_ll",
        "mli_deta_ll",

        "mli_lep_pt",
        "mli_lep_eta",
        "mli_lep2_pt",
        "mli_lep2_eta",

        "mli_dr_hbb",
        "mli_dphi_hbb",
        "mli_discrete_b_score_sum",
        "mli_min_dr_lb",
        "mli_min_dr_ll_bb",
        "mli_lb_pt_sum",

        "mli_hb_candidate_maxdr_jj",
        "mli_hb_candidate_mindr_jj",

        "mli_hb_candidate_mh1",
        "mli_hb_candidate_mh2",
        "mli_hb_candidate_h1_pt",
        "mli_hb_candidate_h2_pt",
        "mli_hb_candidate_dr_h1_h2",
        "mli_hb_candidate_dr_ll_h1",
        "mli_hb_candidate_dr_ll_h2",
        "mli_hb_candidate_mhhh",
        "mli_hb_candidate_dr_h1_llMET",
        "mli_hb_candidate_dr_h2_llMET",
        "mli_hb_candidate_dr_h1_nu",
        "mli_hb_candidate_dr_h2_nu",

        "mli_hb_candidate1_discrete_b_score",
        "mli_hb_candidate1_pt",
        "mli_hb_candidate1_eta",

        "mli_hb_candidate2_discrete_b_score",
        "mli_hb_candidate2_pt",
        "mli_hb_candidate2_eta",

        "mli_hb_candidate3_discrete_b_score",
        "mli_hb_candidate3_pt",
        "mli_hb_candidate3_eta",

        "mli_hb_candidate4_discrete_b_score",
        "mli_hb_candidate4_pt",
        "mli_hb_candidate4_eta",
    ],
    "gatja_scores": [
        f"gatja_output_{i}" for i in range(23)
    ],
})
input_features["gatja_inputs"] = input_features["expanded_hhh_inputs"] + input_features["gatja_scores"]

configs = DotDict({
    "22post": lambda self, requested_configs: ["c22postv14"],
    "23pre": lambda self, requested_configs: ["c23prev14"],
    "22": lambda self, requested_configs: ["c22prev14", "c22postv14"],
    "23": lambda self, requested_configs: ["c23prev14", "c23postv14"],
    "full": lambda self, requested_configs: ["c22prev14", "c22postv14", "c23prev14", "c23postv14"],
})

#
# derived MLModels for HHH
#

# ----------------------- BASELINE BINARY MODELS FOR HHH SIGNAL ------------------------------

Bin = DenseClassifierDL.derive("Bin", cls_dict={
    "input_features": input_features["gatja_inputs"],
    "processes": [
        *processes.hhh,
        *processes.backgrounds_v0_test_hhh,
    ],
    "train_nodes": {
        "sig_hhh_binary": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": processes.hhh,
        },
        "bkg_binary_for_hhh": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": processes.backgrounds_v0_test_hhh,
        },
    },
    # relative class factors between different nodes
    "class_factors": {
        "sig_hhh_binary": 1,
        "bkg_binary_for_hhh": 1,
    },
    # relative process weights within one class
    "sub_process_class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "ttbb_custom": 1,
        "tt_custom": 1,
        "st": 1,
        "dy": 1,
        "ttv": 1,
        "h": 1,
    },
    "epochs": 100,
    "preparation_producer_name": "prepml_geq3b",
})

# ----------------------- BASELINE BINARY MODELS FOR Boosted HHH SIGNAL ------------------------------

boosted_V1 = DenseClassifierDL.derive("boosted_V1", cls_dict={
    "input_features": input_features["expanded_hhh_inputs"],
    "processes": [
        *processes.hhh,
        *processes.backgrounds_v0,
    ],
    "train_nodes": {
        "sig_hhh_binary": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": processes.hhh,
        },
        "bkg_binary_for_hhh": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": processes.backgrounds_v0,
        },
    },
    # relative class factors between different nodes
    "class_factors": {
        "sig_hhh_binary": 1,
        "bkg_binary_for_hhh": 1,
    },
    # relative process weights within one class
    "sub_process_class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "ttbb_custom": 1,
        # "tt_bb_custom": 1,
        # "ttb_custom": 1,
        # "tt2b_custom": 1,
        # "tt_custom": 1,
        "st": 1,
        "dy": 1,
        "ttv": 1,
        "h": 1,
    },
    "epochs": 100,
    "preparation_producer_name": "prepml_boosted",
})

# ----------------------- BASELINE MULTICLASS MODELS SPLIT IN BJET CAT with Gatja Scores ------------------------------

Cat_3b = DenseClassifierDL.derive("Cat_3b", cls_dict={
    "preparation_producer_name": "prepml_eq3b",
    "processes": (
        *processes.hhh,
        "tthh",
        "tth",
        "tt_custom",
        "ttbb_custom",
    ),
    "train_nodes": {
        "hhh_signal": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": processes.hhh,
        },
        "tthh": {"ml_id": 1, "label": r"ttHH"},
        "tth": {"ml_id": 2},
        "tt_custom": {
            "ml_id": 3,
            "label": r"tt",
        },
        "ttbb_custom": {
            "ml_id": 4,
            "label": r"ttbb",
        },
    },
    "input_features": input_features["gatja_inputs"],
})

Cat_4b = DenseClassifierDL.derive("Cat_4b", cls_dict={
    "preparation_producer_name": "prepml_ge4b",
    "processes": (
        *processes.hhh,
        "tthh",
        "tth",
        "ttbb_custom",
    ),
    "train_nodes": {
        "hhh_signal": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": processes.hhh,
        },
        "tth": {"ml_id": 1},
        "tthh": {"ml_id": 2},
        "ttbb_custom": {
            "ml_id": 3,
            "label": r"ttbb",
        },
    },
    "class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "tthh": 1,
        "ttbb_custom": 2,
        "tth": 2,
    },
    "input_features": input_features["gatja_inputs"],
})
