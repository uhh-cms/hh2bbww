# coding: utf-8

"""
ML models using the MLClassifierBase and Mixins
"""

from __future__ import annotations
from hbw.production.ml_trih_inputs import GATJA_MODELS

import law
import json
import os

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
        "st", "dy", "h", "tt_custom", "ttbb_custom",  # "tthh"  # "tt_bb_custom", "tt2b_custom", "ttb_custom" # "ttbb_custom",  # "tthh_4b",   # "vhh_4b",  # noqa E501
    ],
    "backgrounds_v1": [
        "h", "tt_custom", "ttbb_custom",  # "tthh"  # "tt_bb_custom", "tt2b_custom", "ttb_custom" # "ttbb_custom",  # "tthh_4b",   # "vhh_4b",  # noqa E501
    ],
    "backgrounds_v2": [
        "tth", "tthh", "tt_custom", "ttbb_custom",  # "tthh"  # "tt_bb_custom", "tt2b_custom", "ttb_custom" # "ttbb_custom",  # "tthh_4b",   # "vhh_4b",  # noqa E501
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
    "gatja_scores_reduced": [
        f"gatja_output_{i}" for i in range(12)
    ],
    "gatja_scores_reduced_bin": [
        f"gatja_output_{i}" for i in range(9)
    ],
    "gatja_scores_opt_bin": [
        "gatja_output_0",
        "gatja_output_1",
        "gatja_output_2",
        "gatja_output_3",
        "gatja_output_4",
        "gatja_output_6",
        "gatja_output_7",
        "gatja_output_9",
        "gatja_output_10",
    ],
    "gatja_scores_opt_3b": [
        "gatja_output_0",
        "gatja_output_1",
        "gatja_output_3",
        "gatja_output_4",
        "gatja_output_6",
        "gatja_output_7",
        "gatja_output_9",
        "gatja_output_10",
        "gatja_output_15",
    ],
    "gatja_scores_opt_4b": [
        "gatja_output_0",
        "gatja_output_1",
        "gatja_output_2",
        "gatja_output_3",
        "gatja_output_4",
        "gatja_output_6",
        "gatja_output_7",
        "gatja_output_9",
        "gatja_output_10",
        "gatja_output_12",
        "gatja_output_15",
    ],
})
input_features["gatja_inputs"] = input_features["expanded_hhh_inputs"] + \
    input_features["gatja_scores"]
input_features["gatja_inputs_opt_3b"] = input_features["expanded_hhh_inputs"] + \
    input_features["gatja_scores_opt_3b"]
input_features["gatja_inputs_opt_4b"] = input_features["expanded_hhh_inputs"] + \
    input_features["gatja_scores_opt_4b"]
input_features["gatja_inputs_opt_bin"] = input_features["expanded_hhh_inputs"] + \
    input_features["gatja_scores_opt_4b"]
input_features["gatja_inputs_reduced"] = input_features["expanded_hhh_inputs"] + \
    input_features["gatja_scores_reduced"]
input_features["gatja_inputs_reduced_bin"] = input_features["expanded_hhh_inputs"] + \
    input_features["gatja_scores_reduced_bin"]

# input_features["gatja_inputs"] = input_features["expanded_hhh_inputs"] + input_features["gatja_scores"]
input_features[
    "gatja_scores_extern_evaluation"
] = (
    input_features["expanded_hhh_inputs"] + input_features["gatja_scores"]
)

configs = DotDict({
    "22post": lambda self, requested_configs: ["c22postv14"],
    "23pre": lambda self, requested_configs: ["c23prev14"],
    "22": lambda self, requested_configs: ["c22prev14", "c22postv14"],
    "23": lambda self, requested_configs: ["c23prev14", "c23postv14"],
    "full": lambda self, requested_configs: ["c22prev14", "c22postv14", "c23prev14", "c23postv14"],
    "2425": lambda self, requested_configs: ["c24v15", "c25v15"],
})

#
# derived MLModels for HHH
#

# ----------------------- BASELINE BINARY MODELS FOR HHH SIGNAL ------------------------------

Bin = DenseClassifierDL.derive("Bin", cls_dict={
    "input_features": input_features["gatja_inputs"],
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
    "preparation_producer_name": "prepml_geq3b",
    "training_configs": configs["2425"],
})

Bin_V2 = Bin.derive("Bin_V2", cls_dict={
    "processes": [
        *processes.hhh,
        *processes.backgrounds_v2,
    ],
    "train_nodes": {
        "sig_hhh_binary": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",
            "class_factor_mode": "equal",
            "sub_processes": processes.hhh,
        },
        "bkg_binary_for_hhh": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",
            "class_factor_mode": "equal",      # <- was "xsec"
            "sub_processes": processes.backgrounds_v2,
        },
    },
    "sub_process_class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "hhh_4b2w_2l2nu_c3m1_d40": 0.5,
        "tt_custom": 1,
        "ttbb_custom": 1,
        "tth": 1,
        "tthh": 0.5,
    },
    "class_factors": {
        "sig_hhh_binary": 1,
        "bkg_binary_for_hhh": 1,
    },
    "input_features": input_features["gatja_inputs_reduced_bin"],
})

Bin_V2_without_gatja = Bin_V2.derive("Bin_V2_without_gatja", cls_dict={
    "input_features": input_features["expanded_hhh_inputs"],
})
Bin_V2_opt_gatja = Bin_V2.derive("Bin_V2_opt_gatja", cls_dict={
    "input_features": input_features["gatja_inputs_opt_bin"],
})
Bin_V3 = Bin_V2.derive("Bin_V3", cls_dict={
    "input_features": input_features["gatja_inputs"],
})

Bin_V3_test_tt_rate = Bin_V2.derive("Bin_V3_test_tt_rate", cls_dict={
    "preparation_producer_name": "prepml_tt_rate_eq3b",
})

# baseline: everything trained full pahse space --> HH approach
Bin_weight = Bin.derive("Bin_weight", cls_dict={
    "sub_process_class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "hhh_4b2w_2l2nu_c3m1_d40": 0.5,
        "ttbb_custom": 1,
        "st": 1,
        "dy": 1,
        "ttv": 1,
        "h": 1,
    },
})

Bin_V2_best = Bin_V2.derive("Bin_V2_best", cls_dict={
    "class_factors": {"sig_hhh_binary": 1, "bkg_binary_for_hhh": 3},
    "sub_process_class_factors": {
        "hhh_4b2w_2l2nu_c3m1_d40": 1.0,
        "tt_custom": 1,
        "ttbb_custom": 0.5,
        "tth": 0.5,
        "tthh": 0.5,
    },
    "batchsize": 512,
    "dropout": 0.45,
    "layers": (512, 512, 512, 512),
    "learningrate": 0.00030132383348625727,
    "reduce_lr_patience": 5,
    "epochs": 100,
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
    "training_configs": configs["2425"],
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
    "training_configs": configs["2425"],
})

Cat_3b_weighted = Cat_3b.derive("Cat_3b_weighted", cls_dict={
    "sub_process_class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "hhh_4b2w_2l2nu_c3m1_d40": 0.5,
        # "tthh": 0.18,
    },
    "class_factors": {
        "tthh": 0.18,
        "ttbb_custom": 0.5,
    },
})

Cat_3b_without_gatja = Cat_3b.derive("Cat_3b_without_gatja", cls_dict={
    "input_features": input_features["expanded_hhh_inputs"],
})
Cat_3b_opt_gatja = Cat_3b.derive("Cat_3b_opt_gatja", cls_dict={
    "input_features": input_features["gatja_inputs_opt_3b"],
})
Cat_3b_test_tt_rate = Cat_3b.derive("Cat_3b_test_tt_rate", cls_dict={
    "preparation_producer_name": "prepml_tt_rate_eq3b",
})

Cat_3b_best_V2 = Cat_3b.derive("Cat_3b_best_V2", cls_dict={
    "class_factors": {"hhh_signal": 1, "tthh": 2, "tth": 3, "ttbb_custom": 3, "tt_custom": 4},
    "sub_process_class_factors": {"hhh_4b2w_2l2nu_c3m1_d40": 1.0},
    "batchsize": 2048,
    "dropout": 0.25,
    "layers": (256, 256, 256),
    "learningrate": 0.0004204836974036755,
    "reduce_lr_patience": 5,
    "epochs": 100,
})

Cat_incl = Cat_3b.derive("Cat_incl", cls_dict={
    "preparation_producer_name": "prepml_geq3b",
})

Cat_incl_best = Cat_incl.derive("Cat_incl_best", cls_dict={
    "class_factors": {"hhh_signal": 2, "tthh": 3, "tth": 4, "ttbb_custom": 4, "tt_custom": 4},
    "sub_process_class_factors": {"hhh_4b2w_2l2nu_c3m1_d40": 0.5},
    "batchsize": 512,
    "dropout": 0.3,
    "layers": (256, 256, 256),
    "learningrate": 0.00047347146205006644,
    "reduce_lr_patience": 6,
    "epochs": 100,
})

Cat_3b_weighted_V2 = Cat_3b.derive("Cat_3b_weighted_V2", cls_dict={
    "sub_process_class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "hhh_4b2w_2l2nu_c3m1_d40": 0.5,
        # "tthh": 0.18,
    },
    "class_factors": {
        "tthh": 1.,
        "ttbb_custom": 1.,
    },
    "input_features": input_features["gatja_inputs_reduced"],
})

Cat_3b_best = Cat_3b.derive("Cat_3b_best", cls_dict={
    "class_factors": {
        "hhh_signal": 2,
        "tthh": 3,
        "tth": 3,
        "ttbb_custom": 4,
        "tt_custom": 5,
    },
    "sub_process_class_factors": {
        "hhh_4b2w_2l2nu_c3m1_d40": 0.5,
    },
    "batchsize": 2048,
    "dropout": 0.25,
    "layers": (512, 512, 512, 512),
    "learningrate": 0.0005759125610097092,
    "reduce_lr_patience": 4,
    "epochs": 100,
})


Cat_4b = DenseClassifierDL.derive("Cat_4b", cls_dict={
    "preparation_producer_name": "prepml_geq4b",
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
    "training_configs": configs["2425"],
})

Cat_4b_weighted = Cat_4b.derive("Cat_4b_weighted", cls_dict={
    "sub_process_class_factors": {
        "hhh_4b2w_2l2nu_c3m1_d40": 0.5,   # all other hhh stay at 1
    },
    "class_factors": {
        "tthh": 0.5,
        "tth": 1,
        "ttbb_custom": 0.25,
    },
    "batchsize": 900,   # must give 900 // 209 = 4
})
Cat_4b_weighted_V2 = Cat_4b.derive("Cat_4b_weighted_V2", cls_dict={
    "sub_process_class_factors": {
        "hhh_4b2w_2l2nu_c3m1_d40": 0.5,   # all other hhh stay at 1
    },
    "class_factors": {
        "tthh": 1,
        "tth": 1,
        "ttbb_custom": 1,
    },
    "input_features": input_features["gatja_inputs_reduced"],
    # "batchsize": 900,   # must give 900 // 209 = 4
})

Cat_4b_best = Cat_4b.derive("Cat_4b_best", cls_dict={
    "class_factors": {"hhh_signal": 2, "tthh": 3, "tth": 3, "ttbb_custom": 3},
    "sub_process_class_factors": {"hhh_4b2w_2l2nu_c3m1_d40": 0.5},
    "batchsize": 512,
    "dropout": 0.0,
    "layers": (512, 512, 512, 512),
    "learningrate": 0.002601767195273077,
    "reduce_lr_patience": 6,
    "epochs": 100,
})

Cat_4b_best_V2 = Cat_4b.derive("Cat_4b_best_V2", cls_dict={
    "class_factors": {"hhh_signal": 1, "tthh": 1, "tth": 1, "ttbb_custom": 1},
    "sub_process_class_factors": {"hhh_4b2w_2l2nu_c3m1_d40": 0.5},
    "batchsize": 512,
    "dropout": 0.15,
    "layers": (512, 512, 512, 512),
    "learningrate": 0.0015061841909343758,
    "reduce_lr_patience": 5,
    "epochs": 100,
})

Cat_4b_without_gatja = Cat_4b.derive("Cat_4b_without_gatja", cls_dict={
    "input_features": input_features["expanded_hhh_inputs"],
})

Cat_4b_opt_gatja = Cat_4b.derive("Cat_4b_opt_gatja", cls_dict={
    "input_features": input_features["gatja_inputs_opt_4b"],
})
Cat_4b_test_tt_rate = Cat_4b.derive("Cat_4b_test_tt_rate", cls_dict={
    "preparation_producer_name": "prepml_tt_rate_geq4b",
})
#  -----------  Checks to doa fair comparison next to each other ----------------------------------------------


baseline_hh = Cat_3b.derive("baseline_hh", cls_dict={
    "input_features": input_features["expanded_hhh_inputs"],
    "preparation_producer_name": "prepml_geq3b",
})

baseline_hhh_incl = Cat_3b.derive("baseline_hhh_incl", cls_dict={
    "preparation_producer_name": "prepml_geq3b",
})

baseline_hh_bin = Bin.derive("baseline_hh_bin", cls_dict={
    "input_features": input_features["expanded_hhh_inputs"],
})

# baseline categorisation: test splitting the categoroisation task
baseline_cat3b = Cat_3b.derive("baseline_cat3b", cls_dict={
    "input_features": input_features["expanded_hhh_inputs"],
})
baseline_cat4b = Cat_4b.derive("baseline_cat4b", cls_dict={
    "input_features": input_features["expanded_hhh_inputs"],
})

# Binary network dropped --> has to be handled on inference side

# use Gatja as categoriser: Binary above certain Gatja score
cat_by_gatja_3b = Cat_3b.derive("cat_by_gatja_3b", cls_dict={
    "preparation_producer_name": "prepml_non_gatja_h1_conservative",
    "processes": (
        "tthh",
        "tth",
        "tt_custom",
        "ttbb_custom",
    ),
    "train_nodes": {
        "tthh": {"ml_id": 0, "label": r"ttHH"},
        "tth": {"ml_id": 1},
        "tt_custom": {
            "ml_id": 2,
            "label": r"tt",
        },
        "ttbb_custom": {
            "ml_id": 3,
            "label": r"ttbb",
        },
    },
})
cat_by_gatja_4b = Cat_4b.derive("cat_by_gatja_4b", cls_dict={
    "preparation_producer_name": "prepml_non_gatja_h1_conservative",
    "processes": (
        "tthh",
        "tth",
        "ttbb_custom",
    ),
    "train_nodes": {
        "tthh": {"ml_id": 0, "label": r"ttHH"},
        "tth": {"ml_id": 1},
        "ttbb_custom": {
            "ml_id": 2,
            "label": r"ttbb",
        },
    },
})
cat_by_gatja_bin = Bin.derive("cat_by_gatja_bin", cls_dict={
    "preparation_producer_name": "prepml_gatja_h1_conservative",
    "processes": [
        *processes.hhh,
        "ml_bkg_hhh",
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
            "sub_processes": ["ml_bkg_hhh"],
        },
    },
})

# use Gatja as additional cut:
gatja_cut_3b = Cat_3b.derive("gatja_cut_3b", cls_dict={
    "preparation_producer_name": "prepml_gatja_h1",
})
gatja_cut_4b = Cat_4b.derive("gatja_cut_4b", cls_dict={
    "preparation_producer_name": "prepml_gatja_h1",
})
gatja_cut_bin = Bin.derive("gatja_cut_bin", cls_dict={
    "preparation_producer_name": "prepml_gatja_h1",
})


class Cat3bGatja(Cat_3b):
    def training_producers(self, analysis_inst, requested_producers):
        return ["event_weights", "hhh_dl_ml_inputs", "input_features_gatja"]


for m in GATJA_MODELS:
    Cat3bGatja.derive(f"Cat_3b_{m}", cls_dict={
        "preparation_producer_name": f"prepml_eq3b_gatja_{m}",
        "preparation_producer_in_ml_evaluation": True,
        "input_features": input_features["gatja_inputs"],
    })

_hpo = json.loads(os.environ.get("HPO_SETTINGS", "{}"))
Cat_3b_hpo = Cat_3b.derive("Cat_3b_hpo", cls_dict=_hpo)

Cat_4b_hpo = Cat_4b.derive("Cat_4b_hpo", cls_dict=_hpo)
Cat_incl_hpo = Cat_incl.derive("Cat_incl_hpo", cls_dict=_hpo)

Bin_V2_hpo = Bin_V2.derive("Bin_V2_hpo", cls_dict=_hpo)
