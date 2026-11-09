
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
from hbw.ml.derived.dctr import DenseClassifierDCTR


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)

#
# configs
#
processes = DotDict({
    "backgrounds": [
        "tt", "st", "dy_m10to50", "dy_m50toinf",
        "vv", "ttv", "h", "other",
    ],
    "backgrounds_nondy": [
        "tt", "st",
        "vv", "ttv", "h", "other",
    ],
    "backgrounds_binary_highdy": [
        "tt", "st", "dy_m50toinf",
        "vv", "ttv", "h", "other",
    ],
    "backgrounds_multiclass": [
        "tt", "st", "dy_ee_m10to50", "dy_mumu_m10to50", "dy_m50toinf",
        "h",  # "vv",
    ],
})
input_features = DotDict({
    "minimal": DenseClassifierDCTR.input_features,
    "v32_discrete": [
        # new input features:
        "mli_dr_ll_bb",  # used instead of mli_dr_bb_llMET
        # v1 sorted from GGF SHAP
        "mli_mbbllMET",
        "mli_b1_pt",
        "mli_mbb",
        "mli_mll",
        "mli_bb_pt",
        "mli_mllMET",
        "mli_lep_pt",
        "mli_maxdr_jj",
        "mli_mixed_channel",
        "mli_dr_bb",
        "mli_lep2_pt",
        "mli_b2_pt",
        "mli_dr_ll",
        "mli_met_pt",
        "mli_min_dr_llbb",
        # "mli_discrete_b_score_sum",
        "mli_ll_pt",
        "mli_ht",
        # "mli_mindr_lb",  # removed: correlated with mli_min_dr_llbb
        # "mli_dr_bb_llMET",  # using mli_dr_ll_bb instead
        "mli_j1_eta",
        "mli_n_jet",
        "mli_j1_pt",
        "mli_b1_discrete_b_score",
        "mli_b2_discrete_b_score",
        "mli_j1_discrete_b_score",
        "mli_j2_discrete_b_score",
        "mli_boosted_cand_exists",
    ],
    "v3_discrete": [
        # new input features:
        "mli_dr_ll_bb",  # used instead of mli_dr_bb_llMET
        # v1 sorted from GGF SHAP
        "mli_mbbllMET",
        "mli_b1_pt",
        "mli_mbb",
        "mli_mll",
        "mli_bb_pt",
        "mli_mllMET",
        "mli_lep_pt",
        "mli_maxdr_jj",
        "mli_mixed_channel",
        "mli_dr_bb",
        "mli_lep2_pt",
        "mli_b2_pt",
        "mli_dr_ll",
        "mli_met_pt",
        "mli_min_dr_llbb",
        # "mli_discrete_b_score_sum",
        "mli_ll_pt",
        "mli_ht",
        # "mli_mindr_lb",  # removed: correlated with mli_min_dr_llbb
        # "mli_dr_bb_llMET",  # using mli_dr_ll_bb instead
        "mli_j1_eta",
        "mli_n_jet",
        "mli_j1_pt",
        "mli_b1_discrete_b_score",
        "mli_b2_discrete_b_score",
        "mli_j1_discrete_b_score",
        "mli_j2_discrete_b_score",
    ],
    "c2st_reduced": [
        "mli_dr_ll_bb",  # used instead of mli_dr_bb_llMET
        # v1 sorted from GGF SHAP
        "mli_mbbllMET",
        "mli_b1_pt",
        "mli_mbb",
        "mli_mll",
        "mli_bb_pt",
        "mli_mllMET",
        "mli_lep_pt",
        "mli_maxdr_jj",
        # "mli_mixed_channel",
        "mli_dr_bb",
        "mli_lep2_pt",
        "mli_b2_pt",
        "mli_dr_ll",
        "mli_met_pt",
        "mli_min_dr_llbb",
        # "mli_discrete_b_score_sum",
        "mli_ll_pt",
        "mli_ht",
        # "mli_mindr_lb",  # removed: correlated with mli_min_dr_llbb
        # "mli_dr_bb_llMET",  # using mli_dr_ll_bb instead
        "mli_j1_eta",
        "mli_n_jet",
        "mli_j1_pt",
    ],
})

class_factors = {
    "default": DenseClassifierDCTR._default__class_factors,
}

configs = DotDict({
    "22post": lambda self, requested_configs: ["c22postv14"],
    "22": lambda self, requested_configs: ["c22prev14", "c22postv14"],
    "23": lambda self, requested_configs: ["c23prev14", "c23postv14"],
    "full": lambda self, requested_configs: ["c22prev14", "c22postv14", "c23prev14", "c23postv14"],
    "twentyfour": lambda self, requested_configs: ["c24v15"],
    "twentyfive": lambda self, requested_configs: ["c25v15"],
    "run3_2425": lambda self, requested_configs: ["c24v15", "c25v15"],
})

#
# derived MLModels
#

# C2ST / MC inclusive DCTR networks
c2st_24_closure = DenseClassifierDCTR.derive("c2st_24_closure", cls_dict={
    "training_configs": configs.twentyfour,
    "input_features": input_features.c2st_reduced,
    "processes": [
        "data",
        *processes.backgrounds_binary_highdy,
    ],
    "train_nodes": {
        "data": {
            "ml_id": 0,
            "label": "Data",
            "color": "#000000",
            "class_factor_mode": "equal",
        },
        "mc_bkg": {
            "ml_id": 1,
            "label": "MC Background",
            "color": "#e76300",
            "class_factor_mode": "equal",
            "sub_processes": processes.backgrounds_binary_highdy,
        },
    },
    "preparation_producer_name": "prepml_dyvr_mumuV02",
    "epochs": 50,
    "layers": (64,),
    "dropout": 0.0,
    "l2regularization": 0.0,
    "weighting": "c2st",
    "optimizer": "sgd",
})
c2st_24_plots = c2st_24_closure.derive("c2st_24_plots")
c2st_24_learn = c2st_24_closure.derive("c2st_24_learn", cls_dict={
    "learningrate": 0.005,
})
c2st_24_switched = c2st_24_closure.derive("c2st_24_switched", cls_dict={
    "train_nodes": {
        "mc_bkg": {
            "ml_id": 0,
            "label": "MC Background",
            "color": "#e76300",
            "class_factor_mode": "equal",
            "sub_processes": processes.backgrounds_binary_highdy,
        },
        "data": {
            "ml_id": 1,
            "label": "Data",
            "color": "#000000",
            "class_factor_mode": "equal",
        },
    },
    "learningrate": 0.005,
})
c2st_24_minimal = c2st_24_closure.derive("c2st_24_minimal", cls_dict={
    "input_features": input_features.minimal,
    "learningrate": 0.005,
})
c2st_24_min_njlq32 = c2st_24_closure.derive("c2st_24_min_njlq32", cls_dict={
    "input_features": input_features.minimal,
    "learningrate": 0.005,
    "preparation_producer_name": "prepml_dyvr_mumuV1_njlq3",
})

# # dctr / process specific networks
dctr_24_dy = DenseClassifierDCTR.derive("dctr_24_dy", cls_dict={
    "training_configs": configs.twentyfour,
    "input_features": input_features.c2st_reduced,
    "processes": [
        "dy_m50toinf",
        "data",
        *processes.backgrounds_nondy,
    ],
    "train_nodes": {
        "dy_only": {
            "ml_id": 0,
            "label": "DY",
            "color": "#e76300",
            "sub_processes": ["dy_m50toinf"],
        },
        "dy_data": {
            "ml_id": 1,
            "label": "DY Data",
            "color": "#000000",
            "sub_processes": ["data", *processes.backgrounds_nondy],
        },
    },
    "loss_signs": {
        "dy_m50toinf": 1,
        "data": 1,
        "tt": -1, "st": -1, "vv": -1, "ttv": -1, "h": -1, "other": -1,
    },
    "negative_weights": "ignore",
    "preparation_producer_name": "prepml_dyvr_mumuV02",
})
dctr_24_dy_reduced = DenseClassifierDCTR.derive("dctr_24_dy_reduced", cls_dict={
    "training_configs": configs.twentyfour,
    "input_features": input_features.c2st_reduced,
    "processes": [
        "dy_m50toinf",
        "data",
        "tt", "st", "vv", "ttv", "h",
    ],
    "train_nodes": {
        "dy_only": {
            "ml_id": 0,
            "label": "DY",
            "color": "#e76300",
            "sub_processes": ["dy_m50toinf"],
        },
        "dy_data": {
            "ml_id": 1,
            "label": "DY Data",
            "color": "#000000",
            "sub_processes": ["data", "tt", "st", "vv", "ttv", "h"],
        },
    },
    "loss_signs": {
        "dy_m50toinf": 1,
        "data": 1,
        "tt": -1, "st": -1, "vv": -1, "ttv": -1, "h": -1, "other": -1,
    },
    "negative_weights": "ignore",
    "preparation_producer_name": "prepml_dyvr_mumuV02",
})
dctr_24_2d = dctr_24_dy_reduced.derive("dctr_24_2d", cls_dict={
    "input_features": input_features.minimal,
})
dctr_24_dy_2d = dctr_24_2d.derive("dctr_24_dy_2d")
