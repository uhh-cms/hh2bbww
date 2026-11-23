
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
        "tt_custom", "st",
        "vv", "ttv", "h", "tthh", "other",
    ],
})
input_features = DotDict({
    "minimal": DenseClassifierDCTR.input_features,
    "min": [
        # new input features:
        "mli_n_jet",
        "mli_discrete_b_score_sum",
    ],
    "low_level_bjet": [
        # new input features:
        "mli_n_btag",
        "mli_discrete_b_score_sum",
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
    "reduced_tth": [
        # new input features:
        "mli_n_btag",
        "mli_n_jet",
        "mli_ht",
        "mli_ht_lf",
        "mli_ht_hf",
        "mli_dr_hbb",
        "mli_ll_pt",
        "mli_discrete_b_score_sum",

    ],
    "reduced_tth_without_score": [
        # new input features:
        "mli_n_btag",
        "mli_n_jet",
        "mli_ht",
        "mli_ht_lf",
        "mli_ht_hf",
        "mli_dr_hbb",
        "mli_ll_pt",

    ],
})

class_factors = {
    "default": DenseClassifierDCTR._default__class_factors,
}

configs = DotDict({
    "run3_2425": lambda self, requested_configs: ["c24v15", "c25v15"],
})

#
# derived MLModels
#

# C2ST / MC inclusive DCTR networks
c2st_24_closure = DenseClassifierDCTR.derive("c2st_24_closure", cls_dict={
    "training_configs": configs.run3_2425,
    "input_features": input_features.min,
    "processes": [
        "data",
        *processes.backgrounds,
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
            "sub_processes": processes.backgrounds,
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

# # dctr / process specific networks
dctr_test_ttbb_v1 = DenseClassifierDCTR.derive("dctr_test_ttbb_v1", cls_dict={
    "training_configs": configs.run3_2425,
    "input_features": input_features.min,
    "processes": [
        "ttbb_custom",
        "data",
        *processes.backgrounds,
    ],
    "train_nodes": {
        "ttbb_only": {
            "ml_id": 0,
            "label": "ttbb MC",
            "color": "#e76300",
            "sub_processes": ["ttbb_custom"],
        },
        "ttbb_data": {
            "ml_id": 1,
            "label": "ttbb Data",
            "color": "#000000",
            "sub_processes": ["data", *processes.backgrounds],
        },
    },
    "loss_signs": {
        "ttbb_custom": 1,
        "data": 1,
        "tt_custom": -1, "st": -1, "vv": -1, "ttv": -1, "h": -1, "tthh": -1, "other": -1,
    },
    "negative_weights": "ignore",
    "preparation_producer_name": "prepml_ttcr_em",
})

dctr_test_ttbb_v2 = dctr_test_ttbb_v1.derive("dctr_test_ttbb_v2", cls_dict={
    "input_features": input_features.low_level_bjet,
})

dctr_test_ttbb_v3 = dctr_test_ttbb_v1.derive("dctr_test_ttbb_v3", cls_dict={
    "input_features": input_features.reduced_tth,
    "epochs": 100,
    "learningrate": 0.0005,
})
dctr_test_ttbb_v4 = dctr_test_ttbb_v3.derive("dctr_test_ttbb_v4", cls_dict={
    "input_features": input_features.reduced_tth_without_score,
})
