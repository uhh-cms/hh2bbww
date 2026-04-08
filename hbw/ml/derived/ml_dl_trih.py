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
    "hhh_4b2w_2l2nu_c30_d4m1",
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
    "merge_hh": ["sig_ggf", "sig_vbf", "tt", "st", "dy", "h"],
    "backgrounds_binary": [
        "tt", "st", "dy_m10to50", "dy_m50toinf",
        "vv", "ttv", "h", "other",
    ],
    "backgrounds_multiclass": [
        "tt", "st", "dy_m10to50", "dy_m50toinf",
        "vv", "h",
    ],
    "backgrounds_hhh": [
        "tt", "st", "dy", "h", "hh_ggf", "hh_vbf", "other",
    ],
    "backgrounds_hhh_v2": [
        "st", "dy", "h", "tt_custom", "ttbb_custom",  # "tthh_4b",   # "vhh_4b",
    ],
    "sig_hhh": [
        "hhh_4b2w_2l2nu_c30_d40",
        "hhh_4b2w_2l2nu_c30_d499",
        "hhh_4b2w_2l2nu_c30_d4m1",
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
    "previous": [
        # event features
        "mli_ht", "mli_n_jet", "mli_n_btag",
        "mli_b_score_sum",
        # bb system
        "mli_dr_bb", "mli_dphi_bb", "mli_mbb", "mli_bb_pt",
        "mli_mindr_lb",
        # ll system
        "mli_mll", "mli_dr_ll", "mli_dphi_ll", "mli_ll_pt",
        "mli_min_dr_llbb",
        "mli_dphi_bb_nu", "mli_dphi_bb_llMET", "mli_mllMET",
        "mli_mbbllMET", "mli_dr_bb_llMET",
        # VBF features
        "mli_vbf_deta", "mli_vbf_mass", "mli_vbf_tag",
        # low-level features
        "mli_met_pt",
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["b1", "b2", "j1"]
        for var in ["pt", "eta", "b_score"]
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["lep", "lep2"]
        for var in ["pt", "eta"]
    ],
    # NOTE: ATM This is basically all that is available but this should be refined
    "hhh_v0": [
        "mli_mbb1",
        "mli_mbb2",
        "mli_bb_pt",
        "mli_ll_pt",
        "mli_n_jet",
        "mli_dr_bb_bb",
        "mli_dr_ll_bb1",
        "mli_dr_ll_bb2",
        "mli_met_pt",
        "mli_mhhh",
        "mli_m4bllMET",
        "mli_dr_bb1_llMET",
        "mli_dr_bb2_llMET",
        "mli_mll",
        "mli_b_score_sum",
        "mli_mllMET",
        "mli_b1_pt",
        "mli_b2_pt",
        "mli_j1_pt",
        "mli_ht",
        "mli_lep_pt",
        "mli_lep_eta",
        "mli_b1_eta",
        "mli_b2_eta",
        "mli_j1_eta",
        "mli_j2_eta",
        "mli_mixed_channel",
        "mli_lep2_pt",
        "mli_lep2_eta",
        "mli_mllMET",
        "mli_n_jet",
        "mli_n_btag",
        "mli_maxdr_jj",
        "mli_mindr_jj",
        "mli_dr_bb",
    ],
    "hhh_v1": [
        "mli_n_jet",
        "mli_mllMET",
        "mli_hb_candidate_mh1",
        "mli_mll",
        "mli_n_btag",
        "mli_lep2_pt",
        "mli_lep_pt",
        "mli_ll_pt",
        "mli_dr_ll",
        "mli_dphi_ll",
        "mli_deta_ll",
        "mli_met_pt",
        "mli_maxdr_jj",
        "mli_mindr_jj",
        "mli_ht",
        "mli_hb_candidate3_discrete_b_score",
        "mli_hb_candidate4_discrete_b_score",
        "mli_hb_candidate1_discrete_b_score",  # analog to btag 
        "mli_hb_candidate2_discrete_b_score",  # analog to btag 
        "mli_hb_candidate3_eta",
        "mli_hb_candidate4_eta",
        "mli_hb_candidate1_eta",  # analog to btag 
        "mli_hb_candidate2_eta",  # analog to btag 
        "mli_hb_candidate3_pt",
        "mli_hb_candidate4_pt",
        "mli_hb_candidate1_pt",  # analog to btag 
        "mli_hb_candidate2_pt",  # analog to btag 
        "mli_hb_candidate_mh1",  # analog to btag 
        "mli_hb_candidate_mhhh",  # analog to btag 
        "mli_hb_candidate_dr_h1_h2",  # analog to btag 
        "mli_hb_candidate_dr_ll_h1",  # analog to btag 
    ],
})

class_factors_hhh = {
    "default": DenseClassifierDL._default__class_factors,
    "ones": {},  # defaults to 1 (NOTE: do not try to use defaultdict! does not work with hash generation)
    "benchmark": {
        "sig_hhh": 1,
        "hh": 1,
        "tt": 8,
        "st": 2,
        "dy": 2,
        "h": 1,
    },
}

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

# multiclass_hhh_v0 = DenseClassifierDL.derive("multiclass_hhh_v0", cls_dict={
#     # "training_configs": configs.23pre,
#     "input_features": input_features["hhh_v0"],
#     "processes": (
#         *processes.sig_hhh,
#         "tt",
#         "st",
#         "dy",
#         "h",
#         "hh_ggf",
#         "hh_vbf",
#         # "tthh_4b",
#         # "vhh_4b",
#     ),
#     "train_nodes": {
#         "sig_hhh": {
#             "ml_id": 0,
#             "label": r"HHH",
#             "color": "#000000",  # black
#             "class_factor_mode": "equal",
#             "sub_processes": processes.sig_hhh,
#         },
#         # "hhh_4b2w_2l2nu_c30_d40": {"ml_id": 0, "label": r"HHH"},
#         "hh_bkg": {
#             "ml_id": 5,
#             "label": r"HH",
#             "color": "#999999",  # grey
#             "class_factor_mode": "xsec",
#             # "sub_processes": ["hh_ggf", "hh_vbf", "tthh_4b", "vhh_4b"],
#             "sub_processes": ["hh_ggf", "hh_vbf"],
#         },
#         "h": {"ml_id": 4},
#         "tt": {"ml_id": 1},
#         "st": {"ml_id": 2},
#         "dy": {"ml_id": 3},
#     },
#     "preparation_producer_name": "prepml_hhh_sr",
# })

# hhh_v0 = DenseClassifierDL.derive("hhh_v0", cls_dict={
#     # "training_configs": configs.23pre,
#     "input_features": input_features["hhh_v0"],
#     "processes": [
#         "hhh_4b2w_2l2nu_c30_d40",
#         *processes.backgrounds_hhh,
#         # *processes.backgrounds_hhh_v1,
#     ],
#     "train_nodes": {
#         "sig_hhh_binary": {
#             "ml_id": 0,
#             "label": r"HHH",
#             "color": "#000000",  # black
#             "class_factor_mode": "equal",
#             "sub_processes": processes.sig_hhh,
#         },
#         # "hhh_4b2w_2l2nu_c30_d40": {"ml_id": 0, "label": r"HHH"},
#         "bkg_binary_for_hhh": {
#             "ml_id": 1,
#             "label": "Background",
#             "color": "#e76300",  # Spanish Orange
#             "class_factor_mode": "xsec",
#             "sub_processes": processes.backgrounds_hhh,
#             # "sub_processes": processes.backgrounds_hhh_v1,
#         },
#     },
#     # relative class factors between different nodes
#     "class_factors": {
#         "sig_hhh_binary": 1,
#         "bkg_binary_for_hhh": 1,
#     },
#     # relative process weights within one class
#     "sub_process_class_factors": {
#         "hhh_4b2w_2l2nu_c30_d40": 2,
#         "tt": 1,
#         "st": 1,
#         "dy": 1,
#         "ttv": 2,
#         "h": 2,
#         # "vv": 2,
#         "hh_ggf": 4,
#         "hh_vbf": 4,
#         # "hh_other": 4,
#         "other": 8,
#     },
#     "epochs": 100,
#     "preparation_producer_name": "prepml_sr",
# })


# 2024 test

multiclass_hhh_v2 = DenseClassifierDL.derive("multiclass_hhh_v2", cls_dict={
    # "training_configs": ["c24v15"],
    "input_features": input_features["hhh_v0"],
    "processes": (
        "hhh_4b2w_2l2nu_c30_d40",
        # "tt_dl_nonb",
        # "tt_sl_nonb",
        # "tt_fh_nonb",
        # "ttbb_dl_1b",
        # "ttbb_sl_1b",
        # "ttbb_fh_1b",
        "tt_custom",
        "ttbb_custom",
        "st",
        "dy",
        "h",
        # "hh_ggf",
        # "hh_vbf",
        "tthh_4b",
        # "vhh_4b",
        # "vv",
        # "other",
    ),
    "train_nodes": {
        "hhh_4b2w_2l2nu_c30_d40": {"ml_id": 0, "label": r"HHH"},
        "tthh_4b": {"ml_id": 1, "label": r"HH"},
        "h": {"ml_id": 2},
        "tt_custom": {"ml_id": 3},
        "ttbb_custom": {"ml_id": 4},
        # "tt_nonb_custom": {
        #     "ml_id": 3,
        #     "label": r"tt",
        #     "color": "#000000",  # black
        #     "class_factor_mode": "xsec",
        #     "sub_processes": ["tt_dl_nonb", "tt_sl_nonb", "tt_fh_nonb"],
        # },
        # "ttbb_1b_custom": {
        #     "ml_id": 4,
        #     "label": r"tt + b",
        #     "color": "#000000",  # black
        #     "class_factor_mode": "xsec",
        #     "sub_processes": ["ttbb_dl_1b", "ttbb_sl_1b", "ttbb_fh_1b"],
        # },
        "st": {"ml_id": 5},
        "dy": {"ml_id": 6},
    },
    "class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "tthh_4b": 1,
        "tt_custom": 8,
        "ttbb_custom": 4,
        "st": 2,
        "dy": 2,
        "h": 1,
    },
    # # relative process weights within one class
    # "sub_process_class_factors": {
    #     "ttbb_dl_1b": 1,
    #     "ttbb_sl_1b": 1,
    #     "ttbb_fh_1b": 1,
    #     "tt_dl_nonb": 1,
    #     "tt_sl_nonb": 1,
    #     "tt_fh_nonb": 1,
    # },
    # "preparation_producer_name": "prepml_sr",
})
multiclass_hhh_v3 = DenseClassifierDL.derive("multiclass_hhh_v3", cls_dict={
    # "training_configs": ["c24v15"],
    "input_features": input_features["hhh_v0"],
    "processes": (
        "hhh_4b2w_2l2nu_c30_d40",
        "tthh_4b",
        "h",
        # "tt_dl_nonb",
        # "tt_sl_nonb",
        # "tt_fh_nonb",
        # "ttbb_dl_1b",
        # "ttbb_sl_1b",
        # "ttbb_fh_1b",
        "tt_custom",
        "ttbb_custom",
        "st",
        "dy",
        # "hh_ggf",
        # "hh_vbf",
        # "vhh_4b",
        # "vv",
        # "other",
    ),
    "train_nodes": {
        "hhh_4b2w_2l2nu_c30_d40": {"ml_id": 0, "label": r"HHH"},
        "tthh_4b": {"ml_id": 1, "label": r"HH"},
        "h": {"ml_id": 2},
        "tt_custom": {"ml_id": 3},
        "ttbb_custom": {"ml_id": 4},
        # "tt_nonb_custom": {
        #     "ml_id": 3,
        #     "label": r"tt",
        #     "color": "#000000",  # black
        #     "class_factor_mode": "xsec",
        #     "sub_processes": ["tt_dl_nonb", "tt_sl_nonb", "tt_fh_nonb"],
        # },
        # "ttbb_1b_custom": {
        #     "ml_id": 4,
        #     "label": r"tt + b",
        #     "color": "#000000",  # black
        #     "class_factor_mode": "xsec",
        #     "sub_processes": ["ttbb_dl_1b", "ttbb_sl_1b", "ttbb_fh_1b"],
        # },
        "st": {"ml_id": 5},
        "dy": {"ml_id": 6},
    },
    "class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "tthh_4b": 1,
        "tt_custom": 4,
        "ttbb_custom": 2,
        "st": 2,
        "dy": 2,
        "h": 1,
    },
    # # relative process weights within one class
    # "sub_process_class_factors": {
    #     "ttbb_dl_1b": 1,
    #     "ttbb_sl_1b": 1,
    #     "ttbb_fh_1b": 1,
    #     "tt_dl_nonb": 1,
    #     "tt_sl_nonb": 1,
    #     "tt_fh_nonb": 1,
    # },
    # "preparation_producer_name": "prepml_sr",
})
multiclass_hhh_v4 = DenseClassifierDL.derive("multiclass_hhh_v4", cls_dict={
    # "training_configs": ["c24v15"],
    "input_features": input_features["hhh_v0"],
    "processes": (
        "hhh_4b2w_2l2nu_c30_d40",
        "tthh_4b",
        "h",
        # "tt_dl_nonb",
        # "tt_sl_nonb",
        # "tt_fh_nonb",
        # "ttbb_dl_1b",
        # "ttbb_sl_1b",
        # "ttbb_fh_1b",
        "tt_custom",
        "ttbb_custom",
        "st",
        "dy",
        # "hh_ggf",
        # "hh_vbf",
        # "vhh_4b",
        # "vv",
        # "other",
    ),
    "train_nodes": {
        "hhh_4b2w_2l2nu_c30_d40": {"ml_id": 0, "label": r"HHH"},
        "tthh_4b": {"ml_id": 1, "label": r"HH"},
        "h": {"ml_id": 2},
        "tt_custom": {"ml_id": 3},
        "ttbb_custom": {"ml_id": 4},
        # "tt_nonb_custom": {
        #     "ml_id": 3,
        #     "label": r"tt",
        #     "color": "#000000",  # black
        #     "class_factor_mode": "xsec",
        #     "sub_processes": ["tt_dl_nonb", "tt_sl_nonb", "tt_fh_nonb"],
        # },
        # "ttbb_1b_custom": {
        #     "ml_id": 4,
        #     "label": r"tt + b",
        #     "color": "#000000",  # black
        #     "class_factor_mode": "xsec",
        #     "sub_processes": ["ttbb_dl_1b", "ttbb_sl_1b", "ttbb_fh_1b"],
        # },
        "st": {"ml_id": 5},
        "dy": {"ml_id": 6},
    },
    "class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "tthh_4b": 1,
        "tt_custom": 2,
        "ttbb_custom": 1,
        "st": 2,
        "dy": 2,
        "h": 1,
    },
})
multiclass_hhh_v5 = DenseClassifierDL.derive("multiclass_hhh_v5", cls_dict={
    # "training_configs": ["c24v15"],
    "input_features": input_features["hhh_v0"],
    "processes": (
        "hhh_4b2w_2l2nu_c30_d40",
        "tthh_4b",
        "h",
        # "tt_dl_nonb",
        # "tt_sl_nonb",
        # "tt_fh_nonb",
        # "ttbb_dl_1b",
        # "ttbb_sl_1b",
        # "ttbb_fh_1b",
        "tt_custom",
        "ttbb_custom",
        "st",
        "dy",
        # "hh_ggf",
        # "hh_vbf",
        # "vhh_4b",
        # "vv",
        # "other",
    ),
    "train_nodes": {
        "hhh_4b2w_2l2nu_c30_d40": {"ml_id": 0, "label": r"HHH"},
        "tthh_4b": {"ml_id": 1, "label": r"HH"},
        "h": {"ml_id": 2},
        "tt_custom": {"ml_id": 3},
        "ttbb_custom": {"ml_id": 4},
        # "tt_nonb_custom": {
        #     "ml_id": 3,
        #     "label": r"tt",
        #     "color": "#000000",  # black
        #     "class_factor_mode": "xsec",
        #     "sub_processes": ["tt_dl_nonb", "tt_sl_nonb", "tt_fh_nonb"],
        # },
        # "ttbb_1b_custom": {
        #     "ml_id": 4,
        #     "label": r"tt + b",
        #     "color": "#000000",  # black
        #     "class_factor_mode": "xsec",
        #     "sub_processes": ["ttbb_dl_1b", "ttbb_sl_1b", "ttbb_fh_1b"],
        # },
        "st": {"ml_id": 5},
        "dy": {"ml_id": 6},
    },
    "class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "tthh_4b": 1,
        "tt_custom": 1,
        "ttbb_custom": 1,
        "st": 2,
        "dy": 2,
        "h": 1,
    },
    # # relative process weights within one class
    # "sub_process_class_factors": {
    #     "ttbb_dl_1b": 1,
    #     "ttbb_sl_1b": 1,
    #     "ttbb_fh_1b": 1,
    #     "tt_dl_nonb": 1,
    #     "tt_sl_nonb": 1,
    #     "tt_fh_nonb": 1,
    # },
    # "preparation_producer_name": "prepml_sr",
})
multiclass_hhh_v6 = DenseClassifierDL.derive("multiclass_hhh_v6", cls_dict={
    # "training_configs": ["c24v15"],
    "input_features": input_features["hhh_v0"],
    "processes": (
        "hhh_4b2w_2l2nu_c30_d40",
        "tthh_4b",
        "h",
        # "tt_dl_nonb",
        # "tt_sl_nonb",
        # "tt_fh_nonb",
        # "ttbb_dl_1b",
        # "ttbb_sl_1b",
        # "ttbb_fh_1b",
        "tt_custom",
        "ttbb_custom",
        "st",
        "dy",
        # "hh_ggf",
        # "hh_vbf",
        # "vhh_4b",
        # "vv",
        # "other",
    ),
    "train_nodes": {
        "hhh_4b2w_2l2nu_c30_d40": {"ml_id": 0, "label": r"HHH"},
        "tthh_4b": {"ml_id": 1, "label": r"HH"},
        "h": {"ml_id": 2},
        "tt_custom": {"ml_id": 3},
        "ttbb_custom": {"ml_id": 4},
        # "tt_nonb_custom": {
        #     "ml_id": 3,
        #     "label": r"tt",
        #     "color": "#000000",  # black
        #     "class_factor_mode": "xsec",
        #     "sub_processes": ["tt_dl_nonb", "tt_sl_nonb", "tt_fh_nonb"],
        # },
        # "ttbb_1b_custom": {
        #     "ml_id": 4,
        #     "label": r"tt + b",
        #     "color": "#000000",  # black
        #     "class_factor_mode": "xsec",
        #     "sub_processes": ["ttbb_dl_1b", "ttbb_sl_1b", "ttbb_fh_1b"],
        # },
        "st": {"ml_id": 5},
        "dy": {"ml_id": 6},
    },
    "class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "tthh_4b": 1,
        "tt_custom": 1,
        "ttbb_custom": 1,
        "st": 1,
        "dy": 1,
        "h": 1,
    },
    # # relative process weights within one class
    # "sub_process_class_factors": {
    #     "ttbb_dl_1b": 1,
    #     "ttbb_sl_1b": 1,
    #     "ttbb_fh_1b": 1,
    #     "tt_dl_nonb": 1,
    #     "tt_sl_nonb": 1,
    #     "tt_fh_nonb": 1,
    # },
    # "preparation_producer_name": "prepml_sr",
})
multiclass_hhh_v7 = DenseClassifierDL.derive("multiclass_hhh_v7", cls_dict={
    # "training_configs": ["c24v15"],
    "input_features": input_features["hhh_v0"],
    "processes": (
        "hhh_4b2w_2l2nu_c30_d40",
        "tthh_4b",
        "h",
        # "tt_dl_nonb",
        # "tt_sl_nonb",
        # "tt_fh_nonb",
        # "ttbb_dl_1b",
        # "ttbb_sl_1b",
        # "ttbb_fh_1b",
        "tt_custom",
        "ttbb_custom",
        "st",
        "dy",
        # "hh_ggf",
        # "hh_vbf",
        # "vhh_4b",
        # "vv",
        # "other",
    ),
    "train_nodes": {
        "hhh_4b2w_2l2nu_c30_d40": {"ml_id": 0, "label": r"HHH"},
        "tthh_4b": {"ml_id": 1, "label": r"HH"},
        "h": {"ml_id": 2},
        # "tt_custom": {"ml_id": 3},
        # "ttbb_custom": {"ml_id": 4},
        "tt_ml": {
            "ml_id": 3,
            "label": r"tt",
            "color": "#000000",  # black
            "class_factor_mode": "xsec",
            "sub_processes": ["tt_custom", "ttbb_custom"],
        },
        # "ttbb_1b_custom": {
        #     "ml_id": 4,
        #     "label": r"tt + b",
        #     "color": "#000000",  # black
        #     "class_factor_mode": "xsec",
        #     "sub_processes": ["ttbb_dl_1b", "ttbb_sl_1b", "ttbb_fh_1b"],
        # },
        "st": {"ml_id": 4},
        "dy": {"ml_id": 5},
    },
    "class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "tthh_4b": 1,
        "tt_ml": 2,
        # "ttbb_custom": 1,
        "st": 1,
        "dy": 1,
        "h": 1,
    },
    # # relative process weights within one class
    # "sub_process_class_factors": {
    #     "ttbb_dl_1b": 1,
    #     "ttbb_sl_1b": 1,
    #     "ttbb_fh_1b": 1,
    #     "tt_dl_nonb": 1,
    #     "tt_sl_nonb": 1,
    #     "tt_fh_nonb": 1,
    # },
    # "preparation_producer_name": "prepml_sr",
})
multiclass_hhh_v8 = DenseClassifierDL.derive("multiclass_hhh_v8", cls_dict={
    # "training_configs": ["c24v15"],
    "input_features": input_features["hhh_v0"],
    "processes": (
        "hhh_4b2w_2l2nu_c30_d40",
        "tthh_4b",
        "h",
        # "tt_dl_nonb",
        # "tt_sl_nonb",
        # "tt_fh_nonb",
        # "ttbb_dl_1b",
        # "ttbb_sl_1b",
        # "ttbb_fh_1b",
        "tt_custom",
        "ttbb_custom",
        "st",
        "dy",
        # "hh_ggf",
        # "hh_vbf",
        # "vhh_4b",
        # "vv",
        # "other",
    ),
    "train_nodes": {
        "hhh_4b2w_2l2nu_c30_d40": {"ml_id": 0, "label": r"HHH"},
        "tthh_4b": {"ml_id": 1, "label": r"HH"},
        "h": {"ml_id": 2},
        # "tt_custom": {"ml_id": 3},
        # "ttbb_custom": {"ml_id": 4},
        "tt_ml": {
            "ml_id": 3,
            "label": r"tt",
            "color": "#000000",  # black
            "class_factor_mode": "xsec",
            "sub_processes": ["tt_custom", "ttbb_custom"],
        },
        # "ttbb_1b_custom": {
        #     "ml_id": 4,
        #     "label": r"tt + b",
        #     "color": "#000000",  # black
        #     "class_factor_mode": "xsec",
        #     "sub_processes": ["ttbb_dl_1b", "ttbb_sl_1b", "ttbb_fh_1b"],
        # },
        "st": {"ml_id": 4},
        "dy": {"ml_id": 5},
    },
    "class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "tthh_4b": 1,
        "tt_ml": 1,
        # "ttbb_custom": 1,
        "st": 1,
        "dy": 1,
        "h": 1,
    },
    # # relative process weights within one class
    # "sub_process_class_factors": {
    #     "ttbb_dl_1b": 1,
    #     "ttbb_sl_1b": 1,
    #     "ttbb_fh_1b": 1,
    #     "tt_dl_nonb": 1,
    #     "tt_sl_nonb": 1,
    #     "tt_fh_nonb": 1,
    # },
    # "preparation_producer_name": "prepml_sr",
})

multiclass_hhh_V3 = DenseClassifierDL.derive("multiclass_hhh_V3", cls_dict={
    # "training_configs": ["c24v15"],
    "input_features": input_features["hhh_v1"],
    "processes": (
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
        "tthh_4b",
        "hh_ggf",
        "hh_vbf",
        "h",
        "tt_custom",
        "ttbb_custom",
        "st",
        "dy",
        # "hh_ggf",
        # "hh_vbf",
        # "vhh_4b",
        # "vv",
        # "other",
    ),
    "train_nodes": {
        "hhh_signal": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": [
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
            ],
        },
        "hh_custom": {
            "ml_id": 1,
            "label": r"HH",
            "color": "#000000",  # black
            "class_factor_mode": "xsec",
            "sub_processes": [
                "tthh_4b",
                "hh_ggf",
                "hh_vbf",
            ],
        },
        "h": {"ml_id": 2},
        "tt_ml": {
            "ml_id": 3,
            "label": r"tt",
            "color": "#000000",  # black
            "class_factor_mode": "xsec",
            "sub_processes": ["tt_custom", "ttbb_custom"],
        },
        "st": {"ml_id": 4},
        "dy": {"ml_id": 5},
    },
    "class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "tthh_4b": 1,
        "tt_ml": 1,
        # "ttbb_custom": 1,
        "st": 1,
        "dy": 1,
        "h": 1,
    },
    # # relative process weights within one class
    # "sub_process_class_factors": {
    #     "ttbb_dl_1b": 1,
    #     "ttbb_sl_1b": 1,
    #     "ttbb_fh_1b": 1,
    #     "tt_dl_nonb": 1,
    #     "tt_sl_nonb": 1,
    #     "tt_fh_nonb": 1,
    # },
    # "preparation_producer_name": "prepml_sr",
})
multiclass_hhh_V4 = DenseClassifierDL.derive("multiclass_hhh_V4", cls_dict={
    # "training_configs": ["c24v15"],
    "input_features": input_features["hhh_v1"],
    "processes": (
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
        "tthh_4b",
        "tth",
        "tt_custom",
        "ttbb_custom",
        # "hh_ggf",
        # "hh_vbf",
        # "vhh_4b",
        # "vv",
        # "other",
    ),
    "train_nodes": {
        "hhh_signal": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": [
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
            ],
        },
        "tthh_4b": {"ml_id": 1, "label": r"ttHH"},
        "tth": {"ml_id": 2},
        "tt_ml": {
            "ml_id": 3,
            "label": r"tt",
            "color": "#000000",  # black
            "class_factor_mode": "xsec",
            "sub_processes": ["tt_custom", "ttbb_custom"],
        },
    },
    "class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "tthh_4b": 1,
        "tt_ml": 1,
        # "ttbb_custom": 1,
        "st": 1,
        "tth": 1,
    },
    # # relative process weights within one class
    # "sub_process_class_factors": {
    #     "ttbb_dl_1b": 1,
    #     "ttbb_sl_1b": 1,
    #     "ttbb_fh_1b": 1,
    #     "tt_dl_nonb": 1,
    #     "tt_sl_nonb": 1,
    #     "tt_fh_nonb": 1,
    # },
    # "preparation_producer_name": "prepml_sr",
})
multiclass_hhh_V5 = DenseClassifierDL.derive("multiclass_hhh_V5", cls_dict={
    # "training_configs": ["c24v15"],
    "input_features": input_features["hhh_v1"],
    "processes": (
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
        "tthh_4b",
        "tth",
        "tt_custom",
        "ttbb_custom",
        # "hh_ggf",
        # "hh_vbf",
        # "vhh_4b",
        # "vv",
        # "other",
    ),
    "train_nodes": {
        "hhh_signal": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": [
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
            ],
        },
        "tthh_4b": {"ml_id": 1, "label": r"ttHH"},
        "tth": {"ml_id": 2},
        "tt_ml": {
            "ml_id": 3,
            "label": r"tt",
            "color": "#000000",  # black
            "class_factor_mode": "xsec",
            "sub_processes": ["tt_custom", "ttbb_custom", "st"],
        },
    },
    "class_factors": {
        "hhh_4b2w_2l2nu_c30_d40": 1,
        "tthh_4b": 1,
        "tt_ml": 1,
        # "ttbb_custom": 1,
        "st": 1,
        "tth": 1,
    },
    # # relative process weights within one class
    # "sub_process_class_factors": {
    #     "ttbb_dl_1b": 1,
    #     "ttbb_sl_1b": 1,
    #     "ttbb_fh_1b": 1,
    #     "tt_dl_nonb": 1,
    #     "tt_sl_nonb": 1,
    #     "tt_fh_nonb": 1,
    # },
    "preparation_producer_name": "prepml_geq3b",
})

# hhh_v2 = DenseClassifierDL.derive("hhh_v2", cls_dict={
#     # "training_configs": ["c24v15"],
#     "input_features": input_features["hhh_v0"],
#     "processes": [
#         "hhh_4b2w_2l2nu_c30_d40",
#         # *processes.backgrounds_hhh,
#         *processes.backgrounds_hhh_v2,
#         # "tt_dl_nonb",
#         # "tt_sl_nonb",
#         # "tt_fh_nonb",
#         # "ttbb_dl_1b",
#         # "ttbb_sl_1b",
#         # "ttbb_fh_1b",
#     ],
#     "train_nodes": {
#         # "sig_hhh_binary": {
#         #     "ml_id": 0,
#         #     "label": r"HHH",
#         #     "color": "#000000",  # black
#         #     "class_factor_mode": "equal",
#         #     "sub_processes": processes.sig_hhh,
#         # },
#         "hhh_4b2w_2l2nu_c30_d40": {"ml_id": 0, "label": r"HHH"},
#         "bkg_binary_for_hhh": {
#             "ml_id": 1,
#             "label": "Background",
#             "color": "#e76300",  # Spanish Orange
#             "class_factor_mode": "xsec",
#             # "sub_processes": processes.backgrounds_hhh,
#             "sub_processes": processes.backgrounds_hhh_v2,
#         },
#     },
#     # relative class factors between different nodes
#     "class_factors": {
#         "hhh_4b2w_2l2nu_c30_d40": 1,
#         "bkg_binary_for_hhh": 1,
#     },
#     # relative process weights within one class
#     "sub_process_class_factors": {
#         "hhh_4b2w_2l2nu_c30_d40": 1,
#         "ttbb_custom": 1,
#         "tt_custom": 1,
#         # "tthh_4b": 4,
#         "st": 1,
#         "dy": 1,
#         "ttv": 1,
#         "h": 1,
#         # "vv": 2,
#         # "hh_ggf": 4,
#         # "hh_vbf": 4,
#         # "hh_other": 4,
#         # "other": 4,
#     },
#     "epochs": 100,
#     # "preparation_producer_name": "prepml_sr",
# })

hhh_V1 = DenseClassifierDL.derive("hhh_V1", cls_dict={
    # "training_configs": ["c24v15"],
    "input_features": input_features["hhh_v1"],
    "processes": [
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
        *processes.backgrounds_hhh_v2,
    ],
    "train_nodes": {
        "sig_hhh_binary": {
            "ml_id": 0,
            "label": r"HHH",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": [
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
            ],
        },
        "bkg_binary_for_hhh": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            # "sub_processes": processes.backgrounds_hhh,
            "sub_processes": processes.backgrounds_hhh_v2,
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
    # "preparation_producer_name": "prepml_sr",
})

# # hhh_sr
# multiclass_hhh_sr = multiclassv1.derive("multiclass_hhh_sr", cls_dict={
#     "input_features": input_features["hhh_v0"],
#     "preparation_producer_name": "prepml_hhh_sr",
# })
# ggf_hhh_sr = ggfv1.derive("ggf_hhh_sr", cls_dict={
#     "input_features": input_features["hhh_v0"],
#     "preparation_producer_name": "prepml_hhh_sr",
# })
# vbf_hhh_sr = vbfv1.derive("vbf_hhh_sr", cls_dict={
#     "input_features": input_features["hhh_v0"],
#     "preparation_producer_name": "prepml_hhh_sr",
# })

# multiclass_hhh_V2 = multiclass_hhh_V1.derive("multiclass_hhh_V2", cls_dict={"preparation_producer_name": "prepml_sr"})
# hhh_V2 = hhh_V1.derive("hhh_V2", cls_dict={"preparation_producer_name": "prepml_sr"})
