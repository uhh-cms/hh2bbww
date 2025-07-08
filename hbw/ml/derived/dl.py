# coding: utf-8

"""
ML models using the MLClassifierBase and Mixins
"""

from __future__ import annotations

from columnflow.types import Union

import law

from columnflow.util import maybe_import, DotDict

from hbw.ml.base import MLClassifierBase
from hbw.ml.mixins import DenseModelMixin, ModelFitMixin
from hbw.config.styling import color_palette


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


class DenseClassifierDL(DenseModelMixin, ModelFitMixin, MLClassifierBase):
    _default__processes: tuple = (
        "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
        "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
        "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
        "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1",
        "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43",
        "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94",
        "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36",
        "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39",
        "tt",
        "st",
        "dy",
        "h",
    )
    train_nodes: dict = {
        "sig_ggf": {
            "ml_id": 0,
            "label": r"HH_{GGF}",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": (
                "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
            ),
        },
        "sig_vbf": {
            "ml_id": 1,
            "label": r"HH_{VBF}",
            "color": "#999999",  # grey
            "class_factor_mode": "equal",
            "sub_processes": (
                "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
                "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1",
                "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43",
                "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94",
                "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36",
                "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39",
            ),
        },
        "tt": {"ml_id": 2},
        "st": {"ml_id": 3},
        "dy": {"ml_id": 4},
        "h": {"ml_id": 5},
    }
    _default__class_factors: dict = {
        "sig_ggf": 1,
        "sig_vbf": 1,
        "tt": 1,
        "st": 1,
        "dy": 1,
        "h": 1,
    }

    _default__sub_process_class_factors = {
        "hh_ggf_hbb_hvv2l2nu_kl0_kt1": 1,
        "hh_ggf_hbb_hvv2l2nu_kl1_kt1": 1,
        "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1": 1,
        "hh_ggf_hbb_hvv2l2nu_kl5_kt1": 1,
    }

    input_features = [
        # event features
        "mli_ht", "mli_n_jet", "mli_n_btag",
        "mli_b_score_sum",
        "mli_mindr_jj", "mli_maxdr_jj",
        # bb system
        "mli_deta_bb", "mli_dphi_bb", "mli_mbb", "mli_bb_pt",
        "mli_mindr_lb",
        # ll system
        "mli_mll", "mli_dphi_ll", "mli_deta_ll", "mli_ll_pt",
        # HH system
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
    ]

    preparation_producer_name = "prepml"

    folds: int = 5
    negative_weights: str = "ignore"

    # overwriting DenseModelMixin parameters
    _default__activation: str = "relu"
    _default__layers: tuple = (512, 512, 512)
    _default__dropout: float = 0.20
    _default__learningrate: float = 0.00050

    # overwriting ModelFitMixin parameters
    _default__callbacks: set = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True
    _default__reduce_lr_factor: float = 0.8
    _default__reduce_lr_patience: int = 3
    _default__epochs: int = 100
    _default__batchsize: int = 2 ** 12
    steps_per_epoch: Union[int, str] = "iter_smallest_process"

    # parameters to add into the `parameters` attribute to determine the 'parameters_repr' and to store in a yaml file
    bookkeep_params: set[str] = {
        # base params
        "data_loader", "input_features", "train_val_test_split",
        "processes", "sub_process_class_factors", "class_factors", "train_nodes",
        "negative_weights", "folds",
        # DenseModelMixin
        "activation", "layers", "dropout", "learningrate",
        # ModelFitMixin
        "callbacks", "reduce_lr_factor", "reduce_lr_patience",
        "epochs", "batchsize",
    }

    # parameters that can be overwritten via command line
    settings_parameters: set[str] = {
        # base params
        "processes", "class_factors", "sub_process_class_factors",
        # DenseModelMixin
        "activation", "layers", "dropout", "learningrate",
        # ModelFitMixin
        "callbacks", "reduce_lr_factor", "reduce_lr_patience",
        "epochs", "batchsize",
    }

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def cast_ml_param_values(self):
        super().cast_ml_param_values()

    def setup(self) -> None:
        super().setup()


#
# configs
#

hh_train_procs_ggf = lambda hhdecay: [
    f"hh_ggf{hhdecay}_kl0_kt1",
    f"hh_ggf{hhdecay}_kl1_kt1",
    f"hh_ggf{hhdecay}_kl2p45_kt1",
    f"hh_ggf{hhdecay}_kl5_kt1",
]
hh_train_procs_vbf = lambda hhdecay: [
    f"hh_vbf{hhdecay}_kv1_k2v1_kl1",
    f"hh_vbf{hhdecay}_kv1_k2v0_kl1",
    f"hh_vbf{hhdecay}_kvm0p962_k2v0p959_klm1p43",
    f"hh_vbf{hhdecay}_kvm1p21_k2v1p94_klm0p94",
    f"hh_vbf{hhdecay}_kvm1p6_k2v2p72_klm1p36",
    f"hh_vbf{hhdecay}_kvm1p83_k2v3p57_klm3p39",
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
    "ggf_hbb_hvv2l2nu": hh_train_procs_ggf("_hbb_hvv2l2nu"),
    "vbf_hbb_hvv2l2nu": hh_train_procs_vbf("_hbb_hvv2l2nu"),
    "ggf_hh": hh_train_procs_ggf(""),
    "vbf_hh": hh_train_procs_vbf(""),
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
    "sorted_previous": [
        "mli_mbb",
        "mli_b1_pt",
        "mli_j1_pt",
        "mli_n_jet",
        "mli_mbbllMET",
        "mli_dr_bb_llMET",
        "mli_j1_eta",
        "mli_met_pt",
        "mli_mllMET",
        "mli_mll",
        "mli_ll_pt",
        "mli_lep_pt",
        "mli_lep2_pt",
        "mli_dphi_bb_nu",
        "mli_j1_b_score",
        "mli_bb_pt",
        "mli_dr_ll",
        "mli_b_score_sum",
        "mli_min_dr_llbb",
        "mli_b2_pt",
        "mli_lep_eta",
        "mli_b1_eta",
        "mli_dr_bb",
        "mli_mindr_lb",
        "mli_ht",
        "mli_b2_eta",
        "mli_b2_b_score",
        "mli_n_btag",
        "mli_lep2_eta",
        "mli_dhpi_bb",
        "mli_b1_b_score",
        "mli_dhpi_bb_llMET",
        "mli_dphi_ll",
        "mli_vbf_mass",
        "mli_vbf_tag",
        "mli_vbf_deta",
    ],
    "reduced": [
        "mli_mbb",
        "mli_b1_pt",
        "mli_j1_pt",
        "mli_n_jet",
        "mli_mbbllMET",
        "mli_dr_bb_llMET",
        "mli_j1_eta",
        "mli_met_pt",
        "mli_mllMET",
        "mli_mll",
        "mli_ll_pt",
        "mli_lep_pt",
        "mli_lep2_pt",
        "mli_dphi_bb_nu",  # badly modelled ---> please remove in future
        "mli_j1_b_score",
        "mli_bb_pt",
        "mli_dr_ll",
        "mli_b_score_sum",
        "mli_min_dr_llbb",
        "mli_b2_pt",
        # "mli_lep_eta",
        # "mli_b1_eta",
        "mli_dr_bb",
        "mli_mindr_lb",
        "mli_ht",
        # "mli_b2_eta",
        # "mli_b2_b_score",
        # "mli_n_btag",
        # "mli_lep2_eta",
        # "mli_dhpi_bb",
        # "mli_b1_b_score",
        # "mli_dhpi_bb_llMET",
        # "mli_dphi_ll",
        "mli_vbf_mass",  # important for vbf? ---> not really
        "mli_vbf_tag",  # important for vbf? ---> not really
        "mli_vbf_deta",  # important for vbf? ---> not really
    ],
    "v0": [  # reduced + mixed channel
        "mli_mbb",
        "mli_b1_pt",
        "mli_j1_pt",
        "mli_n_jet",
        "mli_mbbllMET",
        "mli_dr_bb_llMET",
        "mli_j1_eta",
        "mli_met_pt",
        "mli_mllMET",
        "mli_mll",
        "mli_ll_pt",
        "mli_lep_pt",
        "mli_lep2_pt",
        "mli_dphi_bb_nu",  # badly modelled ---> please remove in future
        "mli_j1_b_score",
        "mli_bb_pt",
        "mli_dr_ll",
        "mli_b_score_sum",
        "mli_min_dr_llbb",
        "mli_b2_pt",
        "mli_dr_bb",
        "mli_mindr_lb",
        "mli_ht",
        "mli_vbf_mass",  # important for vbf? --> not really
        "mli_vbf_tag",  # important for vbf? --> not really
        "mli_vbf_deta",  # important for vbf? --> not really
        "mli_mixed_channel",
    ],
    "v1": [
        # dphi_bb_nu, vbf_pair_eta, vbf_pair_mass, mindr_lj
        # new input features
        "mli_maxdr_jj",
        # v0 sorted from GGF SHAP
        "mli_mbbllMET",
        "mli_mbb",
        "mli_mll",
        "mli_b1_pt",
        "mli_bb_pt",
        "mli_mllMET",
        "mli_lep_pt",
        "mli_mixed_channel",
        # mli_dphi_bb_nu: removed due to bad modelling
        "mli_dr_bb_llMET",
        "mli_lep2_pt",
        "mli_b2_pt",
        "mli_met_pt",
        "mli_b_score_sum",
        "mli_ll_pt",
        "mli_min_dr_llbb",
        "mli_ht",
        "mli_j1_pt",
        "mli_dr_ll",
        # mli_j1_b_score: removed due to low stats and low importance
        "mli_dr_bb",
        "mli_mindr_lb",
        "mli_n_jet",
        "mli_j1_eta",
        # vbf_tag, vbf_deta and vbf_mass removed due to bad modelling and low importance
    ],
    "fatjet": [
        "mli_fj_particleNetWithMass_HbbvsQCD",
        # "mli_fj_particleNet_XbbvsQCD",
        "mli_fj_pt",
        "mli_fj_eta",
        "mli_fj_phi",
        # "mli_fj_mass",
        # "mli_fj_msoftdrop",
    ],
})
input_features["fatjet_v1"] = input_features["v1"] + input_features["fatjet"]

class_factors = {
    "default": DenseClassifierDL._default__class_factors,
    "ones": {},  # defaults to 1 (NOTE: do not try to use defaultdict! does not work with hash generation)
    "benchmark": {
        "sig_ggf": 1,
        "sig_vbf": 1,
        "tt": 8,
        "st": 2,
        "dy": 2,
        "h": 1,
    },
}

configs = DotDict({
    "22post": lambda self, requested_configs: ["c22postv14"],
    "22": lambda self, requested_configs: ["c22prev14", "c22postv14"],
    "23": lambda self, requested_configs: ["c23prev14", "c23postv14"],
    "full": lambda self, requested_configs: ["c22prev14", "c22postv14", "c23prev14", "c23postv14"],
})

#
# derived MLModels
#

# multiclass = DenseClassifierDL.derive("multiclass", cls_dict={
#     "training_configs": configs.full,
#     "input_features": input_features["v0"],
#     "class_factors": class_factors["ones"],
# })
# dl_22post_test = DenseClassifierDL.derive("dl_22post_test", cls_dict={
#     "training_configs": lambda self, requested_configs: ["c22post"],
#     "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "st_tchannel_t"],
#     "epochs": 20,
# })
# dl_22post_limited = DenseClassifierDL.derive("dl_22post_limited", cls_dict={
#     "training_configs": lambda self, requested_configs: ["l22post"],
#     "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "st_tchannel_t"],
#     "epochs": 6,
# })

# # TODO: include minor processes in training: VVV, ttVV, tttt, w_lnu

# ggf = DenseClassifierDL.derive("ggf", cls_dict={
#     "training_configs": configs.full,
#     "input_features": input_features["v0"],
#     "processes": [
#         "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
#         "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
#         "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
#         "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
#         "tt",
#         "st",
#         "dy_m4to10",
#         "dy_m10to50",
#         "dy_m50toinf",
#         "vv",
#         "ttv",
#         "h",
#     ],
#     "train_nodes": {
#         "sig_ggf_binary": {
#             "ml_id": 0,
#             "label": "Signal",
#             "color": "#000000",
#             "class_factor_mode": "equal",
#             "sub_processes": (
#                 "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
#                 "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
#                 "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
#                 "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
#             ),
#         },
#         "bkg_binary_for_ggf": {
#             "ml_id": 1,
#             "label": "Background",
#             "color": "#e76300",  # Spanish Orange
#             "class_factor_mode": "xsec",
#             "sub_processes": (
#                 "tt",
#                 "st",
#                 "dy_m4to10",
#                 "dy_m10to50",
#                 "dy_m50toinf",
#                 "vv",
#                 "ttv",
#                 "h",
#             ),
#         },
#     },
#     # relative class factors between different nodes
#     "class_factors": {
#         "sig_ggf_binary": 1,
#         "bkg_binary_for_ggf": 1,
#     },
#     # relative process weights within one class
#     "sub_process_class_factors": {
#         "hh_ggf_hbb_hvv2l2nu_kl0_kt1": 1,
#         "hh_ggf_hbb_hvv2l2nu_kl1_kt1": 1,
#         "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1": 1,
#         "hh_ggf_hbb_hvv2l2nu_kl5_kt1": 1,
#         "tt": 1,
#         "st": 1,
#         "dy_m4to10": 0.1,  # assign small weight due to too low statistics for training
#         "dy_m10to50": 1,
#         "dy_m50toinf": 1,
#         "vv": 4,  # assign larger weight due similarity to signal & low MC stats.
#         "ttv": 2,
#         "h": 2,
#     },
#     "epochs": 100,
# })
# vbf = DenseClassifierDL.derive("vbf", cls_dict={
#     "training_configs": configs.full,
#     "input_features": input_features["v0"],
#     "processes": [
#         "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
#         "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1",
#         "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43",
#         "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94",
#         "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36",
#         "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39",
#         "tt",
#         "st",
#         "dy_m4to10",
#         "dy_m10to50",
#         "dy_m50toinf",
#         "vv",
#         "ttv",
#         "h",
#     ],
#     "train_nodes": {
#         "sig_vbf_binary": {
#             "ml_id": 0,
#             "label": "HH VBF",
#             "color": "#000000",
#             "class_factor_mode": "equal",
#             "sub_processes": (
#                 "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
#                 "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1",
#                 "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43",
#                 "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94",
#                 "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36",
#                 "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39",
#             ),
#         },
#         "bkg_binary_for_vbf": {
#             "ml_id": 1,
#             "label": "Background",
#             "color": "#e76300",  # Spanish Orange
#             "class_factor_mode": "xsec",
#             "sub_processes": (
#                 "tt",
#                 "st",
#                 "dy_m4to10",
#                 "dy_m10to50",
#                 "dy_m50toinf",
#                 "vv",
#                 "ttv",
#                 "h",
#             ),
#         },
#     },
#     "class_factors": {
#         "sig_vbf_binary": 1,
#         "bkg_binary_for_vbf": 1,
#     },
#     # relative process weights within one class
#     "sub_process_class_factors": {
#         "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": 1,
#         "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1": 1,
#         "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43": 1,
#         "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94": 1,
#         "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36": 1,
#         "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39": 1,
#         "tt": 1,
#         "st": 1,
#         "dy_m4to10": 0.1,  # assign small weight due to low statistics
#         "dy_m10to50": 1,
#         "dy_m50toinf": 1,
#         "vv": 4,  # assign larger weight due similarity to signal & low MC stats.
#         "ttv": 2,
#         "h": 2,
#     },
#     "epochs": 100,
# })


multiclassv1 = DenseClassifierDL.derive("multiclassv1", cls_dict={
    "training_configs": configs.full,
    "input_features": input_features["v1"],
    "processes": (
        *processes.ggf_hbb_hvv2l2nu,
        *processes.vbf_hbb_hvv2l2nu,
        "tt",
        "st",
        "dy_m10to50",
        "dy_m50toinf",
        "h",
    ),
    "train_nodes": {
        "sig_ggf": {
            "ml_id": 0,
            "label": r"HH_{GGF}",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": processes.ggf_hbb_hvv2l2nu,
        },
        "sig_vbf": {
            "ml_id": 1,
            "label": r"HH_{VBF}",
            "color": "#999999",  # grey
            "class_factor_mode": "equal",
            "sub_processes": processes.vbf_hbb_hvv2l2nu,
        },
        "tt": {"ml_id": 2},
        "st": {"ml_id": 3},
        "dy_m10toinf": {
            "ml_id": 4,
            "sub_processes": ["dy_m10to50", "dy_m50toinf"],
            "label": "DY (m>10)",
            "color": color_palette["yellow"],
            "class_factor_mode": "xsec",
        },
        "h": {"ml_id": 5},
    },
})
ggfv1 = DenseClassifierDL.derive("ggfv1", cls_dict={
    "training_configs": configs.full,
    "input_features": input_features["v1"],
    "processes": [
        *processes.ggf_hbb_hvv2l2nu,
        *processes.backgrounds_binary,
    ],
    "train_nodes": {
        "sig_ggf_binary": {
            "ml_id": 0,
            "label": "Signal",
            "color": "#000000",
            "class_factor_mode": "equal",
            "sub_processes": processes.ggf_hbb_hvv2l2nu,
        },
        "bkg_binary_for_ggf": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": processes.backgrounds_binary,
        },
    },
    # relative class factors between different nodes
    "class_factors": {
        "sig_ggf_binary": 1,
        "bkg_binary_for_ggf": 1,
    },
    # relative process weights within one class
    "sub_process_class_factors": {
        "hh_ggf_hbb_hvv2l2nu_kl0_kt1": 1,
        "hh_ggf_hbb_hvv2l2nu_kl1_kt1": 1,
        "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1": 1,
        "hh_ggf_hbb_hvv2l2nu_kl5_kt1": 1,
        "tt": 1,
        "st": 1,
        "dy_m10to50": 1,
        "dy_m50toinf": 1,
        "vv": 2,
        "ttv": 2,
        "h": 2,
        "other": 8,
    },
    "epochs": 100,
})
vbfv1 = DenseClassifierDL.derive("vbfv1", cls_dict={
    "training_configs": configs.full,
    "input_features": input_features["v1"],
    "processes": [
        *processes.vbf_hbb_hvv2l2nu,
        *processes.backgrounds_binary,
    ],
    "train_nodes": {
        "sig_vbf_binary": {
            "ml_id": 0,
            "label": "HH VBF",
            "color": "#000000",
            "class_factor_mode": "equal",
            "sub_processes": processes.vbf_hbb_hvv2l2nu,
        },
        "bkg_binary_for_vbf": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": processes.backgrounds_binary,
        },
    },
    "class_factors": {
        "sig_vbf_binary": 1,
        "bkg_binary_for_vbf": 1,
    },
    # relative process weights within one class
    "sub_process_class_factors": {
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": 1,
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1": 1,
        "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43": 1,
        "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94": 1,
        "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36": 1,
        "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39": 1,
        "tt": 1,
        "st": 1,
        "dy_m10to50": 1,
        "dy_m50toinf": 1,
        "vv": 2,
        "ttv": 2,
        "h": 2,
        "other": 8,
    },
    "epochs": 100,
})


vbfv1_2b = vbfv1.derive("vbfv1_2b", cls_dict={"preparation_producer_name": "prepml_2b"})
ggfv1_2b = ggfv1.derive("ggfv1_2b", cls_dict={"preparation_producer_name": "prepml_2b"})

multiclass_fatjetv1 = multiclassv1.derive("multiclass_fatjetv1", cls_dict={"input_features": input_features.fatjet_v1})
ggf_fatjetv1 = ggfv1.derive("ggf_fatjetv1", cls_dict={"input_features": input_features.fatjet_v1})
vbf_fatjetv1 = vbfv1.derive("vbf_fatjetv1", cls_dict={"input_features": input_features.fatjet_v1})


#
# adding bbtautau
#
multiclass_allsig = DenseClassifierDL.derive("multiclass_allsig", cls_dict={
    "training_configs": configs.full,
    "input_features": input_features["v1"],
    "processes": (
        *processes.ggf_hh,
        *processes.vbf_hh,
        "tt",
        "st",
        "dy_m10to50",
        "dy_m50toinf",
        "h",
    ),
    "train_nodes": {
        "sig_ggf": {
            "ml_id": 0,
            "label": r"HH_{GGF}",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": processes.ggf_hh,
        },
        "sig_vbf": {
            "ml_id": 1,
            "label": r"HH_{VBF}",
            "color": "#999999",  # grey
            "class_factor_mode": "equal",
            "sub_processes": processes.vbf_hh,
        },
        "tt": {"ml_id": 2},
        "st": {"ml_id": 3},
        "dy_m10toinf": {
            "ml_id": 4,
            "sub_processes": ["dy_m10to50", "dy_m50toinf"],
            "label": "DY (m>10)",
            "color": color_palette["yellow"],
            "class_factor_mode": "xsec",
        },
        "h": {"ml_id": 5},
    },
})
ggf_allsig = DenseClassifierDL.derive("ggf_allsig", cls_dict={
    "training_configs": configs.full,
    "input_features": input_features["v1"],
    "processes": [
        *processes.ggf_hh,
        *processes.backgrounds_binary,
    ],
    "train_nodes": {
        "sig_ggf_binary": {
            "ml_id": 0,
            "label": "Signal",
            "color": "#000000",
            "class_factor_mode": "equal",
            "sub_processes": processes.ggf_hh,
        },
        "bkg_binary_for_ggf": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": processes.backgrounds_binary,
        },
    },
    # relative class factors between different nodes
    "class_factors": {
        "sig_ggf_binary": 1,
        "bkg_binary_for_ggf": 1,
    },
    # relative process weights within one class
    "sub_process_class_factors": {
        "hh_ggf_kl0_kt1": 1,
        "hh_ggf_kl1_kt1": 1,
        "hh_ggf_kl2p45_kt1": 1,
        "hh_ggf_kl5_kt1": 1,
        "tt": 1,
        "st": 1,
        "dy_m10to50": 1,
        "dy_m50toinf": 1,
        "vv": 2,
        "ttv": 2,
        "h": 2,
        "other": 8,
    },
    "epochs": 100,
})
vbf_allsig = DenseClassifierDL.derive("vbf_allsig", cls_dict={
    "training_configs": configs.full,
    "input_features": input_features["v1"],
    "processes": [
        *processes.vbf_hh,
        *processes.backgrounds_binary,
    ],
    "train_nodes": {
        "sig_vbf_binary": {
            "ml_id": 0,
            "label": "HH VBF",
            "color": "#000000",
            "class_factor_mode": "equal",
            "sub_processes": processes.vbf_hh,
        },
        "bkg_binary_for_vbf": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": processes.backgrounds_binary,
        },
    },
    "class_factors": {
        "sig_vbf_binary": 1,
        "bkg_binary_for_vbf": 1,
    },
    # relative process weights within one class
    "sub_process_class_factors": {
        "hh_vbf_kv1_k2v1_kl1": 1,
        "hh_vbf_kv1_k2v0_kl1": 1,
        "hh_vbf_kvm0p962_k2v0p959_klm1p43": 1,
        "hh_vbf_kvm1p21_k2v1p94_klm0p94": 1,
        "hh_vbf_kvm1p6_k2v2p72_klm1p36": 1,
        "hh_vbf_kvm1p83_k2v3p57_klm3p39": 1,
        "tt": 1,
        "st": 1,
        "dy_m10to50": 1,
        "dy_m50toinf": 1,
        "vv": 2,
        "ttv": 2,
        "h": 2,
        "other": 8,
    },
    "epochs": 100,
})


#
# only kl1
#

multiclass_kl1 = DenseClassifierDL.derive("multiclass_kl1", cls_dict={
    "training_configs": configs.full,
    "input_features": input_features["v1"],
    "processes": (
        "hh_ggf_kl1_kt1",
        "hh_vbf_kv1_k2v1_kl1",
        "tt",
        "st",
        "dy_m10to50",
        "dy_m50toinf",
        "h",
    ),
    "train_nodes": {
        "hh_ggf_kl1_kt1": {"ml_id": 0},
        "hh_vbf_kv1_k2v1_kl1": {"ml_id": 1},
        "tt": {"ml_id": 2},
        "st": {"ml_id": 3},
        "dy_m10toinf": {
            "ml_id": 4,
            "sub_processes": ["dy_m10to50", "dy_m50toinf"],
            "label": "DY (m>10)",
            "color": color_palette["yellow"],
            "class_factor_mode": "xsec",
        },
        "h": {"ml_id": 5},
    },
})
ggf_kl1 = DenseClassifierDL.derive("ggf_kl1", cls_dict={
    "training_configs": configs.full,
    "input_features": input_features["v1"],
    "processes": [
        "hh_ggf_kl1_kt1",
        *processes.backgrounds_binary,
    ],
    "train_nodes": {
        "hh_ggf_kl1_kt1": {"ml_id": 0},
        "bkg_binary_for_ggf": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": processes.backgrounds_binary,
        },
    },
    # relative class factors between different nodes
    "class_factors": {
        "hh_ggf_kl1_kt1": 1,
        "bkg_binary_for_ggf": 1,
    },
    # relative process weights within one class
    "sub_process_class_factors": {
        "hh_ggf_kl0_kt1": 1,
        "hh_ggf_kl1_kt1": 1,
        "hh_ggf_kl2p45_kt1": 1,
        "hh_ggf_kl5_kt1": 1,
        "tt": 1,
        "st": 1,
        "dy_m10to50": 1,
        "dy_m50toinf": 1,
        "vv": 2,
        "ttv": 2,
        "h": 2,
        "other": 8,
    },
    "epochs": 100,
})
vbf_kl1 = DenseClassifierDL.derive("vbf_kl1", cls_dict={
    "training_configs": configs.full,
    "input_features": input_features["v1"],
    "processes": [
        "hh_vbf_kv1_k2v1_kl1",
        *processes.backgrounds_binary,
    ],
    "train_nodes": {
        "hh_vbf_kv1_k2v1_kl1": {"ml_id": 0},
        "bkg_binary_for_vbf": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": processes.backgrounds_binary,
        },
    },
    "class_factors": {
        "hh_vbf_kv1_k2v1_kl1": 1,
        "bkg_binary_for_vbf": 1,
    },
    # relative process weights within one class
    "sub_process_class_factors": {
        "hh_vbf_kv1_k2v1_kl1": 1,
        "hh_vbf_kv1_k2v0_kl1": 1,
        "hh_vbf_kvm0p962_k2v0p959_klm1p43": 1,
        "hh_vbf_kvm1p21_k2v1p94_klm0p94": 1,
        "hh_vbf_kvm1p6_k2v2p72_klm1p36": 1,
        "hh_vbf_kvm1p83_k2v3p57_klm3p39": 1,
        "tt": 1,
        "st": 1,
        "dy_m10to50": 1,
        "dy_m50toinf": 1,
        "vv": 2,
        "ttv": 2,
        "h": 2,
        "other": 8,
    },
    "epochs": 100,
})
