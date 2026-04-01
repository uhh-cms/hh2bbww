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
# from hbw.config.styling import color_palette


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


#
# default params
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
    "v2": [
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
        "mli_b_score_sum",
        "mli_ll_pt",
        "mli_ht",
        # "mli_mindr_lb",  # removed: correlated with mli_min_dr_llbb
        # "mli_dr_bb_llMET",  # using mli_dr_ll_bb instead
        "mli_j1_eta",
        "mli_n_jet",
        "mli_j1_pt",
    ],
})

# VBF feature sets
input_features["vbfmqq"] = input_features["v2"] + ["mli_full_vbf_mass"]
input_features["vbftag"] = input_features["v2"] + ["mli_full_vbf_tag"]

class_factors = {
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


class DenseClassifierBBZZ(DenseModelMixin, ModelFitMixin, MLClassifierBase):
    _default__processes: tuple = (
        "hh_ggf_kl0_kt1",
        "hh_ggf_kl1_kt1",
        "hh_ggf_kl2p45_kt1",
        "hh_ggf_kl5_kt1",
        "hh_vbf_kv1_k2v1_kl1",
        "hh_vbf_kv1_k2v0_kl1",
        "hh_vbf_kvm0p962_k2v0p959_klm1p43",
        "hh_vbf_kvm1p21_k2v1p94_klm0p94",
        "hh_vbf_kvm1p6_k2v2p72_klm1p36",
        "hh_vbf_kvm1p83_k2v3p57_klm3p39",
        "tt",
        "st",
        "dy",
        "h",
    )
    train_nodes: dict = {
        "sig_ggf": {
            "ml_id": 0,
            "label": r"HH_{ggF}",
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
        "hh_ggf_kl0_kt1": 1,
        "hh_ggf_kl1_kt1": 1,
        "hh_ggf_kl2p45_kt1": 1,
        "hh_ggf_kl5_kt1": 1,
    }

    input_features = [
        "mli_dr_ll_bb",
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
        "mli_b_score_sum",
        "mli_ll_pt",
        "mli_ht",
        "mli_j1_eta",
        "mli_n_jet",
        "mli_j1_pt",
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
# derived bbZZ DNNs
#

multiclass_dycr = DenseClassifierBBZZ.derive("multiclass_dycr", cls_dict={
    "training_configs": configs.full,
    "input_features": input_features["v2"],
    "preparation_producer_name": "prepml_dycr",
    "processes": (
        *processes.ggf_hh,
        *processes.vbf_hh,
        "tt",
        "st",
        "dy",
        "h",
    ),
    "train_nodes": {
        "sig_ggf": {
            "ml_id": 0,
            "label": r"HH_{ggF}",
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
        "dy": {"ml_id": 4},
        "h": {"ml_id": 5},
    },
})
ggf_dycr = DenseClassifierBBZZ.derive("ggf_dycr", cls_dict={
    "training_configs": configs.full,
    "input_features": input_features["v2"],
    "preparation_producer_name": "prepml_dycr",
    "processes": [
        *processes.ggf_hh,
        "tt", "st", "dy",
        "vv", "ttv", "h", "other",
    ],
    "train_nodes": {
        "sig_ggf_binary": {
            "ml_id": 0,
            "label": r"HH_{ggF}",
            "color": "#000000",
            "class_factor_mode": "equal",
            "sub_processes": processes.ggf_hh,
        },
        "bkg_binary_for_ggf": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": [
                "tt", "st", "dy",
                "vv", "ttv", "h", "other",
            ],
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
        "dy": 1,
        "vv": 2,
        "ttv": 2,
        "h": 2,
        "other": 8,
    },
    "epochs": 100,
})
vbf_dycr = DenseClassifierBBZZ.derive("vbf_dycr", cls_dict={
    "training_configs": configs.full,
    "input_features": input_features["vbftag"],
    "preparation_producer_name": "prepml_dycr",
    "processes": [
        *processes.vbf_hh,
        "tt", "st", "dy",
        "vv", "ttv", "h", "other",
    ],
    "train_nodes": {
        "sig_vbf_binary": {
            "ml_id": 0,
            "label": r"HH_{VBF}",
            "color": "#000000",
            "class_factor_mode": "equal",
            "sub_processes": processes.vbf_hh,
        },
        "bkg_binary_for_vbf": {
            "ml_id": 1,
            "label": "Background",
            "color": "#e76300",  # Spanish Orange
            "class_factor_mode": "xsec",
            "sub_processes": [
                "tt", "st", "dy",
                "vv", "ttv", "h", "other",
            ],
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
        "dy": 1,
        "vv": 2,
        "ttv": 2,
        "h": 2,
        "other": 8,
    },
    "epochs": 100,
})
