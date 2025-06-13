# coding: utf-8

"""
ML models using the MLClassifierBase and Mixins
"""

from __future__ import annotations

from columnflow.types import Union

import law

from columnflow.util import maybe_import

from hbw.ml.base import MLClassifierBase
from hbw.ml.mixins import DenseModelMixin, ModelFitMixin


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


class DenseClassifierSL(DenseModelMixin, ModelFitMixin, MLClassifierBase):

    combine_processes = ()

    _default__processes: tuple = (
        "hh_ggf_hbb_hvvqqlnu_kl0_kt1",
        "hh_ggf_hbb_hvvqqlnu_kl1_kt1",
        "hh_ggf_hbb_hvvqqlnu_kl2p45_kt1",
        "hh_ggf_hbb_hvvqqlnu_kl5_kt1",
        "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1",
        "hh_vbf_hbb_hvvqqlnu_kv1_k2v0_kl1",
        "hh_vbf_hbb_hvvqqlnu_kvm0p962_k2v0p959_klm1p43",
        "hh_vbf_hbb_hvvqqlnu_kvm1p21_k2v1p94_klm0p94",
        "hh_vbf_hbb_hvvqqlnu_kvm1p6_k2v2p72_klm1p36",
        "hh_vbf_hbb_hvvqqlnu_kvm1p83_k2v3p57_klm3p39",
        "tt",
        "st",
        # "v_lep",
        "w_lnu",
        # "dy",
        # "h",
    )
    train_nodes: dict = {
        "sig_ggf": {
            "ml_id": 0,
            "label": r"HH_{GGF}",
            "color": "#000000",  # black
            "class_factor_mode": "equal",
            "sub_processes": (
                "hh_ggf_hbb_hvvqqlnu_kl0_kt1",
                "hh_ggf_hbb_hvvqqlnu_kl1_kt1",
                "hh_ggf_hbb_hvvqqlnu_kl2p45_kt1",
                "hh_ggf_hbb_hvvqqlnu_kl5_kt1",
            ),
        },
        "sig_vbf": {
            "ml_id": 1,
            "label": r"HH_{VBF}",
            "color": "#999999",  # grey
            "class_factor_mode": "equal",
            "sub_processes": (
                "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1",
                "hh_vbf_hbb_hvvqqlnu_kv1_k2v0_kl1",
                "hh_vbf_hbb_hvvqqlnu_kvm0p962_k2v0p959_klm1p43",
                "hh_vbf_hbb_hvvqqlnu_kvm1p21_k2v1p94_klm0p94",
                "hh_vbf_hbb_hvvqqlnu_kvm1p6_k2v2p72_klm1p36",
                "hh_vbf_hbb_hvvqqlnu_kvm1p83_k2v3p57_klm3p39",
            ),
        },
        "tt": {"ml_id": 2},
        "st": {"ml_id": 3},
        "w_lnu": {"ml_id": 4},
        # "dy": {"ml_id": 4},
        # "h": {"ml_id": 5},
    }
    _default__class_factors: dict = {
        "sig_ggf": 1,
        "sig_vbf": 1,
        "tt": 1,
        "st": 1,
        "w_lnu": 1,
        # "dy": 1,
        # "h": 1,
    }

    _default__sub_process_class_factors = {
        "hh_ggf_hbb_hvvqqlnu_kl0_kt1": 1,
        "hh_ggf_hbb_hvvqqlnu_kl1_kt1": 1,
        "hh_ggf_hbb_hvvqqlnu_kl2p45_kt1": 1,
        "hh_ggf_hbb_hvvqqlnu_kl5_kt1": 1,
    }

    input_features = [
        # event features
        "mli_ht", "mli_lt", "mli_n_jet", "mli_n_btag",
        "mli_b_score_sum",
        # bb system
        "mli_dr_bb", "mli_dphi_bb", "mli_mbb",
        # jj system
        "mli_dr_jj", "mli_dphi_jj", "mli_mjj",
        # lnu system
        "mli_dphi_lnu", "mli_mlnu",
        # angles to lepton
        "mli_mindr_lb", "mli_mindr_lj",
        "mli_dphi_wl",
        # ww system
        "mli_mjjlnu", "mli_mjjl",
        # HH system
        "mli_dphi_bb_jjlnu", "mli_dr_bb_jjlnu",
        "mli_dphi_bb_jjl", "mli_dr_bb_jjl", "mli_dphi_bb_nu", "mli_dphi_jj_nu", "mli_dr_bb_l", "mli_dr_jj_l",
        "mli_mbbjjlnu", "mli_mbbjjl", "mli_mindr_jj",
        "mli_s_min",
        # VBF features
        "mli_vbf_deta", "mli_vbf_mass", "mli_vbf_tag", "mli_met_pt",
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["b1", "b2", "j1", "j2"]
        for var in ["pt", "eta", "b_score"]
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["lep"]
        for var in ["pt", "eta"]
    ]

    store_name: str = "inputs_inclusive"

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

processes = {
    "default": DenseClassifierSL._default__processes,
    "merge_hh": ["sig_ggf", "sig_vbf", "tt", "st", "dy", "h"],
}
input_features = {
    "default": DenseClassifierSL.input_features,
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
}
class_factors = {
    "default": DenseClassifierSL._default__class_factors,
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
#
# derived MLModels
#

sl_22post_test = DenseClassifierSL.derive("sl_22post_test", cls_dict={
    "training_configs": lambda self, requested_configs: ["l22post"],
    "processes": ["hh_ggf_hbb_hvvqqlnu_kl1_kt1", "tt"],
    "train_nodes": {"hh_ggf_hbb_hvvqqlnu_kl1_kt1": {"ml_id": 0}, "tt": {"ml_id": 1}},
    "epochs": 10,
},
)
sl_22post = DenseClassifierSL.derive("sl_22post", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
},
)
