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
# from hbw.config.styling import color_palette


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
            "label": r"HH_{ggF}",
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
