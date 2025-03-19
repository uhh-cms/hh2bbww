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


class DenseClassifierDL(DenseModelMixin, ModelFitMixin, MLClassifierBase):

    combine_processes = ()

    _default__processes: tuple = (
        "sig",
        "tt",
        "st",
        "dy",
    )
    _default__train_classes: dict = {}

    _default__ml_process_weights = {
        "hh_ggf_hbb_hvv2l2nu_kl0_kt1": 1,
        "hh_ggf_hbb_hvv2l2nu_kl1_kt1": 1,
        "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1": 1,
        "hh_ggf_hbb_hvv2l2nu_kl5_kt1": 1,
        "sig": 1,
        "tt": 2,
        "st": 2,
        "v_lep": 2,
        "tt_bkg": 2,
        "w_lnu": 2,
        "dy": 2,
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
        "processes", "ml_process_weights", "train_classes",
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
        "processes", "train_classes", "ml_process_weights",
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
# derived MLModels
#


dl_22post = DenseClassifierDL.derive("dl_22post", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "tt", "st", "dy"],
 })
dl_22post_test = dl_22post.derive("dl_22post_test", cls_dict={
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "st_tchannel_t"],
    "epochs": 20,
 })
dl_22post_limited = dl_22post.derive("dl_22post_limited", cls_dict={
    "training_configs": lambda self, requested_configs: ["l22post"],
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "st_tchannel_t"],
    "epochs": 6,
 })
