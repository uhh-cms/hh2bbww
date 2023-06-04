# coding: utf-8

"""
ML models using the MLClassifierBase and Mixins
"""

from __future__ import annotations

from typing import Sequence

import law
import order as od

from columnflow.util import maybe_import

from hbw.ml.base import MLClassifierBase
from hbw.ml.mixins import DenseModelMixin, ModelFitMixin

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


class DenseClassifier(ModelFitMixin, DenseModelMixin, MLClassifierBase):

    processes = [
        "ggHH_kl_1_kt_1_sl_hbbhww",
        "tt",
        "st",
        "v_lep",
        # "w_lnu",
        # "dy_lep",
    ]

    ml_process_weights = {
        "ggHH_kl_1_kt_1_sl_hbbhww": 1,
        "tt": 8,
        "st": 8,
        "v_lep": 8,
        "w_lnu": 8,
        "dy_lep": 8,
    }

    dataset_names = {
        "ggHH_kl_1_kt_1_sl_hbbhww_powheg",
        # TTbar
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        # SingleTop
        "st_tchannel_t_powheg",
        "st_tchannel_tbar_powheg",
        "st_twchannel_t_powheg",
        "st_twchannel_tbar_powheg",
        "st_schannel_lep_amcatnlo",
        # "st_schannel_had_amcatnlo",
        # WJets
        "w_lnu_ht70To100_madgraph",
        "w_lnu_ht100To200_madgraph",
        "w_lnu_ht200To400_madgraph",
        "w_lnu_ht400To600_madgraph",
        "w_lnu_ht600To800_madgraph",
        "w_lnu_ht800To1200_madgraph",
        "w_lnu_ht1200To2500_madgraph",
        "w_lnu_ht2500_madgraph",
        # DY
        "dy_lep_m50_ht70to100_madgraph",
        "dy_lep_m50_ht100to200_madgraph",
        "dy_lep_m50_ht200to400_madgraph",
        "dy_lep_m50_ht400to600_madgraph",
        "dy_lep_m50_ht600to800_madgraph",
        "dy_lep_m50_ht800to1200_madgraph",
        "dy_lep_m50_ht1200to2500_madgraph",
        "dy_lep_m50_ht2500_madgraph",
    }

    input_features = [
        "mli_ht", "mli_n_jet", "mli_n_deepjet",
        # "mli_deepjetsum", "mli_b_deepjetsum", "mli_l_deepjetsum",
        "mli_dr_bb", "mli_dphi_bb", "mli_mbb", "mli_mindr_lb",
        "mli_dr_jj", "mli_dphi_jj", "mli_mjj", "mli_mindr_lj",
        "mli_dphi_lnu", "mli_mlnu", "mli_mjjlnu", "mli_mjjl", "mli_dphi_bb_jjlnu", "mli_dr_bb_jjlnu",
        "mli_dphi_bb_jjl", "mli_dr_bb_jjl", "mli_dphi_bb_nu", "mli_dphi_jj_nu", "mli_dr_bb_l", "mli_dr_jj_l",
        "mli_mbbjjlnu", "mli_mbbjjl", "mli_s_min",
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["b1", "b2", "j1", "j2", "lep", "met"]
        for var in ["pt", "eta"]
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["fj"]
        for var in ["pt", "eta", "phi", "mass", "msoftdrop", "deepTagMD_HbbvsQCD"]
    ]

    store_name = "inputs_dense"

    folds = 5
    layers = [512, 512, 512]
    activation = "relu"
    learningrate = 0.00050
    batchsize = 2 ** 12
    epochs = 500
    epweight = True
    dropout = 0.50

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def training_configs(self, requested_configs: Sequence[str]) -> list[str]:
        # default config
        if len(requested_configs) == 1:
            return list(requested_configs)
        else:
            # use config_2017 per default
            return ["config_2017"]

    def training_calibrators(self, config_inst: od.Config, requested_calibrators: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        return ["skip_jecunc"]

    def training_selector(self, config_inst: od.Config, requested_selector: str) -> str:
        # fix MLTraining Phase Space
        return "default"

    def training_producers(self, config_inst: od.Config, requested_producers: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        return ["ml_inputs"]


cls_dict_test = {
    "epochs": 1,
    "processes": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "st", "v_lep"],
    "dataset_names": [
        "ggHH_kl_1_kt_1_sl_hbbhww_powheg", "tt_dl_powheg",
        "st_tchannel_t_powheg", "w_lnu_ht400To600_madgraph",
    ],
}


dense_test = DenseClassifier.derive("dense_test", cls_dict=cls_dict_test)
dense_default = DenseClassifier.derive("dense_default", cls_dict={})
dense_short = DenseClassifier.derive("dense_short", cls_dict={"epochs": 5})
