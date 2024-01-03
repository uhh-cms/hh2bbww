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


class DenseClassifierRes(ModelFitMixin, DenseModelMixin, MLClassifierBase):

    processes = [
        "graviton_hh_ggf_bbww_m600",
        # "tt",
        # "st",
        # "w_lnu",
        # "dy_lep",
        "t_bkg",
        "v_lep",
    ]

    ml_process_weights = {
        "graviton_hh_ggf_bbww_600": 1,
        "tt": 8,
        "st": 8,
        "w_lnu": 8,
        "dy_lep": 8,
        # "t_bkg": 1,
        # "v_lep": 1,
    }

    dataset_names = {
        "graviton_hh_ggf_bbww_m600_madgraph",
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
        "st_schannel_had_amcatnlo",
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

    # input_features = [
    #     # "ht",
    #     # "m_bb",
    #     # "deltaR_bb",
    #     "mli_ht", "mli_n_jet", "mli_n_deepjet",
    #     # "mli_deepjetsum", "mli_b_deepjetsum", "mli_l_deepjetsum",
    #     "mli_dr_bb", "mli_dphi_bb", "mli_mbb", "mli_mindr_lb", "mli_dr_jj", "mli_dphi_jj", "mli_mjj", "mli_mindr_lj",
    #     "mli_dphi_lnu", "mli_mlnu", "mli_mjjlnu", "mli_mjjl", "mli_dphi_bb_jjlnu", "mli_dr_bb_jjlnu",
    #     "mli_dphi_bb_jjl", "mli_dr_bb_jjl", "mli_dphi_bb_nu", "mli_dphi_jj_nu", "mli_dr_bb_l", "mli_dr_jj_l",
    #     "mli_mbbjjlnu", "mli_mbbjjl", "mli_s_min",
    #     "mli_pt_jj", "mli_eta_jj", "mli_phi_jj",
    #     "mli_pt_lnu", "mli_eta_lnu", "mli_phi_lnu",
    #     "mli_pt_jjlnu", "mli_eta_jjlnu", "mli_phi_jjlnu",
    #     "mli_pt_jjl", "mli_eta_jjl", "mli_phi_jjl",
    #     # "mli_pt_bbjjlnu", "mli_eta_bbjjlnu", "mli_phi_bbjjlnu",
    #     # "mli_pt_bbjjl", "mli_eta_bbjjl", "mli_phi_bbjjl",
    #     "mli_pt_bb", "mli_eta_bb", "mli_phi_bb",
    # ] + [
    #     f"mli_{obj}_{var}"
    #     for obj in ["b1", "b2", "j1", "j2", "lep", "met"]
    #     for var in ["pt", "eta"]
    # ] + [
    #     f"mli_{obj}_{var}"
    #     for obj in ["fj"]
    #     for var in ["pt", "eta", "phi", "mass", "msoftdrop", "deepTagMD_HbbvsQCD"]
    # ]

    input_features = [
        "ht",
        "m_bb",
        "deltaR_bb",
        "mli_ht", "mli_n_jet", "mli_n_deepjet",
        # "mli_deepjetsum", "mli_b_deepjetsum", "mli_l_deepjetsum",
        "mli_dr_bb", "mli_dphi_bb", "mli_mindr_lb", "mli_mindr_lj",

        "mli_m_tbkg2",
        "mli_m_tlep1",
        "mli_m_tlep2",
        "mli_m_tbkg1",
        "mli_mbbjjlnu",
        "mli_pt_bb",
        "mli_pt_jjlnu",
        "pnn_feature",
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["b1", "j1", "j2", "b2", "met"]
        for var in ["pt"]
    ]

    store_name = "inputs_v1"

    folds = 3
    validation_fraction = 0.20
    learningrate = 0.00050
    negative_weights = "handle"

    # overwriting DenseModelMixin parameters
    activation = "relu"
    layers = (100, 200, 100)
    dropout = 0.20

    # overwriting ModelFitMixin parameters
    callbacks = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup = True
    reduce_lr_factor = 0.8
    reduce_lr_patience = 3
    # epochs = 100
    # batchsize = 8000
    epochs = 200
    batchsize = 2 ** 12

    # parameters to add into the `parameters` attribute and store in a yaml file
    bookkeep_params = [
        # base params
        "processes", "dataset_names", "input_features", "validation_fraction", "ml_process_weights",
        "negative_weights", "folds",
        # DenseModelMixin
        "activation", "layers", "dropout",
        # ModelFitMixin
        "callbacks", "reduce_r_factor", "reduce_lr_patience",
        "epochs", "batchsize",
    ]

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def setup(self):
        # dynamically add variables for the quantities produced by this model
        # NOTE: since these variables are only used in ConfigTasks,
        #       we do not need to add these variables to all configs
        for proc in self.processes:
            if f"mlscore.{proc}" not in self.config_inst.variables:
                self.config_inst.add_variable(
                    name=f"mlscore.{proc}",
                    null_value=-1,
                    binning=(40, 0., 1.),
                    x_title=f"DNN output score {self.config_inst.get_process(proc).x.ml_label}",
                )
                self.config_inst.add_variable(
                    # TODO: to be used for rebinning
                    name=f"mlscore.{proc}_manybins",
                    expression=f"mlscore.{proc}",
                    null_value=-1,
                    binning=(1000, 0., 1.),
                    x_title=f"DNN output score {self.config_inst.get_process(proc).x.ml_label}",
                )
                hh_bins = [0.0, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .92, 1.0]
                bkg_bins = [0.0, 0.4, 0.7, 1.0]
                self.config_inst.add_variable(
                    # used for inference as long as we don't have our rebin task in place
                    name=f"mlscore.{proc}_rebin",
                    expression=f"mlscore.{proc}",
                    null_value=-1,
                    binning=hh_bins if "HH" in proc else bkg_bins,
                    x_title=f"DNN output score {self.config_inst.get_process(proc).x.ml_label}",
                )

    def training_configs(self, requested_configs: Sequence[str]) -> list[str]:
        # default config
        if len(requested_configs) == 1:
            return list(requested_configs)
        else:
            # use config_2017 per default
            return ["l17"]

    def training_calibrators(self, config_inst: od.Config, requested_calibrators: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        return ["skip_jecunc"]

    def training_selector(self, config_inst: od.Config, requested_selector: str) -> str:
        # fix MLTraining Phase Space
        return "sl" if self.config_inst.has_tag("is_sl") else "dl"

    def training_producers(self, config_inst: od.Config, requested_producers: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        return ["sl_res_ml_inputs"]


cls_dict_test = {
    "epochs": 1,
    "processes": ["graviton_hh_ggf_bbww_m600", "w_lnu"],
    "dataset_names": {
        "w_lnu_ht70To100_madgraph",
    },
}

# ML Model with reduced number of datasets
dense_test = DenseClassifierRes.derive("dense_test_res", cls_dict=cls_dict_test)

# Graviton ML Models

for m in [250, 350, 450, 600, 750, 1000]:
    processes = [
        f"graviton_hh_ggf_bbww_m{m}",
        "tt",
        "st",
        "w_lnu",
        "dy_lep",
        # "t_bkg",
        # "v_lep",
    ]
    ml_process_weights = {
        f"graviton_hh_ggf_bbww_m{m}": 1,
        "tt": 4,
        "st": 4,
        "w_lnu": 4,
        "dy_lep": 4,
        # "t_bkg": 1,
        # "v_lep": 1,
    }
    dataset_names = {
        f"graviton_hh_ggf_bbww_m{m}_madgraph",
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
        "st_schannel_had_amcatnlo",
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
    cls_dict_res = {
        "processes": processes,
        "dataset_names": dataset_names,
        "ml_process_weights": ml_process_weights,
    }
    dense_m_graviton = DenseClassifierRes.derive(f"dense_graviton{m}", cls_dict=cls_dict_res)
    dense_m_graviton_new_input = DenseClassifierRes.derive(f"dense_graviton{m}_new_input", cls_dict=cls_dict_res)
    # Parametrical NN  (PNN)

    processes = [
        "graviton_hh_ggf_bbww_m250",
        "graviton_hh_ggf_bbww_m350",
        "graviton_hh_ggf_bbww_m450",
        "graviton_hh_ggf_bbww_m600",
        "graviton_hh_ggf_bbww_m750",
        "graviton_hh_ggf_bbww_m1000",
        "tt",
        "st",
        "w_lnu",
        "dy_lep",
        # "t_bkg",
        # "v_lep",
    ]

    ml_process_weights = {
        "graviton_hh_ggf_bbww_250": 1,
        "graviton_hh_ggf_bbww_350": 1,
        "graviton_hh_ggf_bbww_450": 1,
        "graviton_hh_ggf_bbww_600": 1,
        "graviton_hh_ggf_bbww_750": 1,
        "graviton_hh_ggf_bbww_1000": 1,
        "tt": 6,
        "st": 6,
        "w_lnu": 6,
        "dy_lep": 6,
        # "t_bkg": 1,
        # "v_lep": 1,
    }

    dataset_names = {
        "graviton_hh_ggf_bbww_m250_madgraph",
        "graviton_hh_ggf_bbww_m350_madgraph",
        "graviton_hh_ggf_bbww_m450_madgraph",
        "graviton_hh_ggf_bbww_m600_madgraph",
        "graviton_hh_ggf_bbww_m750_madgraph",
        "graviton_hh_ggf_bbww_m1000_madgraph",
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
        "st_schannel_had_amcatnlo",
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
    cls_dict_pnn = {
        "processes": processes,
        "dataset_names": dataset_names,
        "ml_process_weights": ml_process_weights,
    }
    dense_res_pnn = DenseClassifierRes.derive("dense_res_pnn", cls_dict=cls_dict_pnn)
