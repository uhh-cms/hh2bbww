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


class DenseClassifierDL(ModelFitMixin, DenseModelMixin, MLClassifierBase):

    processes = [
        "sig",
        # "sig_all",
        # "ggHH_kl_1_kt_1_dl_hbbhww",
        # "tt",
        # "st",
        "v_lep",
        "t_bkg",
        # "w_lnu",
        # "dy_lep",
    ]

    ml_process_weights = {
        "ggHH_kl_0_kt_1_dl_hbbhww": 1,
        "ggHH_kl_1_kt_1_dl_hbbhww": 1,
        "ggHH_kl_2p45_kt_1_dl_hbbhww": 1,
        "ggHH_kl_5_kt_1_dl_hbbhww": 1,
        "ggHH_sig": 1,
        "ggHH_sig_all": 1,
        "tt": 1,
        "st": 1,
        "v_lep": 1,
        "t_bkg": 1,
        "w_lnu": 1,
        "dy_lep": 1,
    }

    dataset_names = {
        "ggHH_kl_0_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_1_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_2p45_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_5_kt_1_dl_hbbhww_powheg",
        # TTbar
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        # SingleTop
        "st_tchannel_t_powheg",
        # "st_tchannel_tbar_powheg", #problem in previous task for production
        "st_twchannel_t_powheg",
        "st_twchannel_tbar_powheg",
        # "st_schannel_lep_amcatnlo", #problem with normalizatino weights..
        # "st_schannel_had_amcatnlo",
        # WJets commented out because no events avaible and hence no nomralization weights
        # "w_lnu_ht70To100_madgraph",
        # "w_lnu_ht100To200_madgraph",
        # "w_lnu_ht200To400_madgraph",
        #"w_lnu_ht400To600_madgraph",
        # "w_lnu_ht600To800_madgraph",
        #"w_lnu_ht800To1200_madgraph",
        # "w_lnu_ht1200To2500_madgraph",
        #"w_lnu_ht2500_madgraph",
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
        "mli_mll", "mli_min_dr_llbb", "mli_dr_ll", "mli_bb_pt",
        "mli_ht", "mli_n_jet", "mli_n_deepjet",
        "mli_deepjetsum", "mli_b_deepjetsum",
        "mli_dr_bb", "mli_dphi_bb", "mli_mbb", "mli_mindr_lb",
        "mli_dphi_ll", "mli_dphi_bb_nu", "mli_dphi_bb_llMET", "mli_mllMET",
        "mli_mbbllMET", "mli_dr_bb_llMET", "mli_ll_pt", "mli_met_pt",
        # "mli_met_eta",
        # "mli_dr_jj", "mli_dphi_jj", "mli_mjj", "mli_mindr_lj",
        # "mli_dphi_lnu", "mli_mlnu", "mli_mjjlnu", "mli_mjjl", "mli_dphi_bb_jjlnu", "mli_dr_bb_jjlnu",
        # "mli_dphi_bb_jjl", "mli_dr_bb_jjl", "mli_dphi_bb_nu", "mli_dphi_jj_nu", "mli_dr_bb_l", "mli_dr_jj_l",
        # "mli_mbbjjlnu", "mli_mbbjjl", "mli_s_min",
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["b1", "b2", "lep", "lep2"]
        for var in ["pt", "eta"]
    ]
    """
      + [
        f"mli_{obj}_{var}"
        for obj in ["fj"]
        for var in ["pt", "eta", "phi", "mass", "msoftdrop", "deepTagMD_HbbvsQCD"]
    ]
    """

    store_name = "inputs_v1"

    folds = 3
    layers = (164, 164, 164)
    activation = "relu"
    learningrate = 0.0005
    batchsize = 2 ** 12
    epochs = 100
    dropout = 0.50
    negative_weights = "abs"
    validation_fraction = 0.20

    # overwriting DenseModelMixin parameters
    activation = "relu"
    layers = (512, 512, 512)
    dropout = 0.20

    # overwriting ModelFitMixin parameters
    callbacks = {
        "backup", "checkpoint",  # "reduce_lr",
        # "early_stopping",
    }
    remove_backup = True
    reduce_lr_factor = 0.8
    reduce_lr_patience = 3
    epochs = 100
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
            return ["c17"]

    def training_calibrators(self, config_inst: od.Config, requested_calibrators: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        return ["skip_jecunc"]

    def training_selector(self, config_inst: od.Config, requested_selector: str) -> str:
        # fix MLTraining Phase Space
        return "sl" if self.config_inst.has_tag("is_sl") else "dl"

    def training_producers(self, config_inst: od.Config, requested_producers: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        return ["ml_inputs"] if self.config_inst.has_tag("is_sl") else ["dl_ml_inputs"]


# our default MLModel
dense_default = DenseClassifierDL.derive("dense_default_dl", cls_dict={})

cls_dict_test_kl1 = {
    "folds": 3,
    "epochs": 100,
    "processes": ["ggHH_kl_1_kt_1_dl_hbbhww", "v_lep", "t_bkg"],
    "dataset_names": {
        "ggHH_kl_1_kt_1_dl_hbbhww_powheg",
        # TTbar
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        # SingleTop
        "st_tchannel_t_powheg",
        # "st_tchannel_tbar_powheg", #problem in previous task for production
        "st_twchannel_t_powheg",
        "st_twchannel_tbar_powheg",
        # "st_schannel_lep_amcatnlo", #problem with normalizatino weights..
        # "st_schannel_had_amcatnlo",
        # WJets commented out because no events avaible and hence no nomralization weights
        "w_lnu_ht70To100_madgraph",
        "w_lnu_ht100To200_madgraph",
        "w_lnu_ht200To400_madgraph",
        # "w_lnu_ht400To600_madgraph",
        "w_lnu_ht600To800_madgraph",
        # "w_lnu_ht800To1200_madgraph",
        "w_lnu_ht1200To2500_madgraph",
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
}

# ML Model with reduced number of datasets
dense_test_kl1_dl = DenseClassifierDL.derive("dense_test_kl1_dl", cls_dict=cls_dict_test_kl1)

cls_dict_all_proc = {
    "folds": 3,
    "epochs": 100,
    # "processes": ["ggHH_kl_1_kt_1_dl_hbbhww", "dy_lep", "tt", "st", "w_lnu"],
}

# ML Model with reduced number of datasets
dense_all_proc = DenseClassifierDL.derive("dense_all_proc", cls_dict=cls_dict_all_proc)

#cls_dict_test_sig_all_dl = {
#    "folds": 3,
#    "epochs": 100,
#    "processes": ["ggHH_sig_all", "v_lep", "t_bkg"],
#}

# ML Model with reduced number of datasets
#dense_test_sig_all_dl = DenseClassifierDL.derive("dense_test_sig_all_dl", cls_dict=cls_dict_test_sig_all_dl)

cls_dict_test_aachen_dl = {
    "folds": 3,
    "epochs": 100,
    # "processes": ["ggHH_kl_0_kt_1_dl_hbbhww", "ggHH_kl_1_kt_1_dl_hbbhww", "ggHH_kl_2p45_kt_1_dl_hbbhww", "ggHH_kl_5_kt_1_dl_hbbhww", "w_lnu", "dy_lep", "tt", "st"],
    "processes": ["ggHH_sig_all", "w_lnu", "dy_lep", "tt", "st"],
}

# ML Model with reduced number of datasets
dense_test_aachen_dl = DenseClassifierDL.derive("dense_test_aachen_dl", cls_dict=cls_dict_test_aachen_dl)

cls_dict_test_inputfeatures_dl = {
    "folds": 3,
    "epochs": 100,
    # "processes": ["ggHH_kl_0_kt_1_dl_hbbhww", "ggHH_kl_1_kt_1_dl_hbbhww", "ggHH_kl_2p45_kt_1_dl_hbbhww", "ggHH_kl_5_kt_1_dl_hbbhww", "w_lnu", "dy_lep", "tt", "st"],
    "processes": ["ggHH_sig", "dy_lep", "tt", "st"],
    "dataset_names": {
        "ggHH_kl_0_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_1_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_2p45_kt_1_dl_hbbhww_powheg",
        # TTbar
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        # SingleTop
        "st_tchannel_t_powheg",
        # "st_tchannel_tbar_powheg", #problem in previous task for production
        "st_twchannel_t_powheg",
        "st_twchannel_tbar_powheg",
        # "st_schannel_lep_amcatnlo", #problem with normalizatino weights..
        # "st_schannel_had_amcatnlo",
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
}
# ML Model with reduced number of datasets
dense_test_inputfeatures_dl = DenseClassifierDL.derive("dense_test_inputfeatures_dl", cls_dict=cls_dict_test_inputfeatures_dl)

cls_dict_test_aachen_sig_dl = {
    "folds": 3,
    "epochs": 100,
    # "processes": ["ggHH_kl_0_kt_1_dl_hbbhww", "ggHH_kl_1_kt_1_dl_hbbhww", "ggHH_kl_2p45_kt_1_dl_hbbhww", "ggHH_kl_5_kt_1_dl_hbbhww", "w_lnu", "dy_lep", "tt", "st"],
    "processes": ["ggHH_sig", "dy_lep", "tt", "st"],
    "dataset_names": {
        "ggHH_kl_0_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_1_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_2p45_kt_1_dl_hbbhww_powheg",
        # TTbar
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        # SingleTop
        "st_tchannel_t_powheg",
        # "st_tchannel_tbar_powheg", #problem in previous task for production
        "st_twchannel_t_powheg",
        "st_twchannel_tbar_powheg",
        # "st_schannel_lep_amcatnlo", #problem with normalizatino weights..
        # "st_schannel_had_amcatnlo",
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
}

# ML Model with reduced number of datasets
dense_test_aachen_sig_dl = DenseClassifierDL.derive("dense_test_aachen_sig_dl", cls_dict=cls_dict_test_aachen_sig_dl)

cls_dict_test_aachen_weights_dl = {
    "folds": 3,
    "epochs": 100,
    # "processes": ["ggHH_kl_0_kt_1_dl_hbbhww", "ggHH_kl_1_kt_1_dl_hbbhww", "ggHH_kl_2p45_kt_1_dl_hbbhww", "ggHH_kl_5_kt_1_dl_hbbhww", "w_lnu", "dy_lep", "tt", "st"],
    "processes": ["ggHH_sig_all", "dy_lep", "tt", "st"],
}

# ML Model with reduced number of datasets
dense_test_aachen_weights_dl = DenseClassifierDL.derive("dense_test_aachen_weights_dl", cls_dict=cls_dict_test_aachen_weights_dl)

cls_dict_test_aachen_wo_wlnu_dl = {
    "folds": 3,
    "epochs": 100,
    # "processes": ["ggHH_kl_0_kt_1_dl_hbbhww", "ggHH_kl_1_kt_1_dl_hbbhww", "ggHH_kl_2p45_kt_1_dl_hbbhww", "ggHH_kl_5_kt_1_dl_hbbhww", "w_lnu", "dy_lep", "tt", "st"],
    "processes": ["ggHH_sig_all", "dy_lep", "tt", "st"],
}

# ML Model with reduced number of datasets
dense_test_aachen_wo_wlnu_dl = DenseClassifierDL.derive("dense_test_aachen_wo_wlnu_dl", cls_dict=cls_dict_test_aachen_wo_wlnu_dl)

cls_dict_test_sig_all_dl = {
    "folds": 3,
    "epochs": 100,
    "processes": ["ggHH_sig_all", "v_lep", "t_bkg"],
    "dataset_names": {
        "ggHH_kl_0_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_1_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_2p45_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_5_kt_1_dl_hbbhww_powheg",
        # TTbar
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        # SingleTop
        "st_tchannel_t_powheg",
        # "st_tchannel_tbar_powheg", #problem in previous task for production
        "st_twchannel_t_powheg",
        "st_twchannel_tbar_powheg",
        # "st_schannel_lep_amcatnlo", #problem with normalizatino weights..
        # "st_schannel_had_amcatnlo",
        # WJets commented out because no events avaible and hence no nomralization weights
        "w_lnu_ht70To100_madgraph",
        "w_lnu_ht100To200_madgraph",
        "w_lnu_ht200To400_madgraph",
        # "w_lnu_ht400To600_madgraph",
        "w_lnu_ht600To800_madgraph",
        # "w_lnu_ht800To1200_madgraph",
        "w_lnu_ht1200To2500_madgraph",
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
}

# ML Model with reduced number of datasets
dense_test_sig_all_dl = DenseClassifierDL.derive("dense_test_sig_all_dl", cls_dict=cls_dict_test_sig_all_dl)

cls_dict_test_sig_dl = {
    "folds": 3,
    "epochs": 100,
    "processes": ["ggHH_sig", "v_lep", "t_bkg"],
    "dataset_names": {
        "ggHH_kl_0_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_1_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_2p45_kt_1_dl_hbbhww_powheg",
        # TTbar
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        # SingleTop
        "st_tchannel_t_powheg",
        # "st_tchannel_tbar_powheg", #problem in previous task for production
        "st_twchannel_t_powheg",
        "st_twchannel_tbar_powheg",
        # "st_schannel_lep_amcatnlo", #problem with normalizatino weights..
        # "st_schannel_had_amcatnlo",
        # WJets commented out because no events avaible and hence no nomralization weights
        "w_lnu_ht70To100_madgraph",
        "w_lnu_ht100To200_madgraph",
        "w_lnu_ht200To400_madgraph",
        # "w_lnu_ht400To600_madgraph",
        "w_lnu_ht600To800_madgraph",
        # "w_lnu_ht800To1200_madgraph",
        "w_lnu_ht1200To2500_madgraph",
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
}

# ML Model with reduced number of datasets
dense_test_sig_dl = DenseClassifierDL.derive("dense_test_sig_dl", cls_dict=cls_dict_test_sig_dl)


cls_dict_test = {
    "folds": 2,
    "epochs": 90,
    "processes": ["ggHH_kl_1_kt_1_dl_hbbhww", "v_lep", "t_bkg"],
}

# ML Model with reduced number of datasets
dense_test_dl = DenseClassifierDL.derive("dense_test_dl", cls_dict=cls_dict_test)

# our default MLModel
dense_default_dl = DenseClassifierDL.derive("dense_default_dl", cls_dict={})
