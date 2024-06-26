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

    processes = (
        "sig",
        "tt",
        "st",
        "dy_lep",
    )

    ml_process_weights = {
        "ggHH_kl_0_kt_1_dl_hbbhww": 1,
        "ggHH_kl_1_kt_1_dl_hbbhww": 1,
        "ggHH_kl_2p45_kt_1_dl_hbbhww": 1,
        "ggHH_kl_5_kt_1_dl_hbbhww": 1,
        "sig": 1,
        "tt": 2,
        "st": 2,
        "v_lep": 2,
        "tt_bkg": 2,
        "w_lnu": 2,
        "dy_lep": 2,
    }

    input_features = [
        # event features
        "mli_ht", "mli_lt", "mli_n_jet", "mli_n_btag",
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
        "mli_vbf_deta", "mli_vbf_invmass", "mli_vbf_tag",
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
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["fj"]
        for var in ["pt", "eta", "phi", "mass", "msoftdrop"]
    ]

    store_name: str = "inputs_inclusive"

    folds: int = 5
    negative_weights: str = "handle"

    # overwriting DenseModelMixin parameters
    activation: str = "relu"
    layers: tuple = (512, 512, 512)
    dropout: float = 0.20
    learningrate: float = 0.00050

    # overwriting ModelFitMixin parameters
    callbacks: set = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True
    reduce_lr_factor: float = 0.8
    reduce_lr_patience: int = 3
    epochs: int = 100
    batchsize: int = 2 ** 12
    steps_per_epoch: Union[int, str] = "iter_smallest_process"

    # parameters to add into the `parameters` attribute to determine the 'parameters_repr' and to store in a yaml file
    bookkeep_params: set[str] = {
        # base params
        "data_loader", "input_features", "train_val_test_split",
        "processes", "ml_process_weights", "negative_weights", "folds",
        # DenseModelMixin
        "activation", "layers", "dropout", "learningrate",
        # ModelFitMixin
        "callbacks", "reduce_lr_factor", "reduce_lr_patience",
        "epochs", "batchsize",
    }

    # parameters that can be overwritten via command line
    settings_parameters: set[str] = {
        # base params
        "processes", "ml_process_weights",
        "negative_weights",
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

    def setup(self):
        # dynamically add variables for the quantities produced by this model
        # NOTE: since these variables are only used in ConfigTasks,
        #       we do not need to add these variables to all configs
        for proc in self.processes:
            if f"mlscore.{proc}_manybins" not in self.config_inst.variables:
                self.config_inst.add_variable(
                    name=f"mlscore.{proc}_manybins",
                    expression=f"mlscore.{proc}",
                    null_value=-1,
                    binning=(1000, 0., 1.),
                    x_title=f"DNN output score {self.config_inst.get_process(proc).x.ml_label}",
                    aux={"rebin": 40},
                )
                self.config_inst.add_variable(
                    name=f"mlscore40.{proc}",
                    expression=f"mlscore.{proc}",
                    null_value=-1,
                    binning=(40, 0., 1.),
                    x_title=f"DNN output score {self.config_inst.get_process(proc).x.ml_label}",
                )


#
# derived MLModels
#

dl_22post = DenseClassifierDL.derive("dl_22post", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "processes": ["ggHH_kl_1_kt_1_dl_hbbhww", "tt", "st", "dy_lep"],
})
dl_22post_test = dl_22post.derive("dl_22post_test", cls_dict={
    "processes": ["ggHH_kl_1_kt_1_dl_hbbhww", "st_tchannel_t"],
})
dl_22post_limited = dl_22post.derive("dl_22post_limited", cls_dict={
    "training_configs": lambda self, requested_configs: ["l22post"],
    "processes": ["ggHH_kl_1_kt_1_dl_hbbhww", "st_tchannel_t"],
})
dl_22 = DenseClassifierDL.derive("dl_22", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post", "c22pre"],
    "processes": ["ggHH_kl_1_kt_1_dl_hbbhww", "tt", "st", "dy_lep"],
})
dl_17 = DenseClassifierDL.derive("dl_17", cls_dict={
    "training_configs": lambda self, requested_configs: ["c17"],
    "processes": ["sig", "tt", "st", "dy_lep"],
})

# testing of hyperparameter changes
dl_22.derive("dl_22_stepsMax", cls_dict={"steps_per_epoch": "max_iter_valid"})
dl_22.derive("dl_22_steps100", cls_dict={"steps_per_epoch": 100})
dl_22.derive("dl_22_steps1000", cls_dict={"steps_per_epoch": 1000})

# model for testing
dl_22.derive("dl_22_v1")
dl_22_limited = dl_22post.derive("dl_22_limited", cls_dict={
    "training_configs": lambda self, requested_configs: ["l22pre", "l22post"],
    "processes": ["ggHH_kl_1_kt_1_dl_hbbhww", "st_tchannel_t"],
})
