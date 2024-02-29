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


class DenseClassifierSL(ModelFitMixin, DenseModelMixin, MLClassifierBase):
    # NOTE: the order of processes is crucial and should never be changed after having trained
    processes: list = [
        "ggHH_kl_1_kt_1_sl_hbbhww",
        "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww",
        "tt",
        "st",
        "v_lep",
        # "w_lnu",
        # "dy_lep",
    ]

    ml_process_weights: dict = {
        "ggHH_kl_1_kt_1_sl_hbbhww": 1,
        "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww": 1,
        "tt": 2,
        "st": 2,
        "v_lep": 2,
        "w_lnu": 2,
        "dy_lep": 2,
    }

    # NOTE: the order of input features is crucial and should never be changed after training
    input_features: list = [
        # event features
        "mli_ht", "mli_lt", "mli_n_jet", "mli_n_deepjet",
        "mli_deepjetsum", "mli_b_deepjetsum", "mli_l_deepjetsum",
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
        "mli_vbf_deta", "mli_vbf_invmass", "mli_vbf_tag",
    ] + [
        f"mli_{obj}_{var}"
        for obj in ["b1", "b2", "j1", "j2"]
        for var in ["btagDeepFlavB", "pt", "eta"]
    ] + [
        "mli_lep_pt", "mli_lep_eta", "mli_met_pt",
    ]

    store_name: str = "inputs_inclusive"
    dump_arrays: bool = False

    folds: int = 5
    validation_fraction: float = 0.20
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

    # parameters to add into the `parameters` attribute and store in a yaml file
    bookkeep_params: list = [
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
            if f"mlscore.{proc}_manybins" not in self.config_inst.variables:
                self.config_inst.add_variable(
                    # TODO: to be used for rebinning
                    name=f"mlscore.{proc}_manybins",
                    expression=f"mlscore.{proc}",
                    null_value=-1,
                    binning=(1000, 0., 1.),
                    x_title=f"DNN output score {self.config_inst.get_process(proc).x.ml_label}",
                    aux={"rebin": 25},  # automatically rebin to 40 bins for plotting tasks
                )


#
# derived MLModels
#

dense_22post = DenseClassifierSL.derive("dense_22post", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "processes": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "st", "v_lep"],
})
dense_22post_test = dense_22post.derive("dense_22post_test", cls_dict={
    "processes": ["ggHH_kl_1_kt_1_sl_hbbhww", "st_tchannel_t"],
})
dense_22 = DenseClassifierSL.derive("dense_22", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post", "c22pre"],
    "processes": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "st", "v_lep"],
})

# copies of the default DenseClassifierSL for testing hard-coded changes
for i in range(0):
    dense_copy = DenseClassifierSL.derive(f"dense_{i}")

cls_dict_test = {
    "epochs": 4,
    "processes": [
        "ggHH_kl_1_kt_1_sl_hbbhww",
        "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww",
        "tt_dl",
        "st_tchannel_t",
    ],
    "steps_per_epoch": "max_iter_valid",
}

# ML Model with reduced number of datasets
dense_test = DenseClassifierSL.derive("dense_test", cls_dict=cls_dict_test)

# our default MLModel
dense_default = DenseClassifierSL.derive("dense_default", cls_dict={})


dense_max_iter_bs12 = DenseClassifierSL.derive(
    "dense_max_iter_bs12", cls_dict={"steps_per_epoch": "max_iter_valid", "batchsize": 2 ** 12},
)
dense_max_iter_bs14 = DenseClassifierSL.derive(
    "dense_max_iter_bs14", cls_dict={"steps_per_epoch": "max_iter_valid", "batchsize": 2 ** 14},
)

# ML Model with longer epochs
cls_dict = {
    "steps_per_epoch": "max_iter_valid", "batchsize": 2 ** 14, "dropout": 0.50, "epochs": 200,
    "callbacks": {"backup", "checkpoint"},
}

dense_max_iter = DenseClassifierSL.derive("dense_max_iter", cls_dict=cls_dict)

# # for running the default setup with different numbers of epochs
# for n_epochs in (5, 10, 20, 50, 100, 200, 500):
#     _dnn = DenseClassifierSL.derive(f"dense_epochs_{n_epochs}", cls_dict={"epochs": n_epochs})

# # for testing different number of nodes per layer
# for n_nodes in (64, 128, 256, 512):
#     _dnn = DenseClassifierSL.derive(f"dense_3x{n_nodes}", cls_dict={"layers": (n_nodes, n_nodes, n_nodes)})

# # for testing different modes of handling negative weights
# for negative_weights_mode in ("handle", "ignore", "abs"):
#     _dnn = DenseClassifierSL.derive(
#         f"dense_w_{negative_weights_mode}",
#         cls_dict={"negative_weights": negative_weights_mode},
#     )

# # for testing different learning rates
# for learningrate in (0.05000, .00500, 0.00050, 0.00010, 0.00005, 0.00001):
#     _dnn = DenseClassifierSL.derive(
#         f"dense_lr_{str(learningrate).replace('.', 'p')}",
#         cls_dict={"learningrate": learningrate},
#     )

# # for testing different batchsizes
# for batchsize in (11, 12, 13, 14, 15, 16, 17):
#     _dnn = DenseClassifierSL.derive(
#         f"dense_bs_2pow{batchsize}",
#         cls_dict={"batchsize": 2 ** batchsize},
#     )

# # for testing different dropout rates
# for dropout in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
#     _dnn = DenseClassifierSL.derive(
#         f"dense_dropout_{str(dropout).replace('.', 'p')}",
#         cls_dict={"dropout": dropout},
#     )

# # for testing different weights between signal and backgrounds
# for bkg_weight in (1, 2, 4, 8, 16):
#     ml_process_weights = {proc_name: bkg_weight for proc_name in DenseClassifierSL.processes}
#     ml_process_weights["ggHH_kl_1_kt_1_sl_hbbhww"] = 1
#     _dnn = DenseClassifierSL.derive(
#         f"dense_bkgw_{str(bkg_weight)}",
#         cls_dict={"ml_process_weights": ml_process_weights},
#     )

logger.info(f"Number of derived DenseClassifierSLs: {len(DenseClassifierSL._subclasses)}")
