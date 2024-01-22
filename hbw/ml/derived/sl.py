# coding: utf-8

"""
ML models using the MLClassifierBase and Mixins
"""

from __future__ import annotations

from columnflow.types import Sequence, Union

import law
import order as od

from columnflow.util import maybe_import
from columnflow.columnar_util import Route

from hbw.ml.base import MLClassifierBase
from hbw.ml.mixins import DenseModelMixin, ModelFitMixin

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


class DenseClassifierSL(ModelFitMixin, DenseModelMixin, MLClassifierBase):

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

    dataset_names: set = {
        "ggHH_kl_1_kt_1_sl_hbbhww_powheg",
        "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww_madgraph",
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

    input_features: list = [
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
        for var in ["pt", "eta", "phi", "mass", "msoftdrop"]
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

    def uses(self, config_inst: od.Config) -> set[Route | str]:
        if not all(var.startswith("mli_") for var in self.input_features):
            raise Exception(
                "We currently expect all input_features to start with 'mli_', which is not the case"
                f"for one of the variables in the 'input_features' {self.input_features}",
            )
        # include all variables starting with 'mli_' to enable reusing MergeMLEvents outputs
        return {"mli_*"} | {"normalization_weight"}

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
                    aux={"rebin": 25},  # automatically rebin to 40 bins for plotting tasks
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
        return ["ml_inputs"]


# 2022 ML Model
cls_dict = {
    "training_calibrators": lambda self, config_inst, requiested_calibrators: [],
    "training_configs": lambda self, requested_configs: ["c22"],
    "processes": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt"],
    "dataset_names": {
        "ggHH_kl_1_kt_1_sl_hbbhww_powheg", "tt_sl_powheg",
    },
}
dense_2022 = DenseClassifierSL.derive("dense_2022", cls_dict=cls_dict)

# copies of the default DenseClassifierSL for testing hard-coded changes
for i in range(10):
    dense_copy = DenseClassifierSL.derive(f"dense_{i}")

cls_dict_test = {
    "epochs": 4,
    "processes": ["ggHH_kl_1_kt_1_sl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww", "tt", "st", "v_lep"],
    "dataset_names": {
        "ggHH_kl_1_kt_1_sl_hbbhww_powheg", "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww_madgraph", "tt_dl_powheg",
        "st_tchannel_t_powheg", "w_lnu_ht400To600_madgraph",
    },
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

# reduced set of input features (simplification of training process)
inputs1 = [
    "mli_mbb", "mli_ht", "mli_n_jet", "mli_n_deepjet",
    "mli_mindr_lb", "mli_mindr_lj", "mli_mindr_jj", "mli_mlnu",
] + [
    f"mli_{obj}_{var}"
    for obj in ["b1", "b2", "j1", "j2", "lep", "met"]
    for var in ["pt"]
] + [
    f"mli_{obj}_{var}"
    for obj in ["b1", "b2"]
    for var in ["btagDeepFlavB"]
]

cls_dict["input_features"] = inputs1
dense_inputs1 = DenseClassifierSL.derive("dense_inputs1", cls_dict=cls_dict)
dense_inputs1_test = DenseClassifierSL.derive(
    "dense_inputs1_test", cls_dict=law.util.merge_dicts(cls_dict, cls_dict_test),
)

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
