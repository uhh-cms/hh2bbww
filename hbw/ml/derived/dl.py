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

from hbw.config.processes import create_combined_proc_forML

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


class DenseClassifierDL(DenseModelMixin, ModelFitMixin, MLClassifierBase):

    combine_processes = ()

    processes = (
        "sig",
        "tt",
        "st",
        "dy",
    )

    ml_process_weights = {
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
    negative_weights: str = "ignore"

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
        for proc in self.combine_processes:
            if proc not in self.config_inst.processes:
                proc_name = str(proc)
                proc_dict = DotDict(self.combine_processes[proc])
                create_combined_proc_forML(self.config_inst, proc_name, proc_dict)

        for proc in self.processes:
            for config_inst in self.config_insts:
                if f"mlscore.{proc}" not in config_inst.variables:
                    config_inst.add_variable(
                        name=f"mlscore.{proc}",
                        expression=f"mlscore.{proc}",
                        null_value=-1,
                        binning=(1000, 0., 1.),
                        x_title=f"DNN output score {config_inst.get_process(proc).x('ml_label', '')}",
                        aux={"rebin": 40},
                    )
                    config_inst.add_variable(
                        name=f"mlscore40.{proc}",
                        expression=f"mlscore.{proc}",
                        null_value=-1,
                        binning=(40, 0., 1.),
                        x_title=f"DNN output score {config_inst.get_process(proc).x('ml_label', '')}",
                    )


#
# derived MLModels
#

dl_22post = DenseClassifierDL.derive("dl_22post", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "tt", "st", "dy"],
})
dl_22post_test = dl_22post.derive("dl_22post_test", cls_dict={
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "st_tchannel_t"],
})
dl_22post_limited = dl_22post.derive("dl_22post_limited", cls_dict={
    "training_configs": lambda self, requested_configs: ["l22post"],
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "st_tchannel_t"],
})
dl_22 = DenseClassifierDL.derive("dl_22", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post", "c22pre"],
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "tt", "st", "dy"],
})
dl_22_test = DenseClassifierDL.derive("dl_22_test", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post", "c22pre"],
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "dy"],
    # "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "tt"],
})

dl_22post_testproc = dl_22post.derive("dl_22post_testproc", cls_dict={
    "training_configs": lambda self, requested_configs: ["l22post"],
    "combine_processes": {
        "test1": {
            # "name": "tt_and_st",
            "label": "my label",
            "sub_processes": ["st", "hh_ggf_hbb_hvv2l2nu_kl1_kt1"],
            "weighting": "equal",
        },
    },
    "processes": ["test1", "tt"],
})
dl_22post_sigtest = dl_22post.derive("dl_22post_sigtest", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "combine_processes": {
        "sig_01": {
            # "name": "tt_and_st",
            "label": "signal_kl0_1",
            "sub_processes": ["hh_ggf_hbb_hvv2l2nu_kl0_kt1", "hh_ggf_hbb_hvv2l2nu_kl1_kt1"],
            "weighting": "equal",
        },
        "sig_25": {
            # "name": "tt_and_st",
            "label": "signal_kl2p45_5",
            "sub_processes": ["hh_ggf_hbb_hvv2l2nu_kl2p45_kt1", "hh_ggf_hbb_hvv2l2nu_kl5_kt1"],
            "weighting": "xsec",
        },
    },
    "processes": ["sig_01", "sig_25"],
})

dl_22post_test = dl_22post.derive("dl_22post_test", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "combine_processes": {
        "signal": {
            # "name": "tt_and_st",
            "label": "signal",
            "sub_processes": [
                "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
            ],
            "weighting": "xsec",
        },
        "top": {
            # "name": "tt_and_st",
            "label": "top_induced",
            "sub_processes": ["st", "tt"],
            "weighting": "equal",
        },
    },
    "processes": ["signal", "top"],
})

dl_22post_test2 = dl_22post.derive("dl_22post_test2", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "combine_processes": {
        "signal": {
            # "name": "tt_and_st",
            "label": "signal",
            "sub_processes": [
                "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
            ],
            "weighting": "xsec",
        },
        "top": {
            # "name": "tt_and_st",
            "label": "top_induced",
            "sub_processes": ["st", "tt"],
            "weighting": "xsec",
        },
    },
    "processes": ["signal", "top"],
})

dl_22post_ml_study_5 = dl_22post.derive("dl_22post_ml_study_5", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "combine_processes": {
        "signal_ggf5": {
            # "name": "tt_and_st",
            "label": "Signal GGF",
            "sub_processes": [
                "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
            ],
            "weighting": "equal",
        },
        "signal_vbf5": {
            # "name": "tt_and_st",
            "label": "Signal VBF",
            "sub_processes": [
                "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
                "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1",
                "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43",
                "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94",
                "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36",
                "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39",
            ],
            "weighting": "xsec",
        },
    },
    "processes": [
        "signal_ggf5",
        "signal_vbf5",
        "tt",
        "st",
        "dy",
        "h",
    ],
})

dl_22post_ml_study_4 = dl_22post.derive("dl_22post_ml_study_4", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "combine_processes": {
        "signal_ggf4": {
            # "name": "tt_and_st",
            "label": "Signal GGF",
            "sub_processes": [
                "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
            ],
            "weighting": "xsec",
        },
        "signal_vbf4": {
            # "name": "tt_and_st",
            "label": "Signal VBF",
            "sub_processes": [
                "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
                "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1",
                "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43",
                "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94",
                "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36",
                "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39",
            ],
            "weighting": "xsec",
        },
    },
    "processes": [
        "signal_ggf4",
        "signal_vbf4",
        "tt",
        "st",
        "dy",
        "h",
    ],
})

dl_22post_ml_study_1 = dl_22post.derive("dl_22post_ml_study_1", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "combine_processes": {
        "signal_ggf": {
            # "name": "tt_and_st",
            "label": "Signal GGF",
            "sub_processes": [
                "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
            ],
            "weighting": "xsec",
        },
        "signal_vbf": {
            # "name": "tt_and_st",
            "label": "Signal VBF",
            "sub_processes": [
                "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
                "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1",
                "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43",
                "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94",
                "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36",
                "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39",
            ],
            "weighting": "xsec",
        },
    },
    "processes": [
        "signal_ggf",
        "signal_vbf",
        "tt",
        "st",
        "dy",
        "h",
    ],
})

dl_22post_ml_study_2 = dl_22post.derive("dl_22post_ml_study_2", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "combine_processes": {
        "signal_ggf2": {
            # "name": "tt_and_st",
            "label": "Signal GGF",
            "sub_processes": [
                "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
                "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
            ],
            "weighting": "equal",
        },
        "signal_vbf2": {
            # "name": "tt_and_st",
            "label": "Signal VBF",
            "sub_processes": [
                "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
                "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1",
                "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43",
                "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94",
                "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36",
                "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39",

            ],
            "weighting": "xsec",
        },
    },
    "processes": [
        "signal_ggf2",
        "signal_vbf2",
        "tt",
        "st",
        "dy",
        "h",
    ],
})
#
# setups with different processes (0: baseline, 1: add SM vbf + single H, 2: add SL+all HH variations)
# NOTE: we should decide which signal processes exactly to use:
# kl5 might confuse our DNN, and we should not use all vbf variations
#

dl_22_procs0 = DenseClassifierDL.derive("dl_22_procs0", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post", "c22pre"],
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "tt", "st", "dy"],
})
dl_22_procs1 = DenseClassifierDL.derive("dl_22_procs1", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post", "c22pre"],
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1", "tt", "st", "dy", "h"],
})
dl_22_procs1_w0 = dl_22_procs1.derive("dl_22_procs1_w0", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post"],
    "ml_process_weights": {
        "hh_ggf_hbb_hvv2l2nu_kl1_kt1": 1,
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": 1,
        "tt": 2,
        "st": 2,
        "dy": 2,
        "h": 1,
    },
})
dl_22_procs1_w1 = dl_22_procs1.derive("dl_22_procs1_w1", cls_dict={
    "ml_process_weights": {
        "hh_ggf_hbb_hvv2l2nu_kl1_kt1": 1,
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": 1,
        "tt": 16,
        "st": 2,
        "dy": 2,
        "h": 2,
    },
})
dl_22_procs1_w0_inp1 = DenseClassifierDL.derive("dl_22_procs1_w0_inp1", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post", "c22pre"],
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1", "tt", "st", "dy", "h"],
    "ml_process_weights": {
        "hh_ggf_hbb_hvv2l2nu_kl1_kt1": 1,
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": 1,
        "tt": 2,
        "st": 2,
        "dy": 2,
        "h": 2,
    },
    "input_features": [
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
    ],
})


dl_22_procs2 = DenseClassifierDL.derive("dl_22_procs2", cls_dict={
    "training_configs": lambda self, requested_configs: ["c22post", "c22pre"],
    "processes": ["hh_ggf_hbb_hvv", "hh_vbf_hbb_hvv", "tt", "st", "dy", "h"],
})

dl_17 = DenseClassifierDL.derive("dl_17", cls_dict={
    "training_configs": lambda self, requested_configs: ["c17"],
    "processes": ["sig", "tt", "st", "dy"],
})

# testing of hyperparameter changes
dl_22.derive("dl_22_stepsMax", cls_dict={"steps_per_epoch": "max_iter_valid"})
dl_22.derive("dl_22_steps100", cls_dict={"steps_per_epoch": 100})
dl_22.derive("dl_22_steps1000", cls_dict={"steps_per_epoch": 1000})

# model for testing
dl_22.derive("dl_22_v1")
dl_22_limited = dl_22post.derive("dl_22_limited", cls_dict={
    "training_configs": lambda self, requested_configs: ["l22post"],
    "epochs": 4,
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "tt_dl"],
})
