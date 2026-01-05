# coding: utf-8

"""
hbw(dl) inference model.
"""

import law
from columnflow.util import DotDict
import hbw.inference.constants as const  # noqa
from hbw.inference.base import HBWInferenceModelBase


logger = law.logger.get_logger(__name__)


# patch, allowing user to fall back to old versions
use_old_version = law.config.get_expanded("analysis", "use_old_version", False)

#
# Defaults for all the Inference Model parameters
#

# used to set default requirements for cf.CreateDatacards based on the config
ml_model_name = ["multiclassv3", "ggfv3", "vbfv3"]
if use_old_version:
    ml_model_name = ["multiclassv1", "ggfv1", "vbfv1"]

# All categories to be included in the final datacard
config_categories = DotDict({
    "default": [
        "sr__1b__ml_sig_ggf",
        "sr__1b__ml_sig_vbf",
        "sr__1b__ml_tt",
        "sr__1b__ml_st",
        "sr__1b__ml_dy_m10toinf",
        "sr__1b__ml_h",
        "sr__2b__ml_sig_ggf",
        "sr__2b__ml_sig_vbf",
        "sr__2b__ml_tt",
        "sr__2b__ml_st",
        "sr__2b__ml_dy_m10toinf",
        "sr__2b__ml_h",
    ],
    "sr": [
        "sr__1b__ml_sig_ggf",
        "sr__1b__ml_sig_vbf",
        "sr__2b__ml_sig_ggf",
        "sr__2b__ml_sig_vbf",
    ],
    "sr_splitvbf": [
        "sr__1b__ml_sig_ggf",
        "sr__2b__ml_sig_ggf",
        "sr__resolved__1b__ml_sig_vbf",
        "sr__resolved__2b__ml_sig_vbf",
        "sr__boosted__ml_sig_vbf",
    ],
    "sr_boosted": [
        "sr__boosted__ml_sig_ggf",
        "sr__boosted__ml_sig_vbf",
    ],
    "sr_resolved": [
        "sr__resolved__1b__ml_sig_ggf",
        "sr__resolved__1b__ml_sig_vbf",
        "sr__resolved__2b__ml_sig_ggf",
        "sr__resolved__2b__ml_sig_vbf",
    ],
    "background": [
        "sr__1b__ml_tt",
        "sr__1b__ml_st",
        "sr__1b__ml_dy_m10toinf",
        "sr__1b__ml_h",
        "sr__2b__ml_tt",
        "sr__2b__ml_st",
        "sr__2b__ml_dy_m10toinf",
        "sr__2b__ml_h",
    ],
    "background_split": [
        "sr__resolved__1b__ml_tt",
        "sr__resolved__1b__ml_st",
        "sr__resolved__1b__ml_dy_m10toinf",
        "sr__resolved__1b__ml_h",
        "sr__resolved__2b__ml_tt",
        "sr__resolved__2b__ml_st",
        "sr__resolved__2b__ml_dy_m10toinf",
        "sr__resolved__2b__ml_h",
        "sr__boosted__ml_bkg",
    ],
    "background_resolved": [
        "sr__resolved__1b__ml_tt",
        "sr__resolved__1b__ml_st",
        "sr__resolved__1b__ml_dy_m10toinf",
        "sr__resolved__1b__ml_h",
        "sr__resolved__2b__ml_tt",
        "sr__resolved__2b__ml_st",
        "sr__resolved__2b__ml_dy_m10toinf",
        "sr__resolved__2b__ml_h",
    ],
    "no_nn_cats": [
        "sr__1b",
        "sr__2b",
    ],
    "no_nn_cats_with_boosted": [
        "sr__resolved__1b",
        "sr__resolved__2b",
        "sr__boosted",
    ],
    "bjet_incl": [
        "sr__ml_sig_ggf",
        "sr__ml_sig_vbf",
        "sr__ml_tt",
        "sr__ml_st",
        "sr__ml_dy",
        "sr__ml_h",
    ],
})
config_categories.default_boosted = (
    config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background_resolved
)
config_categories.default_boosted_mergedbkg = (
    config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background
)
config_categories.default_boosted_bkg = (
    config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background_resolved +
    ["sr__boosted__ml_bkg"]
)


systematics = DotDict({
    "lumi": [
        # "lumi_13TeV_2016",
        # "lumi_13TeV_2017",
        # "lumi_13TeV_1718",
        # "lumi_13TeV_correlated",
        "lumi_13p6TeV_2022",
        "lumi_13p6TeV_2023",
    ],
    "QCDscale": [
        "QCDscale_ttbar",
        "QCDscale_V",
        "QCDscale_VV",
        "QCDscale_VVV",
        "QCDscale_ggH",
        "QCDscale_qqH",
        "QCDscale_VH",
        "QCDscale_ttH",
        # "QCDscale_bbH",
        # "QCDscale_hh_ggf",  # should be included in inference model (THU_HH)
        "QCDscale_hh_vbf",
        # "QCDscale_VHH",
        # "QCDscale_ttHH",
    ],
    "pdf": [
        "pdf_gg",
        "pdf_qqbar",
        "pdf_qg",
        "pdf_Higgs_gg",
        "pdf_Higgs_qqbar",
        # "pdf_Higgs_qg",  # none so far
        "pdf_Higgs_ttH",
        # "pdf_Higgs_bbH",  # removed
        "pdf_Higgs_hh_ggf",
        "pdf_Higgs_hh_vbf",
        # "pdf_VHH",
        # "pdf_ttHH",
    ],
    "BR": [
        "BR_hbb",
        "BR_hww",
        "BR_hzz",
        "BR_htt",
        "BR_hgg",
    ],
    "rate_unconstrained": [
        "rate_ttbar",
        "rate_dy",
    ],
    "rate_unconstrained1": [
        "rate_ttbar",
        "rate_dy_lf",
        "rate_dy_hf",
    ],
    "rate_unconstrained2": [
        "rate_ttbar"
        "rate_st",
        "rate_dy_lf",
        "rate_dy_hf",
    ],
    "rate_unconstrained3": [
        "rate_ttbar",
        "rate_ttbar_boosted",
        "rate_dy_lf",
        "rate_dy_hf",
    ],
    "rate_unconstrained_bjet_uncorr": [
        "rate_ttbar_{bjet_cat}",
        "rate_dy_{bjet_cat}",
    ],
    "hbb_efficiency": [
        "eff_hbb_signal_ggf",
        "eff_hbb_signal_vbf",
        "eff_hbb_bkg_ggf",
        "eff_hbb_bkg_vbf",
        "eff_hbb_bkg_bkg",
        "eff_hbb_signal_bkg",
    ],
    "murf_envelope": [
        # "murf_envelope_hh_ggf_hbb_hvv2l2nu_kl1_kt1",
        "murf_envelope_ttbar",
        "murf_envelope_st",
        "murf_envelope_dy",
        # "murf_envelope_w",
        "murf_envelope_ttV",  # TODO: ttW has no murf/pdf weights
        "murf_envelope_VV",
        "murf_envelope_H",
        "murf_envelope_hh_ggf_hbb_hww",
        "murf_envelope_hh_ggf_hbb_hzz",
        "murf_envelope_hh_ggf_hbb_htt",
        # "murf_envelope_hh_vbf_hbb_hww",
        # "murf_envelope_hh_vbf_hbb_hzz",
        # "murf_envelope_hh_vbf_hbb_htt",
    ],
    "pdf_shape": [
        "pdf_shape_ttbar",
        "pdf_shape_st",
        "pdf_shape_dy",
        # "pdf_shape_w",
        "pdf_shape_ttV",  # TODO: ttW has no murf/pdf weights
        "pdf_shape_VV",
        "pdf_shape_H",
        "pdf_shape_hh_ggf_hbb_hww",
        "pdf_shape_hh_ggf_hbb_hzz",
        "pdf_shape_hh_ggf_hbb_htt",
        # "pdf_shape_hh_vbf_hbb_hww",
        # "pdf_shape_hh_vbf_hbb_hzz",
        # "pdf_shape_hh_vbf_hbb_htt",
    ],
    "btag": [
        "btag_hf",
        "btag_lf",
        "btag_hfstats1_{campaign}",
        "btag_hfstats2_{campaign}",
        "btag_lfstats1_{campaign}",
        "btag_lfstats2_{campaign}",
        "btag_cferr1",
        "btag_cferr2",
    ],
    "btag_year_uncorr": [
        "btag_hf_{year}",
        "btag_lf_{year}",
        "btag_hfstats1_{campaign}",
        "btag_hfstats2_{campaign}",
        "btag_lfstats1_{campaign}",
        "btag_lfstats2_{campaign}",
        "btag_cferr1_{year}",
        "btag_cferr2_{year}",
    ],
    "btag_bjet_uncorr": [
        "btag_hf_{bjet_cat}",
        "btag_lf_{bjet_cat}",
        "btag_hfstats1_{campaign}_{bjet_cat}",
        "btag_hfstats2_{campaign}_{bjet_cat}",
        "btag_lfstats1_{campaign}_{bjet_cat}",
        "btag_lfstats2_{campaign}_{bjet_cat}",
        "btag_cferr1_{bjet_cat}",
        "btag_cferr2_{bjet_cat}",
    ],
    "btag_cpn_uncorr": [
        "btag_hf_{campaign}",
        "btag_lf_{campaign}",
        "btag_hfstats1_{campaign}",
        "btag_hfstats2_{campaign}",
        "btag_lfstats1_{campaign}",
        "btag_lfstats2_{campaign}",
        "btag_cferr1_{campaign}",
        "btag_cferr2_{campaign}",
    ],
    "experiment": [
        "mu_id_sf",
        "mu_iso_sf",
        "e_sf",
        "e_reco_sf",
        "trigger_sf",
        "minbias_xs",
        "dy_correction",
    ],
    "experiment_cpn_uncorr": [
        "mu_id_sf_{campaign}",
        "mu_iso_sf_{campaign}",
        "e_sf_{campaign}",
        "e_reco_sf_{campaign}",
        "trigger_sf_{campaign}",
        "minbias_xs",  # do not decorrelate PU between campaigns
        "dy_correction",
    ],
    "other": [
        "isr",
        "fsr_ttbar",
        "fsr_st",
        "fsr_V",
        # "fsr_dy",
        # "fsr_w",
        "fsr_VV",
        "fsr_ttV",
        "fsr_H",  # NOTE: skip h_ggf and h_vbf because PSWeights missing in H->tautau
        "top_pt",
    ],
    "jerc_only": [
        "jer",
        "jec_Total",
    ],
    "jerc_only_bjet_uncorr": [
        "jer_{bjet_cat}",
        "jec_Total_{bjet_cat}",
    ],
    "jerc_only_cpn_uncorr": [
        "jer_{campaign}",
        "jec_Total_{campaign}",
    ],
    "jerc_only_year_uncorr": [
        "jer_{year}",
        "jec_Total_{year}",
    ],
})
systematics["rate_default"] = [
    *systematics.lumi,
    *systematics.QCDscale,
    *systematics.pdf,
    *systematics.BR,
    *systematics.hbb_efficiency,
    *systematics.rate_unconstrained3,
]
systematics["rate"] = [
    *systematics.lumi,
    *systematics.QCDscale,
    *systematics.pdf,
    *systematics.rate_unconstrained,
]
systematics["rate1"] = [
    *systematics.lumi,
    *systematics.QCDscale,
    *systematics.pdf,
    *systematics.rate_unconstrained1,
]
systematics["rate2"] = [
    *systematics.lumi,
    *systematics.QCDscale,
    *systematics.pdf,
    *systematics.rate_unconstrained2,
]
systematics["shape_only"] = [
    *systematics.murf_envelope,
    *systematics.pdf_shape,
    *systematics.btag,
    *systematics.experiment,
    *systematics.other,
]
systematics["shape_only_cpn_uncorr"] = [
    *systematics.murf_envelope,
    *systematics.pdf_shape,
    *systematics.btag_cpn_uncorr,
    *systematics.experiment_cpn_uncorr,
    *systematics.other,
]
systematics["shape"] = [
    *systematics.rate,
    *systematics.shape_only,
]
# default set of all systematics
systematics["default"] = [
    *systematics.rate_default,
    *systematics.shape_only_cpn_uncorr,
    *systematics.jerc_only_cpn_uncorr,
]
systematics["default_year_uncorr"] = [
    *systematics.rate_default,
    *systematics.murf_envelope,
    *systematics.pdf_shape,
    *systematics.btag_year_uncorr,
    *systematics.experiment,
    *systematics.other,
    *systematics.jerc_only_year_uncorr,
]
systematics["default_cpn_corr"] = [
    *systematics.rate_default,
    *systematics.shape_only,
    *systematics.jerc_only,
]

# different variations of systematic combinations (testing)
systematics["jerc"] = [
    *systematics.rate,
    *systematics.shape_only,
    *systematics.jerc_only,
]
systematics["jerc1"] = [
    *systematics.rate1,
    *systematics.shape_only,
    *systematics.jerc_only,
]
systematics["jerc2"] = [
    *systematics.rate1,
    *systematics.shape_only,
    *systematics.jerc_only_cpn_uncorr,
]
systematics["jerc3"] = [  # default fullsyst result
    *systematics.rate1,
    *systematics.shape_only_cpn_uncorr,
    *systematics.jerc_only_cpn_uncorr,
]
systematics["jerc4"] = [
    *systematics.rate2,
    *systematics.shape_only_cpn_uncorr,
    *systematics.jerc_only_cpn_uncorr,
]
systematics["rate_bjet_uncorr"] = [
    *systematics.QCDscale,
    *systematics.pdf,
    *systematics.rate_unconstrained_bjet_uncorr,
]
systematics["shape_bjet_uncorr"] = [
    *systematics.rate_bjet_uncorr,
    *systematics.murf_envelope,
    *systematics.pdf_shape,
    *systematics.btag_bjet_uncorr,
    *systematics.experiment,
    *systematics.other,
]
# All systematics with btag and rate uncertainites decorrelated between bjet categories
systematics["jerc_bjet_uncorr"] = [
    *systematics.rate_bjet_uncorr,
    *systematics.shape_bjet_uncorr,
    *systematics.jerc_only,
]
systematics["jerc_bjet_uncorr1"] = [
    *systematics.rate_bjet_uncorr,
    *systematics.shape_bjet_uncorr,
    *systematics.jerc_only_bjet_uncorr,
]

hhprocs_ggf = lambda hhdecay: [
    f"hh_ggf_{hhdecay}_kl0_kt1",
    f"hh_ggf_{hhdecay}_kl1_kt1",
    f"hh_ggf_{hhdecay}_kl2p45_kt1",
    f"hh_ggf_{hhdecay}_kl5_kt1",
]
hhprocs_vbf = lambda hhdecay: [
    f"hh_vbf_{hhdecay}_kv1p74_k2v1p37_kl14p4",
    f"hh_vbf_{hhdecay}_kvm0p758_k2v1p44_klm19p3",
    f"hh_vbf_{hhdecay}_kvm0p012_k2v0p03_kl10p2",
    f"hh_vbf_{hhdecay}_kv2p12_k2v3p87_klm5p96",
    f"hh_vbf_{hhdecay}_kv1_k2v1_kl1",
    f"hh_vbf_{hhdecay}_kv1_k2v0_kl1",  # missing bbtt sample
    f"hh_vbf_{hhdecay}_kvm0p962_k2v0p959_klm1p43",
    f"hh_vbf_{hhdecay}_kvm1p21_k2v1p94_klm0p94",
    f"hh_vbf_{hhdecay}_kvm1p6_k2v2p72_klm1p36",
    f"hh_vbf_{hhdecay}_kvm1p83_k2v3p57_klm3p39",  # missing bbtt sample
]
hhprocs = lambda hhdecay: [*hhprocs_ggf(hhdecay), *hhprocs_vbf(hhdecay)]

backgrounds = [
    "st_tchannel",
    "st_twchannel",
    "st_schannel",
    "tt",
    "ttw",
    "ttz",
    "dy_hf",
    "dy_lf",
    "w_lnu",
    "vv",
    "vvv",
    "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    "thq", "thw", "ttvh",
    "tttt",
    "ttvv",
    # TODO: add bbh
    # "qcd",  # probably not needed
]
backgrounds_skip_dy = [
    "st_tchannel",
    "st_twchannel",
    "st_schannel",
    "tt",
    "ttw",
    "ttz",
    "w_lnu",
    "vv",
    "vvv",
    "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    "thq", "thw", "ttvh",
    "tttt",
    "ttvv",
]

processes_dict = {
    "test": ["tt", *hhprocs("hbb_hww2l2nu")],
    "hww": [*backgrounds, *hhprocs("hbb_hww")],
    "hww2l2nu": [*backgrounds, *hhprocs("hbb_hww2l2nu")],
    "hwwzztt": [*backgrounds, *hhprocs("hbb_hww"), *hhprocs("hbb_hzz"), *hhprocs("hbb_htt")],
    "hwwzztt_skip_dy": [*backgrounds_skip_dy, *hhprocs("hbb_hww"), *hhprocs("hbb_hzz"), *hhprocs("hbb_htt")],
    "hwwzztt_ggf": [*backgrounds, *hhprocs_ggf("hbb_hww"), *hhprocs_ggf("hbb_hzz"), *hhprocs_ggf("hbb_htt")],
}

from hbw.ml.derived.dl import input_features
mli_inputs = input_features.v2


def config_variable_binary_ggf_and_vbf(self, config_cat_inst):
    """
    Function to set the config variable for the binary model.
    """
    if config_cat_inst.name == "sr__boosted":
        return "logit_mlscore.sig_vbf_binary"
    if "sig_ggf" in config_cat_inst.name:
        return "logit_mlscore.sig_ggf_binary"
    elif "sig_vbf" in config_cat_inst.name:
        return "logit_mlscore.sig_vbf_binary"
    elif "ggf" in config_cat_inst.name or "vbf" in config_cat_inst.name:
        return f"logit_mlscore.{config_cat_inst.x.root_cats.get('dnn').replace('ml_', '')}"
    elif config_cat_inst.x.root_cats.get("dnn"):
        # since we merge into 1 bin anyways, we can use either score
        return "logit_mlscore.sig_ggf_binary"
    else:
        # raise ValueError(f"Category {config_cat_inst.name} is not a DNN category.")
        logger.warning(
            f"Category {config_cat_inst.name} is not a DNN category, using binary classifier score.",
        )
        return "logit_mlscore.sig_ggf_binary"


default_cls_dict = {
    "ml_model_name": ml_model_name,
    "processes": processes_dict["hwwzztt"],
    "config_categories": config_categories.default,
    "systematics": systematics.rate,
    "config_variable": config_variable_binary_ggf_and_vbf,
    "mc_stats": True,
    "skip_data": True,
}

dl = HBWInferenceModelBase.derive("dl", cls_dict=default_cls_dict)

#
# currently "final" inference models
#

rate_only = dl.derive("rate_only", cls_dict={
    "systematics": systematics.rate_default,
    "config_categories": config_categories.default_boosted,
})
rate_only_vbftag1 = dl.derive("rate_only_vbftag1", cls_dict={
    "systematics": systematics.rate_default,
    "config_categories": config_categories.default_boosted,
    "ml_model_name": ["multiclassv3", "ggfv3", "vbfv3_tag"],
})
default = dl.derive("default", cls_dict={
    "systematics": systematics.default,
    "config_categories": config_categories.default_boosted,
})
test_bkgs = dl.derive("test_bkgs", cls_dict={
    "systematics": systematics.default,
    "config_categories": config_categories.background,
})

no_boosted = dl.derive("no_boosted", cls_dict={
    "systematics": systematics.default,
    "config_categories": config_categories.sr + config_categories.background,
    "unblind": True,
    "skip_data": False,
})
default_unblind = dl.derive("default_unblind", cls_dict={
    "systematics": systematics.default,
    "config_categories": config_categories.default_boosted,
    "unblind": True,
    "skip_data": False,
})


mbbllMET = default_unblind.derive("mbbllMET", cls_dict={
    "config_variable": lambda self, config_cat_inst: "mli_mbbllMET_rebinned3",
    "config_categories": ["sr"],
    "flow_strategy": "remove",
})
mbb = default_unblind.derive("mbb", cls_dict={
    "config_variable": lambda self, config_cat_inst: "mli_mbb_rebinned3",
    "config_categories": ["sr"],
    "flow_strategy": "remove",
})
b1_pt = default_unblind.derive("b1_pt", cls_dict={
    "config_variable": lambda self, config_cat_inst: "mli_b1_pt_rebinned3",
    "config_categories": ["sr"],
    "flow_strategy": "remove",
})
bb_pt = default_unblind.derive("bb_pt", cls_dict={
    "config_variable": lambda self, config_cat_inst: "mli_bb_pt_rebinned3",
    "config_categories": ["sr"],
    "flow_strategy": "remove",
})
mlscore_sig_ggf_binary = default_unblind.derive("mlscore_sig_ggf_binary", cls_dict={
    "config_variable": lambda self, config_cat_inst: "mlscore.sig_ggf_binary",
    "config_categories": ["sr"],
    "flow_strategy": "remove",
})
mlscore_sig_vbf_binary = default_unblind.derive("mlscore_sig_vbf_binary", cls_dict={
    "config_variable": lambda self, config_cat_inst: "mlscore.sig_vbf_binary",
    "config_categories": ["sr"],
    "flow_strategy": "remove",
})
rebinlogit_mlscore_sig_ggf_binary = default_unblind.derive("rebinlogit_mlscore_sig_ggf_binary", cls_dict={
    "config_variable": lambda self, config_cat_inst: "rebinlogit_mlscore.sig_ggf_binary",
    "config_categories": ["sr"],
    "flow_strategy": "remove",
})
rebinlogit_mlscore_sig_vbf_binary = default_unblind.derive("rebinlogit_mlscore_sig_vbf_binary", cls_dict={
    "config_variable": lambda self, config_cat_inst: "rebinlogit_mlscore.sig_vbf_binary",
    "config_categories": ["sr"],
    "flow_strategy": "remove",
})

mll = default_unblind.derive("mll", cls_dict={
    "config_variable": lambda self, config_cat_inst: "mli_mll",
    "config_categories": ["incl"],
    "flow_strategy": "remove",
})


ll_pt_dycr = default_unblind.derive("ll_pt_dycr", cls_dict={
    "config_variable": lambda self, config_cat_inst: "mli_ll_pt",
    "config_categories": ["dycr"],
    "flow_strategy": "remove",
    "ml_model_name": [],
    "skip_ratify_shapes": True,
})
ll_pt_dycr_before = default_unblind.derive("ll_pt_dycr_before", cls_dict={
    "config_variable": lambda self, config_cat_inst: "mli_ll_pt",
    "config_categories": ["dycr"],
    "flow_strategy": "remove",
    "ml_model_name": [],
    "systematics": list(set(systematics.default) - {"dy_correction"}),
    "skip_ratify_shapes": True,
})
n_jet_dycr = default_unblind.derive("n_jet_dycr", cls_dict={
    "config_variable": lambda self, config_cat_inst: "mli_n_jet",
    "config_categories": ["dycr"],
    "flow_strategy": "remove",
    "ml_model_name": [],
    "skip_ratify_shapes": True,
})
n_jet_dycr_before = default_unblind.derive("n_jet_dycr_before", cls_dict={
    "config_variable": lambda self, config_cat_inst: "mli_n_jet",
    "config_categories": ["dycr"],
    "flow_strategy": "remove",
    "ml_model_name": [],
    "systematics": list(set(systematics.default) - {"dy_correction"}),
    "skip_ratify_shapes": True,
})


met40dnn_unblind = default_unblind.derive("met40dnn_unblind", cls_dict={
    "ml_model_name": ["multiclass_met40", "ggf_met40", "vbf_met40"],
})
vbftag_unblind = default_unblind.derive("vbftag_unblind", cls_dict={
    "ml_model_name": ["multiclassv3_tag", "ggfv3", "vbfv3_tag"],
})
vbftag1_unblind = default_unblind.derive("vbftag1_unblind", cls_dict={
    "ml_model_name": ["multiclassv3", "ggfv3", "vbfv3_tag"],
})
boosted_mergedbkg = default_unblind.derive("boosted_mergedbkg", cls_dict={
    "ml_model_name": ["multiclassv3", "ggfv3", "vbfv3_tag"],
    "config_categories": config_categories.default_boosted_mergedbkg,
})
boosted_bkg = default_unblind.derive("boosted_bkg", cls_dict={
    "ml_model_name": ["multiclassv3", "ggfv3", "vbfv3_tag"],
    "config_categories": config_categories.default_boosted_bkg,
})
incl_dycorr_boosted_bkg = default_unblind.derive("incl_dycorr_boosted_bkg", cls_dict={  # using outputs from Lara
    "ml_model_name": ["multiclassv3", "ggfv3", "vbfv3_tag"],
    "config_categories": config_categories.default_boosted_bkg,
})
pas = default_unblind.derive("pas", cls_dict={  # using outputs from Lara
    "ml_model_name": ["multiclassv3", "ggfv3", "vbfv3_tag"],
    "config_categories": config_categories.default_boosted_bkg,
})
vbftag1_noboosted_unblind = no_boosted.derive("vbftag1_noboosted_unblind", cls_dict={
    "ml_model_name": ["multiclassv3", "ggfv3", "vbfv3_tag"],
})
vbftag1_mergeboosted_unblind = default_unblind.derive("vbftag1_mergeboosted_unblind", cls_dict={
    "ml_model_name": ["multiclassv3", "ggfv3", "vbfv3_tag"],
    "config_categories": config_categories.sr_resolved + ["sr__boosted"] + config_categories.background_resolved,
})
vbfmqq_unblind = default_unblind.derive("vbfmqq_unblind", cls_dict={
    "ml_model_name": ["multiclassv3_mqq", "ggfv3", "vbfv3_mqq"],
})
vbfmqq1_unblind = default_unblind.derive("vbfmqq1_unblind", cls_dict={
    "ml_model_name": ["multiclassv3", "ggfv3", "vbfv3_mqq"],
})
vbfextended_unblind = default_unblind.derive("vbfextended_unblind", cls_dict={
    "ml_model_name": ["multiclassv3", "ggfv3", "vbfv3_vbf_extended"],
})
default_data = dl.derive("default_data", cls_dict={
    "systematics": systematics.default,
    "config_categories": config_categories.default_boosted,
    "skip_data": False,
})
vbftag_data = default_data.derive("vbftag_data", cls_dict={
    "ml_model_name": ["multiclassv3_tag", "ggfv3", "vbfv3_tag"],
})
vbfmqq_data = default_data.derive("vbfmqq_data", cls_dict={
    "ml_model_name": ["multiclassv3_mqq", "ggfv3", "vbfv3_mqq"],
})
vbfmqq1_data = default_data.derive("vbfmqq1_data", cls_dict={
    "ml_model_name": ["multiclassv3", "ggfv3", "vbfv3_mqq"],
})
vbftag1_data = default_data.derive("vbftag1_data", cls_dict={
    "ml_model_name": ["multiclassv3", "ggfv3", "vbfv3_tag"],
})
vbfextended_data = default_data.derive("vbfextended_data", cls_dict={
    "ml_model_name": ["multiclassv3", "ggfv3", "vbfv3_vbf_extended"],
})

vbfmqq_data_cpn_corr = default_data.derive("vbfmqq_data_cpn_corr", cls_dict={
    "systematics": systematics.default_cpn_corr,
    "ml_model_name": ["multiclassv3_mqq", "ggfv3", "vbfv3_mqq"],
})
vbfmqq_data_year_uncorr = default_data.derive("vbfmqq_data_year_uncorr", cls_dict={
    "systematics": systematics.default_year_uncorr,
    "ml_model_name": ["multiclassv3_mqq", "ggfv3", "vbfv3_mqq"],
})
mli_n_jet = default_data.derive("mli_n_jet", cls_dict={
    "processes": processes_dict["hww"],
    # "ml_model_name": [],
    "multi_variables": False,
    "config_categories": config_categories.no_nn_cats_with_boosted,
    "config_variable": lambda self, config_cat_inst: "mli_n_jet",
})
mli = default_data.derive("mli", cls_dict={
    "processes": processes_dict["hww"],
    # "ml_model_name": [],
    "multi_variables": True,
    "config_categories": config_categories.no_nn_cats_with_boosted,
    "config_variable": lambda self, config_cat_inst: mli_inputs,
})


#
# other inference models for testing and systematics studies
#

dl_kl1_dnn = dl.derive("dl_kl1_dnn", cls_dict={
    "config_categories": [
        "sr__1b__ml_hh_ggf_kl1_kt1",
        "sr__1b__ml_hh_vbf_kv1_k2v1_kl1",
        "sr__2b__ml_hh_ggf_kl1_kt1",
        "sr__2b__ml_hh_vbf_kv1_k2v1_kl1",
    ] + config_categories.background,
    "ml_model_name": ["multiclass_kl1", "ggf_kl1", "vbf_kl1"],
})
dl_boosted = dl.derive("dl_boosted", cls_dict={
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background,
})
dl_boosted_skip_dy10 = dl.derive("dl_boosted_skip_dy10", cls_dict={
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background,
    "skip_datasets": {"dy_hf": "dy_m10to50_amcatnlo", "dy_lf": "dy_m10to50_amcatnlo"},
})
dl_boosted_skip_dy = dl.derive("dl_boosted_skip_dy", cls_dict={
    "processes": processes_dict["hwwzztt_skip_dy"],
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background,
})
dl_boosted1 = dl.derive("dl_boosted1", cls_dict={
    "config_categories": config_categories.default_boosted,
})
dl1 = HBWInferenceModelBase.derive("dl1", cls_dict=default_cls_dict)
dl_skip_mc_stats = dl.derive("dl_skip_mc_stats", cls_dict={
    "mc_stats": False,
})
dl_skip_dy_m4to10 = dl.derive("dl_skip_dy_m4to10", cls_dict={
    "skip_datasets": {"dy": "dy_m4to10_amcatnlo"},
})
dl_test = HBWInferenceModelBase.derive("dl_test", cls_dict=default_cls_dict)
dl_bkg_cats = dl.derive("dl_bkg_cats", cls_dict={"config_categories": config_categories.background})
dl_syst = dl.derive("dl_syst", cls_dict={"systematics": systematics.shape})
dl_syst1 = dl.derive("dl_syst1", cls_dict={"systematics": systematics.shape})
dl_jerc_only = dl.derive("dl_jerc_only", cls_dict={"systematics": systematics.jerc_only})
dl_jerc = dl.derive("dl_jerc", cls_dict={"systematics": systematics.jerc})
dl_jerc_boosted = dl.derive("dl_jerc_boosted", cls_dict={
    "systematics": systematics.jerc,
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background,
})
dl_jerc1_boosted = dl.derive("dl_jerc1_boosted", cls_dict={
    "systematics": systematics.jerc1,
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background,
})
dl_jerc2_boosted = dl.derive("dl_jerc2_boosted", cls_dict={
    "systematics": systematics.jerc2,
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background,
})
dl_jerc3_boosted = dl.derive("dl_jerc3_boosted", cls_dict={
    "systematics": systematics.jerc3,
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background,
})
dl_jerc4_boosted = dl.derive("dl_jerc4_boosted", cls_dict={
    "systematics": systematics.jerc4,
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background,
})
dl_jerc4_boosted1 = dl.derive("dl_jerc4_boosted1", cls_dict={
    "systematics": systematics.jerc4,
    "config_categories": config_categories.default_boosted,
})
dl_jerc1_boosted_data = dl.derive("dl_jerc1_boosted_data", cls_dict={
    "systematics": systematics.jerc1,
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background,
    "skip_data": False,
})
dl_jerc3_boosted_data = dl.derive("dl_jerc3_boosted_data", cls_dict={
    "systematics": systematics.jerc3,
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background,
    "skip_data": False,
})
dl_jerc4_boosted_data = dl.derive("dl_jerc4_boosted_data", cls_dict={
    "systematics": systematics.jerc4,
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background,
    "skip_data": False,
})
dl_jerc3_boosted1_data = dl.derive("dl_jerc3_boosted1_data", cls_dict={
    "systematics": systematics.jerc3,
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background_split,  # noqa: E501
    "skip_data": False,
})
dl_jerc4_boosted1_data = dl.derive("dl_jerc4_boosted1_data", cls_dict={
    "systematics": systematics.jerc4,
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background_split,  # noqa: E501
    "skip_data": False,
})
dl_jerc_boosted_bjet_uncorr1 = dl.derive("dl_jerc_boosted_bjet_uncorr1", cls_dict={
    "systematics": systematics.jerc_bjet_uncorr1,
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background,
})
dl_jerc_boosted_bjet_uncorr1_data = dl.derive("dl_jerc_boosted_bjet_uncorr1_data", cls_dict={
    "systematics": systematics.jerc_bjet_uncorr1,
    "config_categories": config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background,
    "skip_data": False,
})
dl_jerc1 = dl.derive("dl_jerc1", cls_dict={"systematics": systematics.jerc1})
dl_jerc_bjet_uncorr = dl.derive("dl_jerc_bjet_uncorr", cls_dict={"systematics": systematics.jerc_bjet_uncorr})
dl_jerc_bjet_uncorr1 = dl.derive("dl_jerc_bjet_uncorr1", cls_dict={"systematics": systematics.jerc_bjet_uncorr1})
dl_data = dl.derive("dl_data", cls_dict={
    "config_categories": config_categories.background,
    "systematics": systematics.jerc,
    "skip_data": False,
})
dl_data_full = dl.derive("dl_data_full", cls_dict={
    # NOTE: needs to be run with --hist-hooks blind !!!!
    "config_categories": config_categories.default,
    "systematics": systematics.jerc,
    "skip_data": False,
})
dl_data_test = dl.derive("dl_data_test", cls_dict={
    "config_categories": config_categories.default,
    "systematics": systematics.rate,
    "skip_data": False,
})

# testing models
no_nn_cats = dl.derive("no_nn_cats", cls_dict={
    "processes": processes_dict["hwwzztt"],
    "systematics": systematics.jerc,
    "config_categories": config_categories.no_nn_cats,
})
bjet_incl = dl.derive("bjet_incl", cls_dict={
    "processes": processes_dict["hwwzztt"],
    "systematics": systematics.jerc,
    "config_categories": config_categories.bjet_incl,
})
test_pdf = dl.derive("test_pdf", cls_dict={
    "processes": processes_dict["test"],
    "systematics": systematics.rate + ["pdf_shape_tt"],
})
test_jec = dl.derive("test_jec", cls_dict={
    "processes": processes_dict["test"],
    "systematics": systematics.rate + ["jec_Total"],
})
only_sig = dl_jerc1_boosted_data.derive("only_sig", cls_dict={
    "processes": processes_dict["hww2l2nu"],
    "ml_model_name": [],
    "multi_variables": True,
    "config_categories": config_categories.no_nn_cats,
    "config_variable": lambda self, config_cat_inst: mli_inputs,
})
test = dl_jerc1_boosted_data.derive("test", cls_dict={
    "processes": processes_dict["hww2l2nu"],
    "ml_model_name": [],
    "multi_variables": True,
    "config_categories": config_categories.no_nn_cats,
    "config_variable": lambda self, config_cat_inst: mli_inputs,
    "systematics": [],
})
dl_mli_inputs = dl_jerc1_boosted_data.derive("dl_mli_inputs", cls_dict={
    "processes": processes_dict["hww2l2nu"],
    "ml_model_name": [],
    "multi_variables": True,
    "config_categories": config_categories.no_nn_cats,
    "config_variable": lambda self, config_cat_inst: mli_inputs,
})
