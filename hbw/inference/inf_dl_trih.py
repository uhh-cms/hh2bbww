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
ml_model_name = ["multiclass_hhh_v0", "hhh_v0"]
if use_old_version:
    ml_model_name = ["multiclassv1", "ggfv1", "vbfv1"]

# All categories to be included in the final datacard
config_categories = DotDict({
    "hhh_v0": [
        "sr__resolved__2b__ml_tt_custom",
        "sr__resolved__2b__ml_ttbb_custom",
        "sr__resolved__2b__ml_st",
        "sr__resolved__2b__ml_dy",
        "sr__resolved__2b__ml_h",
        "sr__resolved__2b__ml_tthh_4b",
        "sr__resolved__2b__ml_hhh_4b2w_2l2nu_c30_d40",
        "sr__resolved__3b__ml_tt_custom",
        "sr__resolved__3b__ml_ttbb_custom",
        "sr__resolved__3b__ml_st",
        "sr__resolved__3b__ml_dy",
        "sr__resolved__3b__ml_h",
        "sr__resolved__3b__ml_tthh_4b",
        "sr__resolved__3b__ml_hhh_4b2w_2l2nu_c30_d40",
        "sr__resolved__4b__ml_tt_custom",
        "sr__resolved__4b__ml_ttbb_custom",
        "sr__resolved__4b__ml_st",
        "sr__resolved__4b__ml_dy",
        "sr__resolved__4b__ml_h",
        "sr__resolved__4b__ml_tthh_4b",
        "sr__resolved__4b__ml_hhh_4b2w_2l2nu_c30_d40",
    ],
    "hhh_v8": [
        "sr__resolved__2b__ml_tt_ml",
        # "sr__resolved__2b__ml_ttbb_custom",
        "sr__resolved__2b__ml_st",
        "sr__resolved__2b__ml_dy",
        "sr__resolved__2b__ml_h",
        "sr__resolved__2b__ml_tthh_4b",
        "sr__resolved__2b__ml_hhh_4b2w_2l2nu_c30_d40",
        "sr__resolved__3b__ml_tt_ml",
        # "sr__resolved__3b__ml_ttbb_custom",
        "sr__resolved__3b__ml_st",
        "sr__resolved__3b__ml_dy",
        "sr__resolved__3b__ml_h",
        "sr__resolved__3b__ml_tthh_4b",
        "sr__resolved__3b__ml_hhh_4b2w_2l2nu_c30_d40",
        "sr__resolved__4b__ml_tt_ml",
        # "sr__resolved__4b__ml_ttbb_custom",
        "sr__resolved__4b__ml_st",
        "sr__resolved__4b__ml_dy",
        "sr__resolved__4b__ml_h",
        "sr__resolved__4b__ml_tthh_4b",
        "sr__resolved__4b__ml_hhh_4b2w_2l2nu_c30_d40",
    ],
    "no_nn_cats": [
        "sr__1b",
        "sr__2b",
    ],
})
# config_categories.default_boosted = (
#     config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background_resolved
# )
# config_categories.default_boosted_mergedbkg = (
#     config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background
# )
# config_categories.default_boosted_bkg = (
#     config_categories.sr_resolved + config_categories.sr_boosted + config_categories.background_resolved +
#     ["sr__boosted__ml_bkg"]
# )


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
        "rate_ttbb"
        "rate_st",
        "rate_dy_lf",
        "rate_dy_hf",
    ],
    "rate_unconstrained3": [
        "rate_ttbar",
        "rate_ttbb",
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
systematics["shape_only_cpn_uncorr"] = [
    *systematics.murf_envelope,
    *systematics.pdf_shape,
    *systematics.btag_cpn_uncorr,
    *systematics.experiment_cpn_uncorr,
    *systematics.other,
]
# default set of all systematics
systematics["default"] = [
    *systematics.rate_default,
    *systematics.shape_only_cpn_uncorr,
    *systematics.jerc_only_cpn_uncorr,
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
hhhprocs = [
    "hhh_4b2w_2l2nu_c30_d40",
    "hhh_4b2w_2l2nu_c30_d499",
    "hhh_4b2w_2l2nu_c30_d4m1",
    "hhh_4b2w_2l2nu_c319_d419",
    "hhh_4b2w_2l2nu_c31_d40",
    "hhh_4b2w_2l2nu_c31_d42",
    "hhh_4b2w_2l2nu_c32_d4m1",
    "hhh_4b2w_2l2nu_c34_d49",
    "hhh_4b2w_2l2nu_c3m1_d40",
    "hhh_4b2w_2l2nu_c3m1_d4m1",
    "hhh_4b2w_2l2nu_c3m1p5_d4m0p5",
]

backgrounds_hhh_v0 = [
    # "st_tchannel",
    "st_twchannel",
    "st_schannel",
    "tt_custom", "ttbb_custom",
    "ttw",
    "ttz",
    "dy",
    "w_lnu",
    "vv",
    "vvv",
    "h_ggf", "h_vbf", "wh", "tth",  # "zh_gg","zh"
    "ttvh",  # "thq", "thw",
    # "tttt",
    "ttvv",
    # "hh_ggf", "hh_vbf",
    "vhh_4b", "tthh_4b",
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
    "hhh_v0": [*backgrounds_hhh_v0, "hhh_4b2w_2l2nu_c30_d40"],
}

from hbw.ml.derived.ml_dl_trih import input_features
mli_inputs = input_features.hhh_v0


def config_variable_hhh(self, config_cat_inst):
    """
    Function to set the config variable for the binary model.
    """

    # Super unnötig atm... well
    if "sig_hhh" in config_cat_inst.name:
        return "logit_mlscore.hhh_4b2w_2l2nu_c30_d40"
    elif config_cat_inst.x.root_cats.get("dnn"):
        # since we merge into 1 bin anyways, we can use either score
        return "logit_mlscore.hhh_4b2w_2l2nu_c30_d40"
    else:
        # raise ValueError(f"Category {config_cat_inst.name} is not a DNN category.")
        logger.warning(
            f"Category {config_cat_inst.name} is not a DNN category, using binary classifier score.",
        )
        return "logit_mlscore.sig_ggf_binary"


default_cls_dict = {
    "ml_model_name": ml_model_name,
    "processes": processes_dict["hhh_v0"],
    "config_categories": config_categories.hhh_v0,
    "systematics": systematics.rate_default,
    "config_variable": config_variable_hhh,
    "mc_stats": True,
    "skip_data": True,
}

dl = HBWInferenceModelBase.derive("dl", cls_dict=default_cls_dict)

#
# currently "final" inference models
#

rate_only_hhh_v2 = dl.derive("rate_only_hhh_v2", cls_dict={
    "systematics": systematics.rate_default,
    "config_categories": config_categories.hhh_v0,
    "ml_model_name": ["multiclass_hhh_v2", "hhh_v2"],
    "config_variable": config_variable_hhh,
    "processes": processes_dict["hhh_v0"],
})
rate_only_hhh_v8 = dl.derive("rate_only_hhh_v8", cls_dict={
    "systematics": systematics.rate_default,
    "config_categories": config_categories.hhh_v8,
    "ml_model_name": ["multiclass_hhh_v8", "hhh_v2"],
    "config_variable": config_variable_hhh,
    "processes": processes_dict["hhh_v0"],
})
