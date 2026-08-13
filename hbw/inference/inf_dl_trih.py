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
    "incl_3b": [
        "sr__3b__ml_tt_custom",
        "sr__3b__ml_ttbb_custom",
        "sr__3b__ml_tth",
        "sr__3b__ml_tthh",
        "sr__3b__ml_hhh_signal",
    ],
    "resolved_3b": [
        "sr__resolved_glo__3b__ml_tt_custom",
        "sr__resolved_glo__3b__ml_ttbb_custom",
        "sr__resolved_glo__3b__ml_tth",
        "sr__resolved_glo__3b__ml_tthh",
        "sr__resolved_glo__3b__ml_hhh_signal",
    ],
    "incl_4b": [
        "sr__4b__ml_ttbb_custom",
        "sr__4b__ml_tth",
        "sr__4b__ml_tthh",
        "sr__4b__ml_hhh_signal",
    ],
    "boosted": [
        "sr__boosted_glo",
    ],
})

systematics = DotDict({
    "lumi": [
        "lumi_13p6TeV_2024",
    ],
    "QCDscale": [
        "QCDscale_ttbar",
        "QCDscale_ttbb",
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
    "rates": [
        "rate_ttbar",
        "rate_ttbar_b",
        "rate_ttbar_bb",
    ],
    "rates_bjet": [
        "rate_ttbar_3b",
        "rate_ttbar_b_{bjet_cat}",
        "rate_ttbar_bb_4b",
    ],
    "murf_envelope": [
        "murf_envelope_ttbar",
        "murf_envelope_ttbb",
        "murf_envelope_st",
        "murf_envelope_dy",
        # "murf_envelope_w",
        "murf_envelope_ttV",  # TODO: ttW has no murf/pdf weights
        "murf_envelope_VV",
        "murf_envelope_H",
        "murf_envelope_hh_ggf_hbb_hww",
        "murf_envelope_hh_ggf_hbb_hzz",
        "murf_envelope_hh_ggf_hbb_htt",
    ],
    "pdf_shape": [
        "pdf_shape_ttbar",
        "pdf_shape_ttbb",
        "pdf_shape_st",
        "pdf_shape_dy",
        # "pdf_shape_w",
        "pdf_shape_ttV",  # TODO: ttW has no murf/pdf weights
        "pdf_shape_VV",
        "pdf_shape_H",
        "pdf_shape_hh_ggf_hbb_hww",
        "pdf_shape_hh_ggf_hbb_hzz",
        "pdf_shape_hh_ggf_hbb_htt",
    ],
    "btag_22_23": [
        "btag_hf",
        "btag_lf",
        "btag_hfstats1_{campaign}",
        "btag_hfstats2_{campaign}",
        "btag_lfstats1_{campaign}",
        "btag_lfstats2_{campaign}",
        "btag_cferr1",
        "btag_cferr2",
    ],
    "btag_24": [
        "btag_fsrdef_bc",
        "btag_isrdef_bc",
        "btag_hdamp_bc",
        "btag_jer_bc",
        "btag_jes_bc",
        "btag_mass_bc",
        "btag_statistic_bc",
        "btag_tune_bc",
        "btag_correlated_light",
        "btag_uncorrelated_light",
    ],
    "btag_short": [
        "btag_bc",
        "btag_light",
    ],
    "btag_short_bjet": [
        "btag_bc_{bjet_cat}",
        "btag_light_{bjet_cat}",
    ],
    "experiment": [
        "mu_id_sf",
        "mu_iso_sf",
        "e_sf",
        "e_reco_sf",
        "trigger_sf",
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
        "isr_ttbar",
        "isr_ttbb",
        "isr_V",
        "isr_ttV",
        "isr_VV",
        "isr_st",
        "isr_H",
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
systematics["default"] = [
    *systematics.lumi,
    *systematics.QCDscale,
    *systematics.pdf,
    *systematics.BR,
    *systematics.rates,
]
systematics["hhh_shape_ffn_full"] = [
    *systematics.lumi,
    *systematics.experiment,
    *systematics.QCDscale,
    *systematics.pdf,
    *systematics.BR,
    *systematics.rates,
    *systematics.murf_envelope,
    *systematics.other,
    *systematics.btag_24,
]
hhhprocs = [
    "hhh_4b2w_2l2nu_c30_d40",
    "hhh_4b2w_2l2nu_c30_d499",
    "hhh_4b2w_2l2nu_c319_d419",
    "hhh_4b2w_2l2nu_c31_d40",
    "hhh_4b2w_2l2nu_c31_d42",
    "hhh_4b2w_2l2nu_c32_d4m1",
    "hhh_4b2w_2l2nu_c34_d49",
    "hhh_4b2w_2l2nu_c3m1_d40",
    "hhh_4b2w_2l2nu_c3m1_d4m1",
    "hhh_4b2w_2l2nu_c3m1p5_d4m0p5",
]
backgrounds_v0 = [
    # "st_tchannel",  # I have a problem with this one
    "st_twchannel",
    "st_schannel",
    "tt_custom",
    "ttb_custom", "tt_bb_custom", "tt2b_custom",
    "ttw",
    "ttz",
    "w_lnu",
    "dy",
    "vv",
    "vvv",
    "h_ggf", "h_vbf", "wh", "tth", "zh_gg", "zh",
    "ttvh", "thq", "thw", "bbh",
    "tttt",
    "ttvv",
    "hh_ggf", "hh_vbf",
    "vhh_4b", "tthh",
    "hh_ggf_hbb_hww_kl0_kt1",
    "hh_ggf_hbb_hww_kl1_kt1",
    "hh_ggf_hbb_hww_kl2p45_kt1",
    "hh_ggf_hbb_hww_kl5_kt1",
    "hh_vbf_hbb_hww_kv1p74_k2v1p37_kl14p4",
    "hh_vbf_hbb_hww_kvm0p758_k2v1p44_klm19p3",
    "hh_vbf_hbb_hww_kvm0p012_k2v0p03_kl10p2",
    "hh_vbf_hbb_hww_kv2p12_k2v3p87_klm5p96",
    "hh_vbf_hbb_hww_kv1_k2v1_kl1",
    "hh_vbf_hbb_hww_kv1_k2v0_kl1",  # missing bbtt sample
    "hh_vbf_hbb_hww_kvm0p962_k2v0p959_klm1p43",
    "hh_vbf_hbb_hww_kvm1p21_k2v1p94_klm0p94",
    "hh_vbf_hbb_hww_kvm1p6_k2v2p72_klm1p36",
    "hh_vbf_hbb_hww_kvm1p83_k2v3p57_klm3p39",  # missing bbtt sample
]

processes_dict = {
    "hhh_SM": [*backgrounds_v0, "hhh_4b2w_2l2nu_c30_d40"],
    "v0": [*backgrounds_v0, *hhhprocs],
}

from hbw.ml.derived.ml_dl_trih import input_features
mli_inputs = input_features.expanded_hhh_inputs


def config_variable_hhh(self, config_cat_inst):
    """
    Function to set the config variable for the binary model.
    """

    # Super unnötig atm... well
    if "sig_hhh" in config_cat_inst.name:
        return "logit_mlscore.sig_hhh_binary"
    elif config_cat_inst.x.root_cats.get("dnn"):
        # since we merge into 1 bin anyways, we can use either score
        return "logit_mlscore.sig_hhh_binary"
    else:
        # raise ValueError(f"Category {config_cat_inst.name} is not a DNN category.")
        logger.warning(
            f"Category {config_cat_inst.name} is not a DNN category, using binary classifier score.",
        )
        return "logit_mlscore.sig_hhh_binary"


def config_variable_hhh_wo_bin(self, config_cat_inst):
    """
    Function to set the config variable for the binary model.
    """
    # Super unnötig atm... well
    if "hhh_signal" in config_cat_inst.name:
        return "logit_mlscore.hhh_signal"
    elif config_cat_inst.x.root_cats.get("tthh"):
        return "logit_mlscore.tthh"
    elif config_cat_inst.x.root_cats.get("tth"):
        return "logit_mlscore.tth"
    elif config_cat_inst.x.root_cats.get("tt_custom"):
        return "logit_mlscore.tt_custom"
    elif config_cat_inst.x.root_cats.get("ttbb_custom"):
        return "logit_mlscore.ttbb_custom"
    else:
        # raise ValueError(f"Category {config_cat_inst.name} is not a DNN category.")
        logger.warning(
            f"Category {config_cat_inst.name} is not a DNN category, using binary classifier score.",
        )
        return "logit_mlscore.hhh_signal"


default_cls_dict = {
    "ml_model_name": ml_model_name,
    "processes": processes_dict["v0"],
    "config_categories": config_categories.v0,
    "systematics": systematics.default,
    "config_variable": config_variable_hhh,
    "mc_stats": True,
    "skip_data": True,
}

hhh = HBWInferenceModelBase.derive("hhh", cls_dict=default_cls_dict)

# ----------------------- BASELINE INFERENCE MODELS ------------------------------------------------------------

# ####################### Baseline Rate unc only / Asimov fits ##################################################
gatja2_3b = hhh.derive("gatja2_3b", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_3b", "Bin"],
    "config_categories": config_categories.incl_3b,
})
gatja2_4b = gatja2_3b.derive("gatja2_4b", cls_dict={
    "ml_model_name": ["Cat_4b", "Bin"],
    "config_categories": config_categories.incl_4b,
})
gatja2_boosted = gatja2_3b.derive("gatja2_boosted", cls_dict={
    "ml_model_name": ["Cat_3b", "Bin"],
    "config_categories": config_categories.boosted,
})
gatja2_resolved_3b = gatja2_3b.derive("gatja2_resolved_3b", cls_dict={
    "ml_model_name": ["Cat_3b", "Bin"],
    "config_categories": config_categories.resolved_3b,
})

# ####################### Baseline with shape unc (excluding JEC/JER) - Asimov fits #################################
gatja2_3b_shape = hhh.derive("gatja2_3b_shape", cls_dict={
    "systematics": systematics.hhh_shape_ffn_full,
    "ml_model_name": ["Cat_3b", "Bin"],
    "config_categories": config_categories.incl_3b,
})
gatja2_4b_shape = gatja2_3b_shape.derive("gatja2_4b_shape", cls_dict={
    "ml_model_name": ["Cat_4b", "Bin"],
    "config_categories": config_categories.incl_4b,
})
# ####################### Baseline with shape unc (excluding JEC/JER) - partially unblinded fits ####################
gatja2_part_unblind = gatja2_3b_shape.derive("gatja2_part_unblind", cls_dict={
    "unblind": False,
    "skip_data": False,
})
gatja2_unblind4 = gatja2_3b_shape.derive("gatja2_unblind4", cls_dict={
    "ml_model_name": ["Cat_4b", "Bin"],
    "config_categories": config_categories.incl_4b,
})
