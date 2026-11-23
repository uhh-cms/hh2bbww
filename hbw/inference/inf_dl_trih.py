# coding: utf-8

"""
hbw(dl) inference model.
"""

from hbw.ml.derived.ml_dl_trih import input_features
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
    "bkg_3b": [
        "sr__3b__ml_tt_custom",
        "sr__3b__ml_ttbb_custom",
        "sr__3b__ml_tth",
        "sr__3b__ml_tthh",
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
    "bkg_4b": [
        "sr__4b__ml_ttbb_custom",
        "sr__4b__ml_tth",
        "sr__4b__ml_tthh",
    ],
    "incl": [
        "sr__4b__ml_ttbb_custom",
        "sr__4b__ml_tth",
        "sr__4b__ml_tthh",
        "sr__4b__ml_hhh_signal",
        "sr__3b__ml_tt_custom",
        "sr__3b__ml_ttbb_custom",
        "sr__3b__ml_tth",
        "sr__3b__ml_tthh",
        "sr__3b__ml_hhh_signal",
    ],
    "boosted": [
        "sr__boosted_glo",
    ],
})

systematics = DotDict({
    "lumi": [
        "lumi_13p6TeV_2024",
        "lumi_13p6TeV_2025",
        "lumi_13p6TeV_correlated",
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
        "btag_uncorrelated_light_{campaign}",
        "btag_statistic_bc_{campaign}",
        "btag_uncorrelated_light",
        "btag_uncorrelated_bc",
        "mu_id_sf_{campaign}",
        "mu_iso_sf_{campaign}",
        "e_sf_{campaign}",
        "e_reco_sf_{campaign}",
        "trigger_sf_{campaign}",
        "minbias_xs",  # do not decorrelate PU between campaigns
        # TODO add scale and smearing
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
    *systematics.experiment_cpn_uncorr,
    *systematics.QCDscale,
    *systematics.pdf,
    *systematics.pdf_shape,
    *systematics.BR,
    *systematics.rates,
    *systematics.murf_envelope,
    *systematics.other,
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
    # "ttw",
    # "ttz",
    "w_lnu",
    "dy",
    "vv",
    "vvv",
    "h_ggf", "h_vbf", "tth", "zh_gg", "zh", "wh",
    "ttvh", "thq", "thw", "bbh",
    "tttt",
    "ttvv",
    "hh_ggf", "hh_vbf",
    "vhh_4b", "tthh",
    "hh_ggf_hbb_hww_kl1_kt1",
    "hh_vbf_hbb_hww_kv1_k2v1_kl1",
]
backgrounds_v1 = [
    # "st_tchannel",  # I have a problem with this one
    "st_twchannel",
    "st_schannel",
    "tt_custom",
    "ttb_custom", "tt_bb_custom", "tt2b_custom",
    # "ttw",
    # "ttz",
    "dy",
    "vv",
    "vvv",
    "h_ggf", "h_vbf", "tth", "zh_gg", "zh", "wh",
    "ttvh", "thq", "thw", "bbh",
    "tttt",
    "ttvv",
    "hh_ggf", "hh_vbf",
    "vhh_4b", "tthh",
    "hh_ggf_hbb_hww_kl1_kt1",
    "hh_vbf_hbb_hww_kv1_k2v1_kl1",
]

processes_dict = {
    "hhh_SM": [*backgrounds_v0, "hhh_4b2w_2l2nu_c30_d40"],
    "v0": [*backgrounds_v0, *hhhprocs],
    "v1": [*backgrounds_v1, *hhhprocs],
}

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
    elif "tthh" in config_cat_inst.name:
        return "logit_mlscore.tthh"
    elif "tth" in config_cat_inst.name:
        return "logit_mlscore.tth"
    elif "tt_custom" in config_cat_inst.name:
        return "logit_mlscore.tt_custom"
    elif "ttbb_custom" in config_cat_inst.name:
        return "logit_mlscore.ttbb_custom"
    else:
        # raise ValueError(f"Category {config_cat_inst.name} is not a DNN category.")
        logger.warning(
            f"Category {config_cat_inst.name} is not a DNN category, using binary classifier score.",
        )
        return "logit_mlscore.hhh_signal"


def config_variable_cat_by_gatja(self, config_cat_inst):
    """
    Function to set the config variable for the binary model.
    """
    # Super unnötig atm... well
    if "hhh_signal" in config_cat_inst.name:
        return "mlscore.sig_hhh_binary"
    # elif config_cat_inst.x.root_cats.get("tthh"):
    elif "tthh" in config_cat_inst.name:
        return "mlscore.tthh"
    elif "tth" in config_cat_inst.name:
        return "mlscore.tth"
    elif "tt_custom" in config_cat_inst.name:
        return "mlscore.tt_custom"
    elif "ttbb_custom" in config_cat_inst.name:
        return "mlscore.ttbb_custom"
    else:
        # raise ValueError(f"Category {config_cat_inst.name} is not a DNN category.")
        logger.warning(
            f"Category {config_cat_inst.name} is not a DNN category, using binary classifier score.",
        )
        return "mlscore.sig_hhh_binary"


default_cls_dict = {
    "ml_model_name": ml_model_name,
    "processes": processes_dict["v0"],
    "config_categories": config_categories.incl_3b,
    "systematics": systematics.default,
    "config_variable": config_variable_hhh,
    "mc_stats": True,
    "skip_data": True,
}

hhh = HBWInferenceModelBase.derive("hhh", cls_dict=default_cls_dict)

# ----------------------- BASELINE INFERENCE MODELS ------------------------------------------------------------

# ####################### Baseline Rate unc only / Asimov fits ##################################################
incl = hhh.derive("incl", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_incl_best", "Bin_V2"],
    "config_categories": config_categories.incl,
})
incl_Bin_V3 = hhh.derive("incl_Bin_V3", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_incl_best", "Bin_V3"],
    "config_categories": config_categories.incl,
})
incl_part_unblind = hhh.derive("incl_part_unblind", cls_dict={
    "systematics": systematics.hhh_shape_ffn_full,
    "ml_model_name": ["Cat_incl_best", "Bin_V3"],
    "config_categories": config_categories.incl,
})
incl_bin_best = hhh.derive("incl_bin_best", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_incl_best", "Bin_V2_best"],
    "config_categories": config_categories.incl,
})
gatja2_3b = hhh.derive("gatja2_3b", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_3b", "Bin"],
    "config_categories": config_categories.incl_3b,
})
gatja2_4b = gatja2_3b.derive("gatja2_4b", cls_dict={
    "ml_model_name": ["Cat_4b", "Bin"],
    "config_categories": config_categories.incl_4b,
})
Bin_V3_3b = hhh.derive("Bin_V3_3b", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_3b", "Bin_V3"],
    "config_categories": config_categories.incl_3b,
})
test_3b = hhh.derive("test_3b", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_3b", "Bin_V3"],
    "config_categories": config_categories.incl_3b,
})
test_rate_tt_3b = hhh.derive("test_rate_tt_3b", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_3b_test_tt_rate", "Bin_V3_test_tt_rate"],
    "config_categories": config_categories.incl_3b,
})
opt_3b = hhh.derive("opt_3b", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_3b_opt_gatja", "Bin_V2_opt_gatja"],
    "config_categories": config_categories.incl_3b,
})
Bin_V3_4b = Bin_V3_3b.derive("Bin_V3_4b", cls_dict={
    "ml_model_name": ["Cat_4b", "Bin_V3"],
    "config_categories": config_categories.incl_4b,
})
test_4b = test_3b.derive("test_4b", cls_dict={
    "ml_model_name": ["Cat_4b", "Bin_V3"],
    "config_categories": config_categories.incl_4b,
})
test_rate_tt_4b = test_3b.derive("test_rate_tt_4b", cls_dict={
    "ml_model_name": ["Cat_4b_test_tt_rate", "Bin_V3_test_tt_rate"],
    "config_categories": config_categories.incl_4b,
})
opt_4b = opt_3b.derive("opt_4b", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_4b_opt_gatja", "Bin_V2_opt_gatja"],
    "config_categories": config_categories.incl_4b,
})
without_gatja_3b = hhh.derive("without_gatja_3b", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_3b_without_gatja", "Bin_V2_without_gatja"],
    "config_categories": config_categories.incl_3b,
})
without_gatja_4b = without_gatja_3b.derive("without_gatja_4b", cls_dict={
    "ml_model_name": ["Cat_4b_without_gatja", "Bin_V2_without_gatja"],
    "config_categories": config_categories.incl_4b,
})
gatjacf_3b = hhh.derive("gatjacf_3b", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_3b_weighted_V2", "Bin_V2"],
    "config_categories": config_categories.incl_3b,
})
gatja_3b_best = hhh.derive("gatja_3b_best", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_3b_best", "Bin_V2"],
    "config_categories": config_categories.incl_3b,
})
Bin_V3_3b_best = hhh.derive("Bin_V3_3b_best", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_3b_best", "Bin_V3"],
    "config_categories": config_categories.incl_3b,
})
bin_3b_best = hhh.derive("bin_3b_best", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Cat_3b_best", "Bin_V2_best"],
    "config_categories": config_categories.incl_3b,
})
gatjacf_4b = gatjacf_3b.derive("gatjacf_4b", cls_dict={
    "ml_model_name": ["Cat_4b_weighted_V2", "Bin_V2"],
    "config_categories": config_categories.incl_4b,
})
gatjacf_4b_best = gatjacf_3b.derive("gatjacf_4b_best", cls_dict={
    "ml_model_name": ["Cat_4b_best", "Bin_V2"],
    "config_categories": config_categories.incl_4b,
})
bin_4b_best = gatjacf_3b.derive("bin_4b_best", cls_dict={
    "ml_model_name": ["Cat_4b_best", "Bin_V2_best"],
    "config_categories": config_categories.incl_4b,
})
Bin_V3_4b_best = gatjacf_3b.derive("Bin_V3_4b_best", cls_dict={
    "ml_model_name": ["Cat_4b_best", "Bin_V3"],
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
gatja_3b_shape = hhh.derive("gatja_3b_shape", cls_dict={
    "systematics": systematics.hhh_shape_ffn_full,
    "ml_model_name": ["Cat_3b", "Bin_V3"],
    "config_categories": config_categories.incl_3b,
})
gatja_4b_shape = gatja_3b_shape.derive("gatja_4b_shape", cls_dict={
    "ml_model_name": ["Cat_4b", "Bin_V3"],
    "config_categories": config_categories.incl_4b,
})
# ####################### Baseline with shape unc (excluding JEC/JER) - partially unblinded fits ####################
gatja2_part_unblind_3b = gatja_3b_shape.derive("gatja2_part_unblind_3b", cls_dict={
    "unblind": False,
    "skip_data": False,
})
gatja2_part_unblind_4b = gatja_3b_shape.derive("gatja2_part_unblind_4b", cls_dict={
    "ml_model_name": ["Cat_4b", "Bin_V3"],
    "config_categories": config_categories.incl_4b,
    "unblind": False,
    "skip_data": False,
})

part_unblind_3b = gatja_3b_shape.derive("part_unblind_3b", cls_dict={
    "unblind": False,
    "skip_data": False,
})
part_unblind_4b = gatja_3b_shape.derive("part_unblind_4b", cls_dict={
    "ml_model_name": ["Cat_4b", "Bin_V3"],
    "config_categories": config_categories.incl_4b,
    "unblind": False,
    "skip_data": False,
})

part_unblind_3b_test_rate_tt = test_rate_tt_3b.derive("part_unblind_3b_test_rate_tt", cls_dict={
    "systematics": systematics.hhh_shape_ffn_full,
    "unblind": False,
    "skip_data": False,
})
part_unblind_4b_test_rate_tt = test_rate_tt_4b.derive("part_unblind_4b_test_rate_tt", cls_dict={
    "systematics": systematics.hhh_shape_ffn_full,
    "config_categories": config_categories.incl_4b,
    "unblind": False,
    "skip_data": False,
})

part_unblind_3b_ratify = gatja_3b_shape.derive("part_unblind_3b_ratify", cls_dict={
    "unblind": False,
    "skip_data": False,
    "skip_ratify_shapes": False,
})
part_unblind_4b_ratify = gatja_3b_shape.derive("part_unblind_4b_ratify", cls_dict={
    "ml_model_name": ["Cat_4b", "Bin_V3"],
    "config_categories": config_categories.incl_4b,
    "unblind": False,
    "skip_data": False,
    "skip_ratify_shapes": False,
})

part_unblind_3b_without_gatja = gatja_3b_shape.derive("part_unblind_3b_without_gatja", cls_dict={
    "ml_model_name": ["Cat_3b_without_gatja", "Bin_V2_without_gatja"],
    "unblind": False,
    "skip_data": False,
})
part_unblind_4b_without_gatja = gatja_3b_shape.derive("part_unblind_4b_without_gatja", cls_dict={
    "ml_model_name": ["Cat_4b_without_gatja", "Bin_V2_without_gatja"],
    "config_categories": config_categories.incl_4b,
    "unblind": False,
    "skip_data": False,
})
part_unblind_3b_reduced_gatja = gatja_3b_shape.derive("part_unblind_3b_reduced_gatja", cls_dict={
    "ml_model_name": ["Cat_3b_opt_gatja", "Bin_V2_opt_gatja"],
    "unblind": False,
    "skip_data": False,
})
part_unblind_4b_reduced_gatja = gatja_3b_shape.derive("part_unblind_4b_reduced_gatja", cls_dict={
    "ml_model_name": ["Cat_4b_opt_gatja", "Bin_V2_opt_gatja"],
    "config_categories": config_categories.incl_4b,
    "unblind": False,
    "skip_data": False,
})


# ---------------------- Inference models for testing the NN startegy --------------------------------------------------

baseline_hh = hhh.derive("baseline_hh", cls_dict={
    "ml_model_name": ["baseline_hh", "baseline_hh_bin"],
    "config_categories": config_categories.incl,
    "systematics": systematics.default,
})

baseline_hhh_incl = hhh.derive("baseline_hhh_incl", cls_dict={
    "ml_model_name": ["baseline_hhh_incl", "Bin"],
    "config_categories": config_categories.incl,
    "systematics": systematics.default,
})

baseline_hh_3b = hhh.derive("baseline_hh_3b", cls_dict={
    "ml_model_name": ["baseline_cat3b", "baseline_hh_bin"],
    "config_categories": config_categories.incl_3b,
    "systematics": systematics.default,
})
baseline_hh_4b = hhh.derive("baseline_hh_4b", cls_dict={
    "ml_model_name": ["baseline_cat4b", "baseline_hh_bin"],
    "config_categories": config_categories.incl_4b,
    "systematics": systematics.default,
})

staged = hhh.derive("staged", cls_dict={
    "ml_model_name": ["staged_cat", "staged_bin"],
    "config_categories": config_categories.incl,
    "systematics": systematics.default,
})

hhh_wo_bin_3b = Bin_V3_3b.derive("hhh_wo_bin_3b", cls_dict={
    "config_variable": config_variable_hhh_wo_bin,
})
hhh_wo_bin_4b = Bin_V3_4b.derive("hhh_wo_bin_4b", cls_dict={
    "config_variable": config_variable_hhh_wo_bin,
})

cat_by_gatja_3b = hhh.derive("cat_by_gatja_3b", cls_dict={
    "ml_model_name": ["cat_by_gatja_3b"],
    "config_categories": config_categories.bkg_3b,
    "systematics": systematics.default,
    "config_variable": config_variable_cat_by_gatja,
})
cat_by_gatja_4b = hhh.derive("cat_by_gatja_4b", cls_dict={
    "ml_model_name": ["cat_by_gatja_4b"],
    "config_categories": config_categories.bkg_4b,
    "systematics": systematics.default,
    "config_variable": config_variable_cat_by_gatja,
})
cat_by_gatja_bin = hhh.derive("cat_by_gatja_bin", cls_dict={
    "ml_model_name": ["cat_by_gatja_bin"],
    "config_categories": ["sr"],
    "systematics": systematics.default,
    "config_variable": config_variable_cat_by_gatja,
})

# ---------------------- Inference models for testing rate param freely floating --------------------------------------------------  # noqa: E501

discrete_b_score_sum_V2 = gatja_3b_shape.derive("discrete_b_score_sum_V2", cls_dict={
    "systematics": systematics.default,
    "unblind": False,
    "skip_data": False,
    "config_categories": ["sr"],
    "config_variable": lambda self, config_cat_inst: "mli_discrete_b_score_sum",
})
discrete_b_score_sum_cr = gatja_3b_shape.derive("discrete_b_score_sum_cr", cls_dict={
    "unblind": False,
    "skip_data": False,
    "config_categories": ["ttvr"],
    "config_variable": lambda self, config_cat_inst: "mli_discrete_b_score_sum",
})

ttcr_emu_2b = gatja_3b_shape.derive("ttcr_emu_2b", cls_dict={
    "unblind": False,
    "skip_data": False,
    "ml_model_name": None,
    "config_categories": ["ttcr__emu__2b"],
    "config_variable": lambda self, config_cat_inst: "n_btag",
})

n_btag = gatja_3b_shape.derive("n_btag", cls_dict={
    "unblind": False,
    "skip_data": False,
    "ml_model_name": None,
    "config_categories": ["sr"],
    "config_variable": lambda self, config_cat_inst: "n_btag",
})

n_btag_ttcr_emu = gatja_3b_shape.derive("n_btag_ttcr_emu", cls_dict={
    "unblind": False,
    "skip_data": False,
    "ml_model_name": None,
    "config_categories": ["ttcr__emu"],
    "config_variable": lambda self, config_cat_inst: "n_btag",
})
n_btag_ttcr_emu_rate = gatja_3b_shape.derive("n_btag_ttcr_emu_rate", cls_dict={
    "unblind": False,
    "skip_data": False,
    "ml_model_name": None,
    "config_categories": ["ttcr__emu"],
    "config_variable": lambda self, config_cat_inst: "n_btag",
    "systematics": systematics.default,
    "processes": processes_dict["v1"],
})
