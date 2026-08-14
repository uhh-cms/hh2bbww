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
    "eq2b": [
        "sr__resolved__2b__ml_tt_ml",
        "sr__resolved__2b__ml_st",
        "sr__resolved__2b__ml_dy",
        "sr__resolved__2b__ml_tth",
        "sr__resolved__2b__ml_tthh_4b",
        "sr__resolved__2b__ml_hhh_signal",
    ],
    "eq3b": [
        "sr__resolved__3b__ml_tt_ml",
        "sr__resolved__3b__ml_tth",
        "sr__resolved__3b__ml_tthh_4b",
        "sr__resolved__3b__ml_hhh_signal",
    ],
    "geq4b": [
        "sr__resolved__4b__ml_tt_ml",
        "sr__resolved__4b__ml_tth",
        "sr__resolved__4b__ml_tthh_4b",
        "sr__resolved__4b__ml_hhh_signal",
    ],
    "v0": [
        "sr__resolved__2b__ml_tt_ml",
        "sr__resolved__2b__ml_tth",
        "sr__resolved__2b__ml_tthh_4b",
        "sr__resolved__2b__ml_hhh_signal",
        "sr__resolved__3b__ml_tt_ml",
        "sr__resolved__3b__ml_tth",
        "sr__resolved__3b__ml_tthh_4b",
        "sr__resolved__3b__ml_hhh_signal",
        "sr__resolved__4b__ml_tt_ml",
        "sr__resolved__4b__ml_tth",
        "sr__resolved__4b__ml_tthh_4b",
        "sr__resolved__4b__ml_hhh_signal",
    ],
    "test_2b_1l": [
        "sr__resolved__2b_1l__ml_tt_ml",
        "sr__resolved__2b_1l__ml_st",
        "sr__resolved__2b_1l__ml_dy",
        "sr__resolved__2b_1l__ml_tth",
        "sr__resolved__2b_1l__ml_tthh_4b",
        "sr__resolved__2b_1l__ml_hhh_signal",
    ],
    "test_1b_1tb": [
        "sr__resolved__1b_1tb__ml_tt_ml",
        "sr__resolved__1b_1tb__ml_st",
        "sr__resolved__1b_1tb__ml_dy",
        "sr__resolved__1b_1tb__ml_tth",
        "sr__resolved__1b_1tb__ml_tthh_4b",
        "sr__resolved__1b_1tb__ml_hhh_signal",
    ],
    "bcats": [
        "sr__2b",
        "sr__3b",
        "sr__4b",
    ],
    "boosted": [
        "sr__boosted",
        "sr__boosted__ml_hhh_signal",
    ],
    "boosted_low": [
        "sr__boosted_low",
        "sr__boosted_low__ml_hhh_signal",
    ],
    "boosted_loose": [
        "sr__boosted_loose",
        "sr__boosted_loose__ml_hhh_signal",
    ],
})

systematics = DotDict({
    "lumi": [
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
    "rates": [
        "rate_ttbar",
        "rate_ttbar_b",
        "rate_ttbar_bb",
        "rate_dy",
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
systematics["hhh_shape_ffn"] = [
    *systematics.lumi,
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
    "h_ggf", "h_vbf", "wh", "tth",  # "zh_gg","zh"
    "ttvh",  # "thq", "thw",
    "tttt",
    "ttvv",
    "hh_ggf", "hh_vbf",
    "vhh_4b", "tthh_4b",
    # TODO: add bbh
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


default_cls_dict = {
    "ml_model_name": ml_model_name,
    "processes": processes_dict["v0"],
    "config_categories": config_categories.v0,
    "systematics": systematics.default,
    "config_variable": config_variable_hhh,
    "mc_stats": True,
    "skip_data": True,
}

dl_trih = HBWInferenceModelBase.derive("dl_trih", cls_dict=default_cls_dict)

#
# current inference models
#

rate_only_hhh_v2 = dl_trih.derive("rate_only_hhh_v2", cls_dict={
    "systematics": systematics.default,
    "config_categories": config_categories.v0,
    "ml_model_name": ["multiclass_hhh_v2", "hhh_v2"],
    "config_variable": config_variable_hhh,
    "processes": processes_dict["v0"],
})

# ----------------------- BASELINE INFERENCE MODELS SPLIT IN BJET CAT ------------------------------

Cat_eq2b_Bin_V5_shape = dl_trih.derive("Cat_eq2b_Bin_V5_shape", cls_dict={
    "systematics": systematics.hhh_shape_ffn,
    "config_categories": config_categories.eq2b,
    "ml_model_name": ["Cat_eq2b_V1", "Bin_V1"],
    "config_variable": config_variable_hhh,
    "processes": processes_dict["v0"],
})
Cat_eq2b_Bin_V5_shape_unblind = Cat_eq2b_Bin_V5_shape.derive("Cat_eq2b_Bin_V5_shape_unblind", cls_dict={
    "unblind": False,
    "skip_data": False,
})
Cat_boosted_Bin_V5_shape_unblind = Cat_eq2b_Bin_V5_shape.derive("Cat_boosted_Bin_V5_shape_unblind", cls_dict={
    "unblind": False,
    "config_categories": config_categories.boosted,
    "skip_data": False,
})
Cat_boosted_Bin_V5_shape = Cat_eq2b_Bin_V5_shape.derive("Cat_boosted_Bin_V5_shape", cls_dict={
    "config_categories": config_categories.boosted,
})
Cat_boosted_Bin_V5_gatja = Cat_eq2b_Bin_V5_shape.derive("Cat_boosted_Bin_V5_gatja", cls_dict={
    "config_categories": config_categories.boosted,
    "ml_model_name": ["Gatja_Cat_eq3b_V3", "Gatja_Bin_V3"],
})
Cat_boosted_low_Bin_V5_shape = Cat_eq2b_Bin_V5_shape.derive("Cat_boosted_low_Bin_V5_shape", cls_dict={
    "config_categories": config_categories.boosted_low,
    "systematics": systematics.default,
})
Cat_boosted_loose_Bin_V5_shape = Cat_eq2b_Bin_V5_shape.derive("Cat_boosted_loose_Bin_V5_shape", cls_dict={
    "config_categories": config_categories.boosted_loose,
    "systematics": systematics.default,
})
Cat_2b_1l_Bin_V5_shape = Cat_eq2b_Bin_V5_shape.derive("Cat_2b_1l_Bin_V5_shape", cls_dict={
    "config_categories": config_categories.test_2b_1l,
    "systematics": systematics.default,
})
Cat_1b_1tb_Bin_V5_shape = Cat_eq2b_Bin_V5_shape.derive("Cat_1b_1tb_Bin_V5_shape", cls_dict={
    "config_categories": config_categories.test_1b_1tb,
    "systematics": systematics.default,
})
Cat_eq3b_Bin_V5_shape = dl_trih.derive("Cat_eq3b_Bin_V5_shape", cls_dict={
    "systematics": systematics.hhh_shape_ffn,
    "config_categories": config_categories.eq3b,
    "ml_model_name": ["Cat_eq3b_V1", "Bin_V1"],
    "config_variable": config_variable_hhh,
    "processes": processes_dict["v0"],
})
Cat_eq3b_Bin_V5_gatja = Cat_eq3b_Bin_V5_shape.derive("Cat_eq3b_Bin_V5_gatja", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Gatja_Cat_eq3b_V3", "Gatja_Bin_V3"],
})
Cat_eq3b_Bin_V5_shape_unblind = Cat_eq3b_Bin_V5_shape.derive("Cat_eq3b_Bin_V5_shape_unblind", cls_dict={
    "unblind": False,
    "skip_data": False,
})
Cat_geq4b_Bin_V5_shape = dl_trih.derive("Cat_geq4b_Bin_V5_shape", cls_dict={
    "systematics": systematics.hhh_shape_ffn,
    "config_categories": config_categories.geq4b,
    "ml_model_name": ["Cat_geq4b_V1", "Bin_V1"],
    "config_variable": config_variable_hhh,
    "processes": processes_dict["v0"],
})
Cat_geq4b_Bin_V5_gatja = Cat_geq4b_Bin_V5_shape.derive("Cat_geq4b_Bin_V5_gatja", cls_dict={
    "systematics": systematics.default,
    "ml_model_name": ["Gatja_Cat_geq4b_V3", "Gatja_Bin_V3"],
})
Cat_geq4b_Bin_V5_shape_unblind = Cat_geq4b_Bin_V5_shape.derive("Cat_geq4b_Bin_V5_shape_unblind", cls_dict={
    "unblind": False,
    "skip_data": False,
})
