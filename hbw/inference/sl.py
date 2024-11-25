# coding: utf-8

"""
hbw inference model.
"""

import hbw.inference.constants as const  # noqa
from hbw.inference.base import HBWInferenceModelBase


#
# Defaults for all the Inference Model parameters
#

# used to set default requirements for cf.CreateDatacards based on the config
ml_model_name = "dense_default"

# default_producers = [f"ml_{ml_model_name}", "event_weights"]

# All processes to be included in the final datacard
processes = [
    "hh_ggf_hbb_hvvqqlnu_kl0_kt1",
    "hh_ggf_hbb_hvvqqlnu_kl1_kt1",
    "hh_ggf_hbb_hvvqqlnu_kl2p45_kt1",
    "hh_ggf_hbb_hvvqqlnu_kl5_kt1",
    "tt",
    # "ttv", "ttvv",
    "st_schannel", "st_tchannel", "st_twchannel",
    "dy",
    "w_lnu",
    # "vv",
    # "vvv",
    # "qcd",
    # "ggZH", "tHq", "tHW", "ggH", "qqH", "ZH", "WH", "VH", "ttH", "bbH",
]

# All config categories to be included in the final datacard
config_categories = [
    "sr__1e__1b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
    "sr__1e__2b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
    "sr__1mu__1b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
    "sr__1mu__2b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
    "sr__1e__ml_hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1",
    "sr__1e__ml_tt",
    "sr__1e__ml_st",
    "sr__1e__ml_v_lep",
    "sr__1mu__ml_hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1",
    "sr__1mu__ml_tt",
    "sr__1mu__ml_st",
    "sr__1mu__ml_v_lep",
]

rate_systematics = [
    # Lumi: should automatically choose viable uncertainties based on campaign
    "lumi_13TeV_2016",
    "lumi_13TeV_2017",
    "lumi_13TeV_1718",
    "lumi_13TeV_2022",
    "lumi_13TeV_2023",
    "lumi_13TeV_correlated",
    # Rate QCDScale uncertainties
    "QCDScale_ttbar",
    "QCDScale_V",
    "QCDScale_VV",
    "QCDScale_VVV",
    "QCDScale_ggH",
    "QCDScale_qqH",
    "QCDScale_VH",
    "QCDScale_ttH",
    "QCDScale_bbH",
    "QCDScale_hh_ggf",  # should be included in inference model (THU_HH)
    "QCDScale_hh_vbf",
    "QCDScale_VHH",
    "QCDScale_ttHH",
    # Rate PDF uncertainties
    "pdf_gg",
    "pdf_qqbar",
    "pdf_qg",
    "pdf_Higgs_gg",
    "pdf_Higgs_qqbar",
    "pdf_Higgs_qg",  # none so far
    "pdf_Higgs_ttH",
    "pdf_Higgs_bbH",  # removed
    "pdf_Higgs_hh_ggf",
    "pdf_Higgs_hh_vbf",
    "pdf_VHH",
    "pdf_ttHH",
]

shape_systematics = [
    # Shape Scale uncertainties
    # "murf_envelope_hh_ggf_hbb_hvv2l2nu_kl1_kt1",
    "murf_envelope_tt",
    "murf_envelope_st",
    "murf_envelope_dy",
    "murf_envelope_w_lnu",
    "murf_envelope_ttV",
    "murf_envelope_VV",
    # Shape PDF Uncertainties
    "pdf_shape_tt",
    "pdf_shape_st",
    "pdf_shape_dy",
    "pdf_shape_w_lnu",
    "pdf_shape_ttV",
    "pdf_shape_VV",
    # Scale Factors (TODO)
    "btag_hf",
    "btag_lf",
    "btag_hfstats1_{year}",
    "btag_hfstats2_{year}",
    "btag_lfstats1_{year}",
    "btag_lfstats2_{year}",
    "btag_cferr1",
    "btag_cferr2",
    "mu_id_sf",
    "mu_iso_sf",
    # "mu_trig_sf",
    "e_sf",
    # "e_trig_sf",
    "minbias_xs",
    "top_pt",
]

# All systematics to be included in the final datacard
systematics = rate_systematics + shape_systematics

default_cls_dict = {
    "ml_model_name": ml_model_name,
    "processes": processes,
    "config_categories": config_categories,
    "systematics": systematics,
    "mc_stats": True,
    "skip_data": True,
}

default = HBWInferenceModelBase.derive("default", cls_dict=default_cls_dict)

#
# derive some additional Inference Models
#

# inference model with only rate uncertainties
sl_rates_only = default.derive("rates_only", cls_dict={"systematics": rate_systematics})


# minimal model for quick test purposes
cls_dict_test = {
    "ml_model_name": "dense_22post_test",
    "processes": [
        "hh_ggf_hbb_hvvqqlnu_kl1_kt1",
        "tt",
    ],
    "config_categories": [
        "sr__1e__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
        "sr__1e__ml_tt",
    ],
    "systematics": [
        "lumi_13TeV_2016",
        "lumi_13TeV_2017",
        "lumi_13TeV_1718",
        "lumi_13TeV_2022",
        "lumi_13TeV_2023",
        "lumi_13TeV_correlated",
    ],
}
sl_22post_test = default.derive("sl_22post_test", cls_dict=cls_dict_test)

# model but with different fit variable
jet1_pt = default.derive("jet1_pt", cls_dict={
    "ml_model_name": None,
    "config_variable": lambda config_cat_inst: "jet1_pt",
})


#
# 2022
#

processes_22 = [
    "hh_ggf_hbb_hvvqqlnu_kl1_kt1",
    "tt",
    # "st_schannel",
    "st_tchannel", "st_twchannel",
    "dy",
    "w_lnu",
]

config_categories_22 = [
    # Signal regions
    "sr__1e__1b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
    "sr__1e__2b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
    "sr__1mu__1b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
    "sr__1mu__2b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
    # Background regions
    "sr__1e__ml_tt",
    "sr__1e__ml_st",
    "sr__1e__ml_v_lep",
    "sr__1mu__ml_tt",
    "sr__1mu__ml_st",
    "sr__1mu__ml_v_lep",
]

sl_22 = default.derive("sl_22", cls_dict={
    "dummy_kl_variation": True,
    "processes": processes_22,
    "config_categories": config_categories_22,
    "ml_model_name": "dense_22",
    "systematics": rate_systematics,
})
sl_22_shapes = sl_22.derive("sl_22_shapes", cls_dict={
    "systematics": rate_systematics + shape_systematics,
})
