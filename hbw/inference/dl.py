# coding: utf-8

"""
hbw(dl) inference model.
"""

import hbw.inference.constants as const  # noqa
from hbw.inference.base import HBWInferenceModelBase


#
# Defaults for all the Inference Model parameters
#

# used to set default requirements for cf.CreateDatacards based on the config
ml_model_name = "dense_default_dl"
# default_producers = [f"ml_{ml_model_name}", "event_weights"]

# All processes to be included in the final datacard
processes = [
    "hh_ggf_kl0_kt1_hbb_hvv2l2nu",
    "hh_ggf_kl1_kt1_hbb_hvv2l2nu",
    "hh_ggf_kl2p45_kt1_hbb_hvv2l2nu",
    "hh_ggf_kl5_kt1_hbb_hvv2l2nu",
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

# All categories to be included in the final datacard
config_categories = [
    "sr__2e__ml_hh_ggf_kl1_kt1_hbb_hvv2l2nu",
    "sr__2e__ml_hh_vbf_kv1_k2v1_kl1_hbb_hvv2l2nu",
    "sr__2e__ml_tt",
    "sr__2e__ml_t_bkg",
    "sr__2e__ml_st",
    "sr__2e__ml_sig",
    "sr__2e__ml_v_lep",
    "sr__2mu__ml_hh_ggf_kl1_kt1_hbb_hvv2l2nu",
    "sr__2mu__ml_hh_vbf_kv1_k2v1_kl1_hbb_hvv2l2nu",
    "sr__2mu__ml_tt",
    "sr__2mu__ml_tt_bkg",
    "sr__2mu__ml_st",
    "sr__2mu__ml_sig",
    "sr__2mu__ml_v_lep",
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
    # "murf_envelope_hh_ggf_kl1_kt1_hbb_hvv2l2nu",
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

dl = HBWInferenceModelBase.derive("dl", cls_dict=default_cls_dict)

cls_dict = default_cls_dict.copy()

cls_dict["processes"] = [
    # "hh_ggf_kl0_kt1_hbb_hvv2l2nu",
    "hh_ggf_kl1_kt1_hbb_hvv2l2nu",
    # "hh_ggf_kl2p45_kt1_hbb_hvv2l2nu",
    # "hh_ggf_kl5_kt1_hbb_hvv2l2nu",
    "tt",
]

cls_dict["config_categories"] = [
    "2e__ml_hh_ggf_kl1_kt1_hbb_hvv2l2nu",
    "2e__ml_tt",
]

cls_dict["systematics"] = [
    "lumi_13TeV_2017",
]

cls_dict["ml_model_name"] = "dense_test_dl"

# minimal model for quick test purposes
dl_test = dl.derive("dl_test", cls_dict=cls_dict)

#
# 2022
#

processes_22 = [
    "hh_ggf_kl1_kt1_hbb_hvv2l2nu",
    "tt",
    # "st_schannel",
    "st_tchannel", "st_twchannel",
    "dy",
    "w_lnu",
]

config_categories_22 = [
    # Signal regions
    "sr__2e__1b__ml_hh_ggf_kl1_kt1_hbb_hvv2l2nu",
    "sr__2e__2b__ml_hh_ggf_kl1_kt1_hbb_hvv2l2nu",
    "sr__2mu__1b__ml_hh_ggf_kl1_kt1_hbb_hvv2l2nu",
    "sr__2mu__2b__ml_hh_ggf_kl1_kt1_hbb_hvv2l2nu",
    "sr__emu__1b__ml_hh_ggf_kl1_kt1_hbb_hvv2l2nu",
    "sr__emu__2b__ml_hh_ggf_kl1_kt1_hbb_hvv2l2nu",
    # Background regions
    "sr__2e__ml_tt",
    "sr__2e__ml_st",
    "sr__2e__ml_dy",
    "sr__2mu__ml_tt",
    "sr__2mu__ml_st",
    "sr__2mu__ml_dy",
    "sr__emu__ml_tt",
    "sr__emu__ml_st",
    "sr__emu__ml_dy",
]

dl_22 = dl.derive("dl_22", cls_dict={
    "dummy_kl_variation": True,
    "processes": processes_22,
    "config_categories": config_categories_22,
    "ml_model_name": "dl_22",
    "systematics": rate_systematics,
})
dl_22_shapes = dl_22.derive("dl_22_shapes", cls_dict={
    "systematics": rate_systematics + shape_systematics,
})
dl_22_test = dl.derive("dl_22_test", cls_dict={
    "processes": [
        "hh_ggf_kl1_kt1_hbb_hvv2l2nu",
        "st_tchannel",
    ],
    "config_categories": [
        "sr__2mu__2b__ml_hh_ggf_kl1_kt1_hbb_hvv2l2nu",
        "sr__2mu__2b__ml_tt",
        "sr__2mu__2b__ml_st",
    ],
    "ml_model_name": "dl_22",
})
