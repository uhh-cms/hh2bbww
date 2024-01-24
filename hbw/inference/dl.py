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
    "ggHH_kl_0_kt_1_dl_hbbhww",
    "ggHH_kl_1_kt_1_dl_hbbhww",
    "ggHH_kl_2p45_kt_1_dl_hbbhww",
    "ggHH_kl_5_kt_1_dl_hbbhww",
    "tt",
    # "ttv", "ttvv",
    "st_schannel", "st_tchannel", "st_twchannel",
    "dy_lep",
    "w_lnu",
    # "vv",
    # "vvv",
    # "qcd",
    # "ggZH", "tHq", "tHW", "ggH", "qqH", "ZH", "WH", "VH", "ttH", "bbH",
]

# All categories to be included in the final datacard
config_categories = [
    "2e__ml_t_bkg",
    "2mu__ml_t_bkg",
    "emu__ml_t_bkg",
    # "2e__2b__ml_t_bkg",
    # "2mu__2b__ml_t_bkg",
    # "emu__2b__ml_t_bkg",
    "2e__ml_sig",
    "2mu__ml_sig",
    "emu__ml_sig",
    # "2e__2b__ml_sig",
    # "2mu__2b__ml_sig",
    # "emu__2b__ml_sig",
    "2e__ml_v_lep",
    "2mu__ml_v_lep",
    "emu__ml_v_lep",
    # "2e__2b__ml_v_lep",
    # "2mu__2b__ml_v_lep",
    # "emu__2b__ml_v_lep",
    # "2e__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    # "2e__ml_qqHH_CV_1_C2V_1_kl_1_dl_hbbhww",
    # "2e__ml_tt",
    # "2e__ml_t_bkg",
    # "2e__ml_st",
    # "2e__ml_sig",
    # "2e__ml_v_lep",
    # "2mu__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    # "2mu__ml_qqHH_CV_1_C2V_1_kl_1_dl_hbbhww",
    # "2mu__ml_tt",
    # "2mu__ml_tt_bkg",
    # "2mu__ml_st",
    # "2mu__ml_sig",
    # "2mu__ml_v_lep",
]

rate_systematics = [
    # Lumi: should automatically choose viable uncertainties based on campaign
    "lumi_13TeV_2016",
    "lumi_13TeV_2017",
    "lumi_13TeV_1718",
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
    "QCDScale_ggHH",  # should be included in inference model (THU_HH)
    "QCDScale_qqHH",
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
    "pdf_Higgs_ggHH",
    "pdf_Higgs_qqHH",
    "pdf_VHH",
    "pdf_ttHH",
]

shape_systematics = [
    # Shape Scale uncertainties
    # "murf_envelope_ggHH_kl_1_kt_1_dl_hbbhww",
    "murf_envelope_tt",
    "murf_envelope_st_schannel",
    "murf_envelope_st_tchannel",
    "murf_envelope_st_twchannel",
    "murf_envelope_dy_lep",
    "murf_envelope_w_lnu",
    "murf_envelope_ttV",
    "murf_envelope_VV",
    # Shape PDF Uncertainties
    "pdf_shape_tt",
    "pdf_shape_st_schannel",
    "pdf_shape_st_tchannel",
    "pdf_shape_st_twchannel",
    "pdf_shape_dy_lep",
    "pdf_shape_w_lnu",
    "pdf_shape_ttV",
    "pdf_shape_VV",
    # Scale Factors (TODO)
    "btag_hf",
    "btag_lf",
    "btag_hfstats1_2017",
    "btag_hfstats2_2017"
    "btag_lfstats1_2017"
    "btag_lfstats2_2017"
    "btag_cferr1",
    "btag_cferr2",
    "mu_sf",
    # "mu_trig",
    "e_sf",
    # "e_trig",
    # "minbias_xs",
    # "top_pt",
]


# All systematics to be included in the final datacard

systematics = rate_systematics  # + shape_systematics

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
    # "ggHH_kl_0_kt_1_dl_hbbhww",
    "ggHH_kl_1_kt_1_dl_hbbhww",
    # "ggHH_kl_2p45_kt_1_dl_hbbhww",
    # "ggHH_kl_5_kt_1_dl_hbbhww",
    "tt",
]

cls_dict["config_categories"] = [
    "2e__1b",
    "2mu__1b",
    "emu__1b",
    "2e__2b",
    "2mu__2b",
    "emu__2b",
    # "2e__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    # "2e__ml_tt",
]

cls_dict["systematics"] = [
    "lumi_13TeV_2017",
]

# cls_dict["ml_model_name"] = "dense_test_dl"
cls_dict["ml_model_name"] = None

# minimal model for quick test purposes
dl_test = dl.derive("dl_test", cls_dict=cls_dict)

cls_dict = default_cls_dict.copy()

cls_dict["config_categories"] = [
    #"2e__1b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    "2e__1b__ml_tt",
    "2e__1b__ml_st",
    #"2mu__1b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    "emu__1b__ml_tt",
    "emu__1b__ml_st",
    #"emu__1b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    "2mu__1b__ml_tt",
    "2mu__1b__ml_st",
    #"2e__2b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    "2e__2b__ml_tt",
    "2e__2b__ml_st",
    #"2mu__2b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    "emu__2b__ml_tt",
    "emu__2b__ml_st",
    #"emu__2b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    "2mu__2b__ml_tt",
    "2mu__2b__ml_st",
    #"2e__1b__ml_ggHH_kl_0_kt_1_dl_hbbhww",
    "2e__1b__ml_w_lnu",
    "2e__1b__ml_dy_lep",
    #"2mu__1b__ml_ggHH_kl_0_kt_1_dl_hbbhww",
    "emu__1b__ml_w_lnu",
    "emu__1b__ml_dy_lep",
    #"emu__1b__ml_ggHH_kl_0_kt_1_dl_hbbhww",
    "2mu__1b__ml_w_lnu",
    "2mu__1b__ml_dy_lep",
    #"2e__2b__ml_ggHH_kl_0_kt_1_dl_hbbhww",
    "2e__2b__ml_w_lnu",
    "2e__2b__ml_dy_lep",
    #"2mu__2b__ml_ggHH_kl_0_kt_1_dl_hbbhww",
    "emu__2b__ml_w_lnu",
    "emu__2b__ml_dy_lep",
    #"emu__2b__ml_ggHH_kl_0_kt_1_dl_hbbhww",
    "2mu__2b__ml_w_lnu",
    "2mu__2b__ml_dy_lep",
    "2e__1b__ml_ggHH_sig_all",
    "2mu__1b__ml_ggHH_sig_all",
    "emu__1b__ml_ggHH_sig_all",
    "2e__2b__ml_ggHH_sig_all",
    "emu__2b__ml_ggHH_sig_all",
    "2mu__2b__ml_ggHH_sig_all",
    #"2e__1b__ml_ggHH_kl_5_kt_1_dl_hbbhww",
    #"2mu__1b__ml_ggHH_kl_5_kt_1_dl_hbbhww",
    #"emu__1b__ml_ggHH_kl_5_kt_1_dl_hbbhww",
    #"2e__2b__ml_ggHH_kl_5_kt_1_dl_hbbhww",
    #"emu__2b__ml_ggHH_kl_5_kt_1_dl_hbbhww",
    #"2mu__2b__ml_ggHH_kl_5_kt_1_dl_hbbhww",
    #"2e__1b__ml_ggHH_kl_2p45_kt_1_dl_hbbhww",
    #"2mu__1b__ml_ggHH_kl_2p45_kt_1_dl_hbbhww",
    #"emu__1b__ml_ggHH_kl_2p45_kt_1_dl_hbbhww",
    #"2e__2b__ml_ggHH_kl_2p45_kt_1_dl_hbbhww",
    #"emu__2b__ml_ggHH_kl_2p45_kt_1_dl_hbbhww",
    #"2mu__2b__ml_ggHH_kl_2p45_kt_1_dl_hbbhww",
]

cls_dict["ml_model_name"] = "dense_test_aachen_dl"

# minimal model for quick test purposes
dl_test_aachen = dl.derive("dl_test_aachen", cls_dict=cls_dict)

cls_dict = default_cls_dict.copy()

# cls_dict["processes"] = [
#    # "sig_all",
#    "ggHH_kl_1_kt_1_dl_hbbhww",
#    "v_lep",
#    "t_bkg",
# ]

cls_dict["config_categories"] = [
    "2e__1b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    "2e__1b__ml_v_lep",
    "2e__1b__ml_t_bkg",
    "2mu__1b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    "emu__1b__ml_v_lep",
    "emu__1b__ml_t_bkg",
    "emu__1b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    "2mu__1b__ml_v_lep",
    "2mu__1b__ml_t_bkg",
    "2e__2b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    "2e__2b__ml_v_lep",
    "2e__2b__ml_t_bkg",
    "2mu__2b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    "emu__2b__ml_v_lep",
    "emu__2b__ml_t_bkg",
    "emu__2b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
    "2mu__2b__ml_v_lep",
    "2mu__2b__ml_t_bkg",
]

cls_dict["ml_model_name"] = "dense_test_kl1_dl"

# minimal model for quick test purposes
dl_test_kl1 = dl.derive("dl_test_kl1", cls_dict=cls_dict)

cls_dict = default_cls_dict.copy()

# cls_dict["processes"] = [
#    "ggHH_sig",
#    "v_lep",
#    "t_bkg",
# ]

cls_dict["config_categories"] = [
    "2e__1b__ml_ggHH_sig",
    "2e__1b__ml_v_lep",
    "2e__1b__ml_t_bkg",
    "emu__1b__ml_ggHH_sig",
    "emu__1b__ml_v_lep",
    "emu__1b__ml_t_bkg",
    "2mu__1b__ml_ggHH_sig",
    "2mu__1b__ml_v_lep",
    "2mu__1b__ml_t_bkg",
    "2e__2b__ml_ggHH_sig",
    "2e__2b__ml_v_lep",
    "2e__2b__ml_t_bkg",
    "emu__2b__ml_ggHH_sig",
    "emu__2b__ml_v_lep",
    "emu__2b__ml_t_bkg",
    "2mu__2b__ml_ggHH_sig",
    "2mu__2b__ml_v_lep",
    "2mu__2b__ml_t_bkg",
]

cls_dict["ml_model_name"] = "dense_test_sig_dl"

# minimal model for quick test purposes
dl_test_sig = dl.derive("dl_test_sig", cls_dict=cls_dict)

cls_dict = default_cls_dict.copy()

# cls_dict["processes"] = [
#    "sig_all",
#    "ggHH_kl_1_kt_1_dl_hbbhww",
#    "v_lep",
#    "t_bkg",
# ]

cls_dict["config_categories"] = [
    "2e__1b__ml_ggHH_sig_all",
    "2e__1b__ml_v_lep",
    "2e__1b__ml_t_bkg",
    "emu__1b__ml_ggHH_sig_all",
    "emu__1b__ml_v_lep",
    "emu__1b__ml_t_bkg",
    "2mu__1b__ml_ggHH_sig_all",
    "2mu__1b__ml_v_lep",
    "2mu__1b__ml_t_bkg",
    "2e__2b__ml_ggHH_sig_all",
    "2e__2b__ml_v_lep",
    "2e__2b__ml_t_bkg",
    "emu__2b__ml_ggHH_sig_all",
    "emu__2b__ml_v_lep",
    "emu__2b__ml_t_bkg",
    "2mu__2b__ml_ggHH_sig_all",
    "2mu__2b__ml_v_lep",
    "2mu__2b__ml_t_bkg",
]

cls_dict["ml_model_name"] = "dense_test_sig_all_dl"

# minimal model for quick test purposes
dl_test_sig_all = dl.derive("dl_test_sig_all", cls_dict=cls_dict)

# Inference model not on ML score but on hardcoded variable (bb_pt)
cls_dict = default_cls_dict.copy()

cls_dict["config_categories"] = [
    "2e__1b",
    "2mu__1b",
    "emu__1b",
    "2e__2b",
    "2mu__2b",
    "emu__2b",
]

cls_dict["ml_model_name"] = None

# minimal model for quick test purposes
dl_noML = dl.derive("dl_noML", cls_dict=cls_dict)
