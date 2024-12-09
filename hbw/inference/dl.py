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
ml_model_name = "dl_22_procs1_w0_inp1"
# default_producers = [f"ml_{ml_model_name}", "event_weights"]

# All processes to be included in the final datacard
processes = [
    "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
    "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
    "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
    "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
    # "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",  # TODO
    "tt",
    # "ttv",  # TODO
    # "ttvv",  # TODO
    # "st_schannel",  # Not datasets anyways
    # TODO: merge st_schannel, st_tchannel
    "st_tchannel",
    "st_twchannel",
    "dy",
    "w_lnu",
    "vv",
    # "vvv",  # TODO
    "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    # TODO: add thq, thw, bbh
    # "qcd",  # probably not needed
]

# All categories to be included in the final datacard
config_categories = [
    "sr__1b__ml_hh_ggf_hbb_hvv2l2nu_kl1_kt1",
    "sr__1b__ml_hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
    "sr__1b__ml_tt",
    "sr__1b__ml_st",
    "sr__1b__ml_dy",
    "sr__1b__ml_h",
    "sr__2b__ml_hh_ggf_hbb_hvv2l2nu_kl1_kt1",
    "sr__2b__ml_hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
    "sr__2b__ml_tt",
    "sr__2b__ml_st",
    "sr__2b__ml_dy",
    "sr__2b__ml_h",
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
    # "murf_envelope_ttv",
    # "murf_envelope_vv",
    # "murf_envelope_h",
    # Shape PDF Uncertainties
    "pdf_shape_tt",
    "pdf_shape_st",
    "pdf_shape_dy",
    "pdf_shape_w_lnu",
    # "pdf_shape_ttv",
    # "pdf_shape_vv",
    # "pdf_shape_h",
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

# "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
# "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1",
# "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43",
# "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94",
# "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36",
# "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39",
# "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
# "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
# "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
# "hh_ggf_hbb_hvv2l2nu_kl5_kt1",


dl.derive("dl_ml_study_1", cls_dict={
    "ml_model_name": "dl_22post_ml_study_1",
    "config_categories": [
        "sr__1b__ml_signal_ggf",
        "sr__1b__ml_signal_vbf",
        "sr__1b__ml_tt",
        "sr__1b__ml_st",
        "sr__1b__ml_dy",
        "sr__1b__ml_h",
        "sr__2b__ml_signal_ggf",
        "sr__2b__ml_signal_vbf",
        "sr__2b__ml_tt",
        "sr__2b__ml_st",
        "sr__2b__ml_dy",
        "sr__2b__ml_h",
    ],
    "processes": [
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1",
        "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43",
        "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94",
        "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36",
        "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39",
        "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
        "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
        "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
        "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
        "tt",
        "dy",
        "w_lnu",
        "vv",
        "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    ],
    "systematics": rate_systematics,
})

dl.derive("dl_ml_study_2", cls_dict={
    "ml_model_name": "dl_22post_ml_study_2",
    "config_categories": [
        "sr__1b__ml_signal_ggf2",
        "sr__1b__ml_signal_vbf2",
        "sr__1b__ml_tt",
        "sr__1b__ml_st",
        "sr__1b__ml_dy",
        "sr__1b__ml_h",
        "sr__2b__ml_signal_ggf2",
        "sr__2b__ml_signal_vbf2",
        "sr__2b__ml_tt",
        "sr__2b__ml_st",
        "sr__2b__ml_dy",
        "sr__2b__ml_h",
    ],
    "processes": [
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1",
        "hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43",
        "hh_vbf_hbb_hvv2l2nu_kvm1p21_k2v1p94_klm0p94",
        "hh_vbf_hbb_hvv2l2nu_kvm1p6_k2v2p72_klm1p36",
        "hh_vbf_hbb_hvv2l2nu_kvm1p83_k2v3p57_klm3p39",
        "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
        "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
        "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
        "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
        "tt",
        "dy",
        "w_lnu",
        "vv",
        "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    ],
    "systematics": rate_systematics,
})

dl.derive("dl_hww_and_hzz", cls_dict={
    "processes": [
        "hh_ggf_hbb_hww_kl0_kt1",
        "hh_ggf_hbb_hww_kl1_kt1",
        "hh_ggf_hbb_hww_kl2p45_kt1",
        "hh_ggf_hbb_hww_kl5_kt1",
        "hh_ggf_hbb_hzz_kl0_kt1",
        "hh_ggf_hbb_hzz_kl1_kt1",
        "hh_ggf_hbb_hzz_kl2p45_kt1",
        "hh_ggf_hbb_hzz_kl5_kt1",
        "tt",
        "dy",
        "w_lnu",
        "vv",
        "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    ],
})
dl.derive("dl_lch", cls_dict={
    "processes": [
        "hh_ggf_hbb_hww_kl0_kt1",
        "hh_ggf_hbb_hww_kl1_kt1",
        "hh_ggf_hbb_hww_kl2p45_kt1",
        "hh_ggf_hbb_hww_kl5_kt1",
        "tt",
        "dy",
        "w_lnu",
        "vv",
        "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    ],
    "config_categories": [
        "sr__2mu__1b__ml_hh_ggf_hbb_hvv2l2nu_kl1_kt1",
        "sr__2mu__1b__ml_hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
        "sr__2mu__2b__ml_hh_ggf_hbb_hvv2l2nu_kl1_kt1",
        "sr__2mu__2b__ml_hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
        "sr__2e__1b__ml_hh_ggf_hbb_hvv2l2nu_kl1_kt1",
        "sr__2e__1b__ml_hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
        "sr__2e__2b__ml_hh_ggf_hbb_hvv2l2nu_kl1_kt1",
        "sr__2e__2b__ml_hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
        "sr__emu__1b__ml_hh_ggf_hbb_hvv2l2nu_kl1_kt1",
        "sr__emu__1b__ml_hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
        "sr__emu__2b__ml_hh_ggf_hbb_hvv2l2nu_kl1_kt1",
        "sr__emu__2b__ml_hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
        "sr__1b__ml_tt",
        "sr__1b__ml_st",
        "sr__1b__ml_dy",
        "sr__1b__ml_h",
        "sr__2b__ml_tt",
        "sr__2b__ml_st",
        "sr__2b__ml_dy",
        "sr__2b__ml_h",
    ],
})
dl.derive("dl_hww2l2nu", cls_dict={
    "processes": [
        "hh_ggf_hbb_hww2l2nu_kl0_kt1",
        "hh_ggf_hbb_hww2l2nu_kl1_kt1",
        "hh_ggf_hbb_hww2l2nu_kl2p45_kt1",
        "hh_ggf_hbb_hww2l2nu_kl5_kt1",
        "tt",
        "dy",
        "w_lnu",
        "vv",
        "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    ],
})
dl.derive("dl_hww_v1", cls_dict={
    "processes": [
        "hh_ggf_hbb_hww_kl0_kt1",
        "hh_ggf_hbb_hww_kl1_kt1",
        "hh_ggf_hbb_hww_kl2p45_kt1",
        "hh_ggf_hbb_hww_kl5_kt1",
        "tt",
        "dy",
        "w_lnu",
        "vv",
        "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    ],
})

dl.derive("dl_dummy_vbf_variation", cls_dict={
    "processes": [
        "hh_ggf_hbb_hww_kl0_kt1",
        "hh_ggf_hbb_hww_kl1_kt1",
        "hh_ggf_hbb_hww_kl2p45_kt1",
        "hh_ggf_hbb_hww_kl5_kt1",
        "hh_ggf_hbb_hzz_kl0_kt1",
        "hh_ggf_hbb_hzz_kl1_kt1",
        "hh_ggf_hbb_hzz_kl2p45_kt1",
        "hh_ggf_hbb_hzz_kl5_kt1",
        "hh_vbf_hbb_hww_kv1_k2v1_kl1",
        "hh_vbf_hbb_hzz_kv1_k2v1_kl1",
        "tt",
        "dy",
        "w_lnu",
        "vv",
        "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    ],
    "dummy_vbf_variation": True,
})
dl.derive("dl_with_vbf2", cls_dict={
    "processes": [
        "hh_vbf_hbb_hww_kv1_k2v1_kl1",
        "hh_vbf_hbb_hww_kv1_k2v0_kl1",
        "hh_vbf_hbb_hww_kv1p74_k2v1p37_kl14p4",
        "hh_vbf_hbb_hww_kvm0p012_k2v0p03_kl10p2",
        "hh_vbf_hbb_hww_kvm0p758_k2v1p44_klm19p3",
        "hh_vbf_hbb_hww_kvm0p962_k2v0p959_klm1p43",
        "hh_vbf_hbb_hww_kvm1p21_k2v1p94_klm0p94",
        "hh_vbf_hbb_hww_kvm1p6_k2v2p72_klm1p36",
        "hh_vbf_hbb_hww_kvm1p83_k2v3p57_klm3p39",
        "hh_vbf_hbb_hww_kvm2p12_k2v3p87_klm5p96",
        # "hh_vbf_hbb_hzz_kv1_k2v1_kl1",
        # "hh_vbf_hbb_hzz_kv1_k2v0_kl1",
        # "hh_vbf_hbb_hzz_kv1p74_k2v1p37_kl14p4",
        # "hh_vbf_hbb_hzz_kvm0p012_k2v0p03_kl10p2",
        # "hh_vbf_hbb_hzz_kvm0p758_k2v1p44_klm19p3",
        # "hh_vbf_hbb_hzz_kvm0p962_k2v0p959_klm1p43",
        # "hh_vbf_hbb_hzz_kvm1p21_k2v1p94_klm0p94",
        # "hh_vbf_hbb_hzz_kvm1p6_k2v2p72_klm1p36",
        # "hh_vbf_hbb_hzz_kvm1p83_k2v3p57_klm3p39",
        # "hh_vbf_hbb_hzz_kvm2p12_k2v3p87_klm5p96",
        "hh_ggf_hbb_hww_kl0_kt1",
        "hh_ggf_hbb_hww_kl1_kt1",
        "hh_ggf_hbb_hww_kl2p45_kt1",
        "hh_ggf_hbb_hww_kl5_kt1",
        # "hh_ggf_hbb_hzz_kl0_kt1",
        # "hh_ggf_hbb_hzz_kl1_kt1",
        # "hh_ggf_hbb_hzz_kl2p45_kt1",
        # "hh_ggf_hbb_hzz_kl5_kt1",
        # "hh_vbf_hbb_hvv_kv1_k2v1_kl1",  # TODO
        "tt",
        "dy",
        "w_lnu",
        "vv",
        "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    ],
})


dl.derive("dl_tmp", cls_dict={
    "processes": [
        "hh_ggf_hbb_hvv2l2nu_kl0_kt1",
        "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
        "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1",
        "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
        # "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",  # TODO
        "tt",
        "dy",
        "w_lnu",
        "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    ],
})


dl.derive("dl_22_limited", cls_dict={
    "ml_model_name": "dl_22_limited",
    "processes": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "tt_dl"],
    "config_categories": [
        "sr__2b__ml_hh_ggf_hbb_hvv2l2nu_kl1_kt1",
        "sr__2b__ml_tt_dl",
    ],
    "systematics": rate_systematics},
)
dl.derive("dl_rates_only", cls_dict={"systematics": rate_systematics})
