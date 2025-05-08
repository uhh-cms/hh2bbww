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
ml_model_name = "dl_22post_benchmark"

# All processes to be included in the final datacard
processes = [
    "hh_vbf_hbb_hww2l2nu_kv1p74_k2v1p37_kl14p4",
    "hh_vbf_hbb_hww2l2nu_kvm0p758_k2v1p44_klm19p3",
    "hh_vbf_hbb_hww2l2nu_kvm0p012_k2v0p03_kl10p2",
    "hh_vbf_hbb_hww2l2nu_kvm2p12_k2v3p87_klm5p96",
    "hh_vbf_hbb_hww2l2nu_kv1_k2v1_kl1",
    "hh_vbf_hbb_hww2l2nu_kv1_k2v0_kl1",
    "hh_vbf_hbb_hww2l2nu_kvm0p962_k2v0p959_klm1p43",
    "hh_vbf_hbb_hww2l2nu_kvm1p21_k2v1p94_klm0p94",
    "hh_vbf_hbb_hww2l2nu_kvm1p6_k2v2p72_klm1p36",
    "hh_vbf_hbb_hww2l2nu_kvm1p83_k2v3p57_klm3p39",
    "hh_ggf_hbb_hww2l2nu_kl0_kt1",
    "hh_ggf_hbb_hww2l2nu_kl1_kt1",
    "hh_ggf_hbb_hww2l2nu_kl2p45_kt1",
    "hh_ggf_hbb_hww2l2nu_kl5_kt1",
    # TODO: merge st_schannel, st_tchannel
    "st_tchannel",
    "st_twchannel",
    # "st_schannel",  # Not datasets anyways
    "tt",
    # "ttw",  # TODO: dataset not working?
    "ttz",
    "dy",
    "w_lnu",
    "vv",
    "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    # "ttv",  # TODO
    # "ttvv",  # TODO
    # "vvv",  # TODO
    # TODO: add thq, thw, bbh
    # "qcd",  # probably not needed
]

# All categories to be included in the final datacard
config_categories = [
    "sr__1b__ml_sig_ggf",
    "sr__1b__ml_sig_vbf",
    "sr__1b__ml_tt",
    "sr__1b__ml_st",
    "sr__1b__ml_dy",
    "sr__1b__ml_h",
    "sr__2b__ml_sig_ggf",
    "sr__2b__ml_sig_vbf",
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
    # Scale Factors
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
    "e_sf",
    "trigger_sf",
    "minbias_xs",
    "top_pt",
]

# All systematics to be included in the final datacard
systematics = rate_systematics + shape_systematics

default_cls_dict = {
    "ml_model_name": ml_model_name,
    "processes": processes,
    "config_categories": config_categories,
    "systematics": rate_systematics,
    "mc_stats": True,
    "skip_data": True,
}


def config_variable_binary_ggf_and_vbf(self, config_cat_inst):
    """
    Function to set the config variable for the binary model.
    """
    if "sig_ggf" in config_cat_inst.name:
        return "logit_mlscore.sig_ggf_binary"
    elif "sig_vbf" in config_cat_inst.name:
        return "logit_mlscore.sig_vbf_binary"
    elif config_cat_inst.x.root_cats.get("dnn"):
        return "mlscore.max_score"
    else:
        raise ValueError(f"Category {config_cat_inst.name} is not a DNN category.")


dl = HBWInferenceModelBase.derive("dl", cls_dict=default_cls_dict)
dl_syst = dl.derive("dl_syst", cls_dict={"systematics": systematics})

# TODO: remove since outdated model
dl_binary = dl.derive("dl_binary", cls_dict={
    "ml_model_name": ["dl_22post_benchmark", "dl_22post_binary_test2"],
    "config_variable": lambda self, config_cat_inst: "logit_mlscore.sig_ggf_binary",
    "systematics": rate_systematics,
})


previous = dl.derive("previous", cls_dict={
    "ml_model_name": ["dl_22post_previous_merge_hh"],
    "systematics": rate_systematics,
})
previous_and_binary = dl.derive("previous_and_binary", cls_dict={
    "ml_model_name": ["dl_22post_previous_merge_hh", "dl_22post_binary_test3"],
    "config_variable": lambda self, config_cat_inst: "logit_mlscore.sig_ggf_binary",
    "systematics": rate_systematics,
})
# TODO: change config variable to use the correct scores per category
benchmark = dl.derive("benchmark", cls_dict={
    "ml_model_name": ["dl_22post_benchmark_v3", "dl_22post_binary_test3"],
    "config_variable": lambda self, config_cat_inst: "logit_mlscore.sig_ggf_binary",
    "systematics": rate_systematics,
})
hh_multi_dataset = dl.derive("hh_multi_dataset", cls_dict={
    "ml_model_name": ["dl_22post_previous"],
    "systematics": rate_systematics,
})
hh_multi_dataset_and_binary = dl.derive("hh_multi_dataset_and_binary", cls_dict={
    "ml_model_name": ["dl_22post_previous", "dl_22post_binary_test3"],
    "config_variable": lambda self, config_cat_inst: "logit_mlscore.sig_ggf_binary",
    "systematics": rate_systematics,
})

with_vbf_binary = dl.derive("with_vbf_binary", cls_dict={
    "ml_model_name": ["dl_22post_previous", "dl_22post_binary_test3", "dl_22post_vbf"],
    "config_variable": config_variable_binary_ggf_and_vbf,
    "systematics": rate_systematics,
})

weight1 = dl.derive("weight1", cls_dict={
    "ml_model_name": ["dl_22post_weight1", "dl_22post_binary_test3", "dl_22post_vbf"],
    "config_variable": config_variable_binary_ggf_and_vbf,
    "systematics": rate_systematics,
})
