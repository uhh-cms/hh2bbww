# coding: utf-8

"""
hbw(dl) inference model.
"""

import law
from columnflow.util import DotDict
import hbw.inference.constants as const  # noqa
from hbw.inference.base import HBWInferenceModelBase


logger = law.logger.get_logger(__name__)

#
# Defaults for all the Inference Model parameters
#

# used to set default requirements for cf.CreateDatacards based on the config
ml_model_name = ["multiclass", "ggf", "vbf"]

# All categories to be included in the final datacard
config_categories = DotDict({
    "default": [
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
    ],
    "sr": [
        "sr__1b__ml_sig_ggf",
        "sr__1b__ml_sig_vbf",
        "sr__2b__ml_sig_ggf",
        "sr__2b__ml_sig_vbf",
    ],
    "background": [
        "sr__1b__ml_tt",
        "sr__1b__ml_st",
        "sr__1b__ml_dy",
        "sr__1b__ml_h",
        "sr__2b__ml_tt",
        "sr__2b__ml_st",
        "sr__2b__ml_dy",
        "sr__2b__ml_h",
    ],
    "no_nn_cats": [
        "sr__1b",
        "sr__2b",
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
    # unconstrained rate uncertainties
    "rate_ttbar",
    "rate_dy",
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

jerc_systematics = [
    "jer",
    "jec_Total",
]

# All systematics to be included in the final datacard
systematics = rate_systematics + shape_systematics

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
    f"hh_vbf_{hhdecay}_kvm2p12_k2v3p87_klm5p96",
    f"hh_vbf_{hhdecay}_kv1_k2v1_kl1",
    f"hh_vbf_{hhdecay}_kv1_k2v0_kl1",
    f"hh_vbf_{hhdecay}_kvm0p962_k2v0p959_klm1p43",
    f"hh_vbf_{hhdecay}_kvm1p21_k2v1p94_klm0p94",
    f"hh_vbf_{hhdecay}_kvm1p6_k2v2p72_klm1p36",
    f"hh_vbf_{hhdecay}_kvm1p83_k2v3p57_klm3p39",
]
hhprocs = lambda hhdecay: [*hhprocs_ggf(hhdecay), *hhprocs_vbf(hhdecay)]

backgrounds = [
    # TODO: merge st_schannel, st_tchannel
    "st_tchannel",
    "st_twchannel",
    "st_schannel",
    "tt",
    "ttw",
    "ttz",
    "dy_hf", "dy_lf",
    # "w_lnu",  # TODO: bogus norm?
    "vv",
    "vvv",
    "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
    "thq", "thw", "ttvh",
    "tttt",
    "ttvv",
    # TODO: add bbh
    # "qcd",  # probably not needed
]

processes_dict = {
    "test": ["tt", *hhprocs("hbb_hww2l2nu")],
    "hww": [*backgrounds, *hhprocs("hbb_hww")],
    "hwwzztt": [*backgrounds, *hhprocs("hbb_hww"), *hhprocs("hbb_hzz"), *hhprocs("hbb_htt")],
    "hwwzztt_ggf": [*backgrounds, *hhprocs_ggf("hbb_hww"), *hhprocs_ggf("hbb_hzz"), *hhprocs_ggf("hbb_htt")],
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
        # raise ValueError(f"Category {config_cat_inst.name} is not a DNN category.")
        logger.warning(
            f"Category {config_cat_inst.name} is not a DNN category, using binary classifier score.",
        )
        return "logit_mlscore.sig_ggf_binary"


default_cls_dict = {
    "ml_model_name": ml_model_name,
    "processes": processes_dict["hwwzztt"],
    "config_categories": config_categories.default,
    "systematics": rate_systematics,
    "config_variable": config_variable_binary_ggf_and_vbf,
    "mc_stats": True,
    "skip_data": True,
}


dl = HBWInferenceModelBase.derive("dl", cls_dict=default_cls_dict)
dl_test = HBWInferenceModelBase.derive("dl_test", cls_dict=default_cls_dict)
dl_bkg_cats = dl.derive("dl_bkg_cats", cls_dict={"config_categories": config_categories.background})
dl_syst = dl.derive("dl_syst", cls_dict={"systematics": systematics})
dl_jerc_only = dl.derive("dl_jerc_only", cls_dict={"systematics": jerc_systematics})
dl_jerc = dl.derive("dl_jerc", cls_dict={"systematics": systematics + jerc_systematics})
dl_data = dl.derive("dl_data", cls_dict={
    "config_categories": config_categories.background,
    "systematics": systematics + jerc_systematics,
    "skip_data": False,
})

no_nn_cats = dl.derive("no_nn_cats", cls_dict={
    "processes": processes_dict["hwwzztt"],
    "systematics": systematics + jerc_systematics,
    "config_categories": config_categories.no_nn_cats,
})
bjet_incl = dl.derive("bjet_incl", cls_dict={
    "processes": processes_dict["hwwzztt"],
    "systematics": systematics + jerc_systematics,
    "config_categories": config_categories.bjet_incl,
})
test = dl.derive("test", cls_dict={
    "processes": processes_dict["test"],
    "systematics": rate_systematics + ["pdf_shape_tt"],
})
test_syst = dl.derive("test_syst", cls_dict={
    "processes": processes_dict["test"],
    "systematics": rate_systematics + jerc_systematics,
})
