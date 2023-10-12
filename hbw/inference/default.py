# coding: utf-8

"""
hbw inference model.
"""

from columnflow.inference import InferenceModel, inference_model
import hbw.inference.constants as const  # noqa

from hbw.inference.functions import (
    add_inference_categories, add_inference_processes, add_inference_parameters,
)

#
# Defaults for all the Inference Model parameters
#

# used to set default requirements for cf.CreateDatacards based on the config
ml_model_name = "dense_default"
# default_producers = [f"ml_{ml_model_name}", "event_weights"]

# All processes to be included in the final datacard
processes = [
    "ggHH_kl_0_kt_1_sl_hbbhww",
    "ggHH_kl_1_kt_1_sl_hbbhww",
    "ggHH_kl_2p45_kt_1_sl_hbbhww",
    "ggHH_kl_5_kt_1_sl_hbbhww",
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

# All inference channels to be included in the final datacard
channels = [
    "cat_1e_ggHH_kl_1_kt_1_sl_hbbhww",
    "cat_1e_tt",
    "cat_1e_st",
    "cat_1e_v_lep",
    "cat_1mu_ggHH_kl_1_kt_1_sl_hbbhww",
    "cat_1mu_tt",
    "cat_1mu_st",
    "cat_1mu_v_lep",
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
    "pdf_ggHH",
    "pdf_qqHH",
    "pdf_VHH",
    "pdf_ttHH",
]

shape_systematics = [
    # Shape Scale uncertainties
    # "murf_envelope_ggHH_kl_1_kt_1_sl_hbbhww",
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
systematics = rate_systematics + shape_systematics


default_cls_dict = {
    "ml_model_name": ml_model_name,
    "processes": processes,
    "channels": channels,
    "systematics": systematics,
    "mc_stats": True,
    "skip_data": True,
}


@inference_model(
    **default_cls_dict,
)
def default(self: InferenceModel):
    """
    This is the default Inference model.
    Idea: first build an inclusive Inference Model with all Channels/Processes/Systematics,
    then remove anything not listed in the attributes.
    """
    year = self.config_inst.campaign.x.year  # noqa; not used right now

    add_inference_categories(self)
    add_inference_processes(self)
    add_inference_parameters(self)

    #
    # post-processing
    #

    self.cleanup()


#
# derive some additional Inference Models
#

cls_dict = default_cls_dict.copy()

cls_dict["systematics"] = rate_systematics

# inference model with only rate uncertainties
rates_only = default.derive("rates_only", cls_dict=cls_dict)

cls_dict["processes"] = [
    "ggHH_kl_0_kt_1_sl_hbbhww",
    "ggHH_kl_1_kt_1_sl_hbbhww",
    "ggHH_kl_2p45_kt_1_sl_hbbhww",
    "ggHH_kl_5_kt_1_sl_hbbhww",
    "st_schannel",
]

cls_dict["channels"] = [
    "cat_1e_ggHH_kl_1_kt_1_sl_hbbhww",
    "cat_1e_st",
]

cls_dict["systematics"] = [
    "lumi_13TeV_2017",
]

cls_dict["ml_model_name"] = "dense_test"

# minimal model for quick test purposes
test = default.derive("test", cls_dict=cls_dict)
