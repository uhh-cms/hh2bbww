# coding: utf-8

"""
hbw inference model.
"""

import hbw.inference.constants as const  # noqa
from hbw.inference.base import HBWInferenceModelBase
from columnflow.inference import InferenceModel


class HBWNoMLInferenceModel(HBWInferenceModelBase):
    def add_inference_categories(self: InferenceModel):
        """
        This function creates categories for the inference model
        """

        lepton_config_categories = self.config_inst.x.lepton_config_categories

        for lep in lepton_config_categories:
            cat_name = f"cat_{lep}"
            if cat_name not in self.config_categories:
                continue

            cat_kwargs = {
                "config_category": f"{lep}",
                "config_variable": "m_Heavy_Higgs",
                "mc_stats": self.mc_stats,
            }
            if self.skip_data:
                cat_kwargs["data_from_processes"] = self.processes
            else:
                cat_kwargs["config_data_datasets"] = const.data_datasets[lep]

            self.add_category(cat_name, **cat_kwargs)

            # get the inference category to do some customization
            cat = self.get_category(cat_name)
            # variables that are plotting via hbw.InferencePlots for this category
            cat.plot_variables = ["jet1_pt"]


#
# Defaults for all the Inference Model parameters
#

# used to set default requirements for cf.CreateDatacards based on the config
ml_model_name = "dense_graviton600"

# All processes to be included in the final datacard
processes = [
    "graviton_hh_ggf_bbww_m600",
    "tt",
    "st",
    "dy_lep",
    "w_lnu",
    # "v_lep",
    # "t_bkg",
]

# All inference config_categories to be included in the final datacard
config_categories = [
    # "1e__ml",
    "cat__1e_graviton_hh_ggf_bbww_m600",
    # "1e__ml_tt",
    # "1e__ml_st",
    # "1e__ml_dy_lep",
    # "1e__ml_w_lnu",
    "1e__ml_v_lep",
    "1e__ml_t_bkg",
    # "1mu__ml",
    "1mu__ml_graviton_hh_ggf_bbww_m600",
    # "1mu__ml_tt",
    # "1mu__ml_st",
    # "1mu__ml_dy_lep",
    # "1mu__ml_w_lnu",
    "1mu__ml_v_lep",
    "1mu__ml_t_bkg",
]

config_categories_not_ml = [
    # "1e__ml",
    "1e_graviton_hh_ggf_bbww_m600",
    # "1e__ml_tt",
    # "1e__ml_st",
    # "1e__ml_dy_lep",
    # "1e__ml_w_lnu",
    "1e__ml_v_lep",
    "1e__ml_t_bkg",
    # "1mu__ml",
    "1mu__ml_graviton_hh_ggf_bbww_m600",
    # "1mu__ml_tt",
    # "1mu__ml_st",
    # "1mu__ml_dy_lep",
    # "1mu__ml_w_lnu",
    "1mu__ml_v_lep",
    "1mu__ml_t_bkg",
]

rate_systematics = [
    # Lumi: should automatically choose viable uncertainties based on campaign
    "lumi_13TeV_2016",
    "lumi_13TeV_2017",
    "lumi_13TeV_1718",
    "lumi_13TeV_correlated",
    # Rate QCDScale uncertainties
    # "QCDScale_ttbar",
    # "QCDScale_V",
    # "QCDScale_VV",
    # "QCDScale_VVV",
    # "QCDScale_ggH",
    # "QCDScale_qqH",
    # "QCDScale_VH",
    # "QCDScale_ttH",
    # "QCDScale_bbH",
    # "QCDScale_ggHH",  # should be included in inference model (THU_HH)
    # "QCDScale_qqHH",
    # "QCDScale_VHH",
    # "QCDScale_ttHH",
    # # # Rate PDF uncertainties
    # "pdf_gg",
    # "pdf_qqbar",
    # "pdf_qg",
    # "pdf_Higgs_gg",
    # "pdf_Higgs_qqbar",
    # "pdf_Higgs_qg",  # none so far
    # "pdf_Higgs_ttH",
    # "pdf_Higgs_bbH",  # removed
    # "pdf_Higgs_ggHH",
    # "pdf_Higgs_qqHH",
    # "pdf_VHH",
    # "pdf_ttHH",
]

shape_systematics = [
    # # Shape Scale uncertainties
    # # "murf_envelope_ggHH_kl_1_kt_1_sl_hbbhww",
    # "murf_envelope_tt",
    # "murf_envelope_st_schannel",
    # "murf_envelope_st_tchannel",
    # "murf_envelope_st_twchannel",
    # "murf_envelope_dy_lep",
    # "murf_envelope_w_lnu",
    # "murf_envelope_ttV",
    # "murf_envelope_VV",
    # # Shape PDF Uncertainties
    # "pdf_shape_tt",
    # "pdf_shape_st_schannel",
    # "pdf_shape_st_tchannel",
    # "pdf_shape_st_twchannel",
    # "pdf_shape_dy_lep",
    # "pdf_shape_w_lnu",
    # "pdf_shape_ttV",
    # "pdf_shape_VV",
    # # Scale Factors (TODO)
    # "btag_hf",
    # "btag_lf",
    # "btag_hfstats1_2017",
    # "btag_hfstats2_2017"
    # "btag_lfstats1_2017"
    # "btag_lfstats2_2017"
    # "btag_cferr1",
    # "btag_cferr2",
    # "mu_sf",
    # # # "mu_trig",
    # "e_sf",
    # # "e_trig",
    # "minbias_xs",
    # "top_pt",
]

# All systematics to be included in the final datacard
systematics = rate_systematics + shape_systematics

default_cls_dict = {
    "ml_model_name": "dense_default",
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

for m in [250, 350, 450, 600, 750, 1000]:

    processes_ml = [
        f"graviton_hh_ggf_bbww_m{m}",
        "tt",
        "st",
        "dy_lep",
        "w_lnu",
    ]

    processes_merged_bkg = [
        f"graviton_hh_ggf_bbww_m{m}",
        "v_lep",
        "t_bkg",
    ]

    config_categories_ml_merged_bkg = [
        f"1e__ml_graviton_hh_ggf_bbww_m{m}",
        "1e__ml_v_lep",
        "1e__ml_t_bkg",
        f"1mu__ml_graviton_hh_ggf_bbww_m{m}",
        "1mu__ml_v_lep",
        "1mu__ml_t_bkg",
    ]
    config_categories_ml = [
        f"1e__ml_graviton_hh_ggf_bbww_m{m}",
        "1e__ml_tt",
        "1e__ml_st",
        "1e__ml_dy_lep",
        "1e__ml_w_lnu",
        f"1mu__ml_graviton_hh_ggf_bbww_m{m}",
        "1mu__ml_tt",
        "1mu__ml_st",
        "1mu__ml_dy_lep",
        "1mu__ml_w_lnu",
    ]
    ml_model_name = f"dense_graviton{m}"
    config_categories__not_ml = [
        "1e__ml",
        "1mu__ml",
    ]
    cls_dict_ml = {
        "ml_model_name": ml_model_name,
        "processes": processes_ml,
        "config_categories": config_categories_ml,
        "systematics": systematics,
        "mc_stats": True,
        "skip_data": True,
    }
    cls_dict_ml_merged_bkg = {
        "ml_model_name": ml_model_name,
        "processes": processes_merged_bkg,
        "config_categories": config_categories_ml_merged_bkg,
        "systematics": systematics,
        "mc_stats": True,
        "skip_data": True,
    }
    cls_dict_not_ml = {
        "ml_model_name": None,
        "processes": processes_ml,
        "config_categories": config_categories__not_ml,
        "systematics": systematics,
        "mc_stats": True,
        "skip_data": True,
    }

    # default_new = HBWInferenceModelBase.derive(f"graviton{m}_ml", cls_dict=cls_dict_ml)
    # default_new_merged_bkg = HBWInferenceModelBase.derive(
    #     f"graviton{m}_ml_merged_bkg", cls_dict=cls_dict_ml_merged_bkg,
    # )
    # default_new_not_ml = HBWNoMLInferenceModel.derive(
    #     f"graviton{m}_not_ml", cls_dict=cls_dict_not_ml,
    # )
