# coding: utf-8

"""
Collection of configurations that stay constant for the analysis
"""

# collection of all signal processes
signals_ggHH = {
    "ggHH_kl_0_kt_1_sl_hbbhww", "ggHH_kl_1_kt_1_sl_hbbhww",
    "ggHH_kl_2p45_kt_1_sl_hbbhww", "ggHH_kl_5_kt_1_sl_hbbhww",
}
signals_qqHH = {
    "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww", "qqHH_CV_1_C2V_1_kl_0_sl_hbbhww", "qqHH_CV_1_C2V_1_kl_2_sl_hbbhww",
    "qqHH_CV_1_C2V_0_kl_1_sl_hbbhww", "qqHH_CV_1_C2V_2_kl_1_sl_hbbhww",
    "qqHH_CV_0p5_C2V_1_kl_1_sl_hbbhww", "qqHH_CV_1p5_C2V_1_kl_1_sl_hbbhww",
}
signals = {*signals_ggHH, *signals_qqHH}

# mapping between lepton categories and datasets (only 2017 ATM)
data_datasets = {
    "1e": {f"data_e_{i}" for i in ["b", "c", "d", "e", "f"]},
    "1mu": {f"data_mu_{i}" for i in ["b", "c", "d", "e", "f"]},
    "2e": {"data_e_b"},  # TODO: 2 lep datasets in cmsdb + config
    "2mu": {"data_mu_b"},  # TODO
    "emu": {"data_mu_b"},  # TODO
}
merged_datasets = set().union(*data_datasets.values())

# mapping between process names in the config and inference model
inference_procnames = {
    # key: config process name, value: inference model process name
    "foo": "bar",
    # "st": "ST",
    # "tt": "TT",
}

# mapping, which processes are used for which QCDScale (rate) uncertainty
processes_per_QCDScale = {
    "ttbar": ["tt", "st_tchannel", "st_schannel", "st_twchannel", "ttW", "ttZ"],
    "V": ["dy_lep", "w_lnu"],
    "VV": ["WW", "ZZ", "WZ", "qqZZ"],
    "VVV": ["vvv"],
    "ggH": ["ggH"],
    "qqH": ["qqH"],
    "VH": ["ZH", "WH", "VH"],
    "ttH": ["ttH", "tHq", "tHW"],
    "bbH": ["bbH"],  # contains also pdf and alpha_s
    # "ggHH": signals_ggHH,  # included in inference model (THU_HH)
    "qqHH": signals_qqHH,
    "VHH": [],
    "ttHH": [],
}

# mapping, which processes are used for which pdf (rate) uncertainty
processes_per_pdf_rate = {
    "gg": ["tt", "ttZ", "ggZZ"],
    "qqbar": ["st_schannel", "st_tchannel", "dy_lep", "w_lnu", "vvv", "qqZZ", "ttW"],
    "qg": ["st_twchannel"],
    "Higgs_gg": ["ggH"],
    "Higgs_qqbar": ["qqH", "ZH", "WH", "VH"],
    # "Higgs_qg": [],  # none so far
    "Higgs_ttH": ["ttH", "tHq", "tHW"],
    # "Higgs_bbh": ["bbH"],  # removed
    "Higgs_ggHH": signals_ggHH,
    "Higgs_qqHH": signals_qqHH,
    "Higgs_VHH": ["HHZ", "HHW+", "HHW-"],
    "Higgs_ttHH": ["ttHH"],
}

# mapping for each shape uncertainty, which process is used.
# If "all" is included, takes all processes except for the ones specified (starting with !)
processes_per_shape = {
    "btag_hf": ["all"],
    "btag_lf": ["all"],
    "btag_hfstats1_2017": ["all"],
    "btag_hfstats2_2017": ["all"],
    "btag_lfstats1_2017": ["all"],
    "btag_lfstats2_2017": ["all"],
    "btag_cferr1": ["all"],
    "btag_cferr2": ["all"],
    "mu_trig": ["all"],
    "e_sf": ["all"],
    "e_trig": ["all"],
    "minbias_xs": ["all"],
    "top_pt": ["all"],
    "pdf_shape_ggHH_kl_0_kt_1_sl_hbbhww": ["ggHH_kl_0_kt_1_sl_hbbhww"],
    "pdf_shape_ggHH_kl_1_kt_1_sl_hbbhww": ["ggHH_kl_1_kt_1_sl_hbbhww"],
    "pdf_shape_ggHH_kl_2p45_kt_1_sl_hbbhww": ["ggHH_kl_2p45_kt_1_sl_hbbhww"],
    "pdf_shape_ggHH_kl_5_kt_1_sl_hbbhww": ["ggHH_kl_5_kt_1_sl_hbbhww"],
    "pdf_shape_tt": ["tt"],
    "pdf_shape_st": ["st_schannel", "st_twchannel"],  # TODO: there was some bug with "st_tchannel"
    "pdf_shape_dy": ["dy_lep"],
    "pdf_shape_w": ["w_lnu"],
    "murf_envelope_ggHH_kl_0_kt_1_sl_hbbhww": ["ggHH_kl_0_kt_1_sl_hbbhww"],
    "murf_envelope_ggHH_kl_1_kt_1_sl_hbbhww": ["ggHH_kl_1_kt_1_sl_hbbhww"],
    "murf_envelope_ggHH_kl_2p45_kt_1_sl_hbbhww": ["ggHH_kl_2p45_kt_1_sl_hbbhww"],
    "murf_envelope_ggHH_kl_5_kt_1_sl_hbbhww": ["ggHH_kl_5_kt_1_sl_hbbhww"],
    "murf_envelope_tt": ["tt"],
    "murf_envelope_st": ["st_schannel", "st_tchannel", "st_twchannel"],
    "murf_envelope_dy": ["dy_lep"],
    "murf_envelope_w": ["w_lnu"],
}

# mapping for each shape uncertainty, which shift source is used
# per default: shape and source have the same name (except for pdf and murf, which are implemented per process)
source_per_shape = {shape: shape for shape in processes_per_shape.keys()}
for shape in processes_per_shape.keys():
    if "pdf_shape" in shape:
        source_per_shape[shape] = "pdf"
    elif "murf_envelope" in shape:
        source_per_shape[shape] = "murf_envelope"
