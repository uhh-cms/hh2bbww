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

# collection of all datasets (only 2017 ATM)
e_datasets = {f"data_e_{i}" for i in ["b", "c", "d", "e", "f"]}
mu_datasets = {f"data_mu_{i}" for i in ["b", "c", "d", "e", "f"]}
datasets = {*e_datasets, *mu_datasets}

# mapping, which processes are used for which QCDScale (rate) uncertainty
QCDScale_mapping = {
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
pdf_mapping = {
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
