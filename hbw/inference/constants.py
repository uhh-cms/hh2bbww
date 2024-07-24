# coding: utf-8

"""
Collection of configurations that stay constant for the analysis
"""

# collection of all signal processes
signals_hh_ggf = {
    "hh_ggf_kl0_kt1_hbb_hvvqqlnu", "hh_ggf_kl1_kt1_hbb_hvvqqlnu",
    "hh_ggf_kl2p45_kt1_hbb_hvvqqlnu", "hh_ggf_kl5_kt1_hbb_hvvqqlnu",
    "hh_ggf_kl0_kt1_hbb_hvv2l2nu", "hh_ggf_kl1_kt1_hbb_hvv2l2nu",
    "hh_ggf_kl2p45_kt1_hbb_hvv2l2nu", "hh_ggf_kl5_kt1_hbb_hvv2l2nu",
}
signals_hh_vbf = {
    "hh_vbf_kv1_k2v1_kl1_hbb_hvvqqlnu", "hh_vbf_kv1_k2v1_kl0_hbb_hvvqqlnu", "hh_vbf_kv1_k2v1_kl2_hbb_hvvqqlnu",
    "hh_vbf_kv1_k2v0_kl1_hbb_hvvqqlnu", "hh_vbf_kv1_k2v2_kl1_hbb_hvvqqlnu",
    "hh_vbf_kv0p5_k2v1_kl1_hbb_hvvqqlnu", "hh_vbf_kv1p5_k2v1_kl1_hbb_hvvqqlnu",
    "hh_vbf_kv1_k2v1_kl1_hbb_hvv2l2nu", "hh_vbf_kv1_k2v1_kl0_hbb_hvv2l2nu", "hh_vbf_kv1_k2v1_kl2_hbb_hvv2l2nu",
    "hh_vbf_kv1_k2v0_kl1_hbb_hvv2l2nu", "hh_vbf_kv1_k2v2_kl1_hbb_hvv2l2nu",
    "hh_vbf_kv0p5_k2v1_kl1_hbb_hvv2l2nu", "hh_vbf_kv1p5_k2v1_kl1_hbb_hvv2l2nu",
}
signals = {*signals_hh_ggf, *signals_hh_vbf}

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
    "V": ["dy", "w_lnu"],
    "VV": ["WW", "ZZ", "WZ", "qqZZ"],
    "VVV": ["vvv"],
    "ggH": ["ggH"],
    "qqH": ["qqH"],
    "VH": ["ZH", "WH", "VH"],
    "ttH": ["ttH", "tHq", "tHW"],
    "bbH": ["bbH"],  # contains also pdf and alpha_s
    # "hh_ggf": signals_hh_ggf,  # included in inference model (THU_HH)
    "hh_vbf": signals_hh_vbf,
    "VHH": [],
    "ttHH": [],
}

# mapping, which processes are used for which pdf (rate) uncertainty
processes_per_pdf_rate = {
    "gg": ["tt", "ttZ", "ggZZ"],
    "qqbar": ["st_schannel", "st_tchannel", "dy", "w_lnu", "vvv", "qqZZ", "ttW"],
    "qg": ["st_twchannel"],
    "Higgs_gg": ["ggH"],
    "Higgs_qqbar": ["qqH", "ZH", "WH", "VH"],
    # "Higgs_qg": [],  # none so far
    "Higgs_ttH": ["ttH", "tHq", "tHW"],
    # "Higgs_bbh": ["bbH"],  # removed
    "Higgs_hh_ggf": signals_hh_ggf,
    "Higgs_hh_vbf": signals_hh_vbf,
    "Higgs_VHH": ["HHZ", "HHW+", "HHW-"],
    "Higgs_ttHH": ["ttHH"],
}

# mapping for each shape uncertainty, which process is used.
# If "all" is included, takes all processes except for the ones specified (starting with !)
processes_per_shape = {
    "btag_hf": ["all"],
    "btag_lf": ["all"],
    "btag_hfstats1_{year}": ["all"],
    "btag_hfstats2_{year}": ["all"],
    "btag_lfstats1_{year}": ["all"],
    "btag_lfstats2_{year}": ["all"],
    "btag_cferr1": ["all"],
    "btag_cferr2": ["all"],
    "mu_id_sf": ["all"],
    "mu_iso_sf": ["all"],
    "mu_trig_sf": ["all"],
    "e_sf": ["all"],
    "e_trig_sf": ["all"],
    "minbias_xs": ["all"],
    "top_pt": ["tt"],
    "pdf_shape_hh_ggf_kl0_kt1_hbb_hvvqqlnu": ["hh_ggf_kl0_kt1_hbb_hvvqqlnu"],
    "pdf_shape_hh_ggf_kl1_kt1_hbb_hvvqqlnu": ["hh_ggf_kl1_kt1_hbb_hvvqqlnu"],
    "pdf_shape_hh_ggf_kl2p45_kt1_hbb_hvvqqlnu": ["hh_ggf_kl2p45_kt1_hbb_hvvqqlnu"],
    "pdf_shape_hh_ggf_kl5_kt1_hbb_hvvqqlnu": ["hh_ggf_kl5_kt1_hbb_hvvqqlnu"],
    "pdf_shape_hh_ggf_kl0_kt1_hbb_hvv2l2nu": ["hh_ggf_kl0_kt1_hbb_hvv2l2nu"],
    "pdf_shape_hh_ggf_kl1_kt1_hbb_hvv2l2nu": ["hh_ggf_kl1_kt1_hbb_hvv2l2nu"],
    "pdf_shape_hh_ggf_kl2p45_kt1_hbb_hvv2l2nu": ["hh_ggf_kl2p45_kt1_hbb_hvv2l2nu"],
    "pdf_shape_hh_ggf_kl5_kt1_hbb_hvv2l2nu": ["hh_ggf_kl5_kt1_hbb_hvv2l2nu"],
    "pdf_shape_tt": ["tt"],
    "pdf_shape_st": ["st_schannel", "st_twchannel"],  # TODO: there was some bug with "st_tchannel"
    "pdf_shape_dy": ["dy"],
    "pdf_shape_w": ["w_lnu"],
    "murf_envelope_hh_ggf_kl0_kt1_hbb_hvvqqlnu": ["hh_ggf_kl0_kt1_hbb_hvvqqlnu"],
    "murf_envelope_hh_ggf_kl1_kt1_hbb_hvvqqlnu": ["hh_ggf_kl1_kt1_hbb_hvvqqlnu"],
    "murf_envelope_hh_ggf_kl2p45_kt1_hbb_hvvqqlnu": ["hh_ggf_kl2p45_kt1_hbb_hvvqqlnu"],
    "murf_envelope_hh_ggf_kl5_kt1_hbb_hvvqqlnu": ["hh_ggf_kl5_kt1_hbb_hvvqqlnu"],
    "murf_envelope_hh_ggf_kl0_kt1_hbb_hvv2l2nu": ["hh_ggf_kl0_kt1_hbb_hvv2l2nu"],
    "murf_envelope_hh_ggf_kl1_kt1_hbb_hvv2l2nu": ["hh_ggf_kl1_kt1_hbb_hvv2l2nu"],
    "murf_envelope_hh_ggf_kl2p45_kt1_hbb_hvv2l2nu": ["hh_ggf_kl2p45_kt1_hbb_hvv2l2nu"],
    "murf_envelope_hh_ggf_kl5_kt1_hbb_hvv2l2nu": ["hh_ggf_kl5_kt1_hbb_hvv2l2nu"],
    "murf_envelope_tt": ["tt"],
    "murf_envelope_st": ["st_schannel", "st_tchannel", "st_twchannel"],
    "murf_envelope_dy": ["dy"],
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
