# coding: utf-8

"""
Collection of configurations that stay constant for the analysis
"""

# TODO: mapping to naming that is accepted by inference
# collection of all signal processes
signals_hh_ggf = {
    "hh_ggf_hbb_hvvqqlnu_kl0_kt1", "hh_ggf_hbb_hvvqqlnu_kl1_kt1",
    "hh_ggf_hbb_hvvqqlnu_kl2p45_kt1", "hh_ggf_hbb_hvvqqlnu_kl5_kt1",
    "hh_ggf_hbb_hvv2l2nu_kl0_kt1", "hh_ggf_hbb_hvv2l2nu_kl1_kt1",
    "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1", "hh_ggf_hbb_hvv2l2nu_kl5_kt1",
}
# TODO: they are different in run 2 and run 3
signals_hh_vbf = {
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1", "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl0", "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl2",
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v0_kl1", "hh_vbf_hbb_hvvqqlnu_kv1_k2v2_kl1",
    "hh_vbf_hbb_hvvqqlnu_kv0p5_k2v1_kl1", "hh_vbf_hbb_hvvqqlnu_kv1p5_k2v1_kl1",
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1", "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl0", "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl2",
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1", "hh_vbf_hbb_hvv2l2nu_kv1_k2v2_kl1",
    "hh_vbf_hbb_hvv2l2nu_kv0p5_k2v1_kl1", "hh_vbf_hbb_hvv2l2nu_kv1p5_k2v1_kl1",
}
signals = {*signals_hh_ggf, *signals_hh_vbf}

# mapping between process names in the config and inference model
# key: config process name, value: inference model process name
# NOTE: when possible, we use the names from the config (only map to inference names if necessary)
inference_procnames = {
    # main backgrounds ( optional name conventions)
    # "tt": "TT",
    # "st_tchannel": "ST",  # TODO: add st_schannel to ST
    # "st_twchannel": "TW",
    # "w_lnu": "W",
    # "dy": "DY",
    # "vv": "VV",
    # single higgs (required name conventions)
    "h_ggf": "ggH",
    "h_vbf": "qqH",
    "vh": "VH",
    "zh": "ZH",
    "wh": "WH",
    "zh_gg": "ggZH",
    "tth": "ttH",
    "bbh": "bbH",
    "thq": "tHq",
    "thw": "tHW",
}

# mapping: which processes are used for which QCDScale (rate) uncertainty
processes_per_QCDScale = {
    "ttbar": ["tt", "st_tchannel", "st_schannel", "st_twchannel", "ttw", "ttz"],
    "V": ["dy", "w_lnu"],
    "VV": ["vv", "ww", "zz", "wz", "qqZZ"],
    "VVV": ["vvv"],
    "ggH": ["h_ggf"],
    "qqH": ["h_vbf"],
    "VH": ["zh", "wh", "vh"],
    "ttH": ["tth", "thq", "thw"],
    "bbH": ["bbh"],  # contains also pdf and alpha_s
    # "hh_ggf": signals_hh_ggf,  # included in inference model (THU_HH)
    "hh_vbf": signals_hh_vbf,
    "VHH": [],
    "ttHH": [],
}

# mapping, which processes are used for which pdf (rate) uncertainty
processes_per_pdf_rate = {
    "gg": ["tt", "ttz", "ggZZ"],
    "qqbar": ["st_schannel", "st_tchannel", "dy", "w_lnu", "vvv", "qqZZ", "ttw"],
    "qg": ["st_twchannel"],
    "Higgs_gg": ["h_ggf"],
    "Higgs_qqbar": ["h_vbf", "zh", "wh", "vh"],
    # "Higgs_qg": [],  # none so far
    "Higgs_ttH": ["tth", "thq", "thw"],
    # "Higgs_bbh": ["bbH"],  # removed
    "Higgs_hh_ggf": signals_hh_ggf,
    "Higgs_hh_vbf": signals_hh_vbf,
    "Higgs_VHH": ["HHZ", "HHW+", "HHW-"],
    "Higgs_ttHH": ["tthh"],
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
    "trigger_sf": ["all"],
    "minbias_xs": ["all"],
    "top_pt": ["tt"],
    "pdf_shape_tt": ["tt"],
    "pdf_shape_st": ["st_schannel", "st_twchannel"],  # TODO: there was some bug with "st_tchannel"
    "pdf_shape_dy": ["dy"],
    "pdf_shape_w": ["w_lnu"],
    "pdf_shape_vv": ["vv", "ww", "zz", "wz"],
    "pdf_shape_ttV": ["ttw", "ttz"],
    "pdf_shape_h": ["h", "h_ggf", "h_vbf", "vh", "wh", "zh", "tth", "bbh"],
    "pdf_shape_hh_ggf_hbb_hvvqqlnu_kl0_kt1": ["hh_ggf_hbb_hvvqqlnu_kl0_kt1"],
    "pdf_shape_hh_ggf_hbb_hvvqqlnu_kl1_kt1": ["hh_ggf_hbb_hvvqqlnu_kl1_kt1"],
    "pdf_shape_hh_ggf_hbb_hvvqqlnu_kl2p45_kt1": ["hh_ggf_hbb_hvvqqlnu_kl2p45_kt1"],
    "pdf_shape_hh_ggf_hbb_hvvqqlnu_kl5_kt1": ["hh_ggf_hbb_hvvqqlnu_kl5_kt1"],
    "pdf_shape_hh_ggf_hbb_hvv2l2nu_kl0_kt1": ["hh_ggf_hbb_hvv2l2nu_kl0_kt1"],
    "pdf_shape_hh_ggf_hbb_hvv2l2nu_kl1_kt1": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1"],
    "pdf_shape_hh_ggf_hbb_hvv2l2nu_kl2p45_kt1": ["hh_ggf_hbb_hvv2l2nu_kl2p45_kt1"],
    "pdf_shape_hh_ggf_hbb_hvv2l2nu_kl5_kt1": ["hh_ggf_hbb_hvv2l2nu_kl5_kt1"],
    "murf_envelope_hh_ggf_hbb_hvvqqlnu_kl0_kt1": ["hh_ggf_hbb_hvvqqlnu_kl0_kt1"],
    "murf_envelope_hh_ggf_hbb_hvvqqlnu_kl1_kt1": ["hh_ggf_hbb_hvvqqlnu_kl1_kt1"],
    "murf_envelope_hh_ggf_hbb_hvvqqlnu_kl2p45_kt1": ["hh_ggf_hbb_hvvqqlnu_kl2p45_kt1"],
    "murf_envelope_hh_ggf_hbb_hvvqqlnu_kl5_kt1": ["hh_ggf_hbb_hvvqqlnu_kl5_kt1"],
    "murf_envelope_hh_ggf_hbb_hvv2l2nu_kl0_kt1": ["hh_ggf_hbb_hvv2l2nu_kl0_kt1"],
    "murf_envelope_hh_ggf_hbb_hvv2l2nu_kl1_kt1": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1"],
    "murf_envelope_hh_ggf_hbb_hvv2l2nu_kl2p45_kt1": ["hh_ggf_hbb_hvv2l2nu_kl2p45_kt1"],
    "murf_envelope_hh_ggf_hbb_hvv2l2nu_kl5_kt1": ["hh_ggf_hbb_hvv2l2nu_kl5_kt1"],
    "murf_envelope_tt": ["tt"],
    "murf_envelope_st": ["st_schannel", "st_tchannel", "st_twchannel"],
    "murf_envelope_dy": ["dy"],
    "murf_envelope_w": ["w_lnu"],
    "murf_envelope_vv": ["vv", "ww", "zz", "wz"],
    "murf_envelope_ttv": ["ttw", "ttz"],
    "murf_envelope_h": ["h", "h_ggf", "h_vbf", "vh", "wh", "zh", "tth", "bbh"],
}

# mapping for each shape uncertainty, which shift source is used
# per default: shape and source have the same name (except for pdf and murf, which are implemented per process)
source_per_shape = {shape: shape for shape in processes_per_shape.keys()}
for shape in processes_per_shape.keys():
    if "pdf_shape" in shape:
        source_per_shape[shape] = "pdf"
    elif "murf_envelope" in shape:
        source_per_shape[shape] = "murf_envelope"
