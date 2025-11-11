# coding: utf-8

"""
Collection of configurations that stay constant for the analysis
"""


# TODO: mapping to naming that is accepted by inference
# collection of all signal processes


signals_hh_ggf_hhdecay = lambda hhdecay: [
    f"hh_ggf_{hhdecay}_kl0_kt1",
    f"hh_ggf_{hhdecay}_kl1_kt1",
    f"hh_ggf_{hhdecay}_kl2p45_kt1",
    f"hh_ggf_{hhdecay}_kl5_kt1",
]
signals_hh_vbf_hhdecay = lambda hhdecay: [
    f"hh_vbf_{hhdecay}_kv1_k2v1_kl1",
    f"hh_vbf_{hhdecay}_kv1_k2v0_kl1",
    f"hh_vbf_{hhdecay}_kv1p74_k2v1p37_kl14p4",
    f"hh_vbf_{hhdecay}_kvm0p012_k2v0p03_kl10p2",
    f"hh_vbf_{hhdecay}_kvm0p758_k2v1p44_klm19p3",
    f"hh_vbf_{hhdecay}_kvm0p962_k2v0p959_klm1p43",
    f"hh_vbf_{hhdecay}_kvm1p21_k2v1p94_klm0p94",
    f"hh_vbf_{hhdecay}_kvm1p6_k2v2p72_klm1p36",
    f"hh_vbf_{hhdecay}_kvm1p83_k2v3p57_klm3p39",
    f"hh_vbf_{hhdecay}_kvm2p12_k2v3p87_klm5p96",
]

signals_hh_ggf = {
    *signals_hh_ggf_hhdecay("hbb_hww"),
    *signals_hh_ggf_hhdecay("hbb_hzz"),
    *signals_hh_ggf_hhdecay("hbb_htt"),
}
signals_hh_vbf = {
    *signals_hh_vbf_hhdecay("hbb_hww"),
    *signals_hh_vbf_hhdecay("hbb_hzz"),
    *signals_hh_vbf_hhdecay("hbb_htt"),
}
signals = {*signals_hh_ggf, *signals_hh_vbf}


# mapping between process names in the config and inference model
# key: config process name, value: inference model process name
# NOTE: when possible, we use the names from the config (only map to inference names if necessary)
inference_procnames = {
    # main backgrounds ( optional name conventions)
    "tt": "ttbar",
    # "st_tchannel": "ST",  # TODO: add st_schannel to ST
    # "st_twchannel": "TW",
    "w_lnu": "W",
    "dy": "DY",
    "vv": "VV",
    "ttv": "ttV",
    "vvv": "VVV",
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

# Higgs BR uncertainties
higgs_br_uncertainties = {
    "hbb": (0.9874, 1.0124),
    "htt": (0.9837, 1.0165),
    "hgg": (0.9791, 1.0205),
    "hww": (0.9848, 1.0153),
    "hzz": (0.9848, 1.0153),
}

# mapping: which processes are used for which QCDscale (rate) uncertainty
# TODO: some uncertainties (e.g. ttz) might still be missing in cmsdb for 13.6 TeV
processes_per_QCDscale = {
    "ttbar": [
        "tt", "st_tchannel", "st_schannel", "st_twchannel", "ttw", "ttz",
        "tttt",
    ],
    "V": ["dy", "dy_lf", "dy_hf", "w_lnu"],
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
    "gg": [
        "tt", "ttz", "ggZZ",
        "tttt",
    ],
    "qqbar": [
        "vv", "ww", "zz", "wz", "st_schannel", "st_tchannel", "dy", "dy_lf", "dy_hf",
        "w_lnu", "vvv", "qqZZ", "ttw",
    ],
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

processes_per_rate_unconstrained = {
    "ttbar_{bjet_cat}": ["tt"],
    "dy_{bjet_cat}": ["dy", "dy_lf", "dy_hf"],
    "ttbar": ["tt"],
    "ttbar_boosted": ["tt"],
    "dy": ["dy", "dy_lf", "dy_hf"],
    "dy_lf": ["dy_lf"],
    "dy_hf": ["dy_hf"],
    "st": ["st_schannel", "st_tchannel", "st_twchannel"],
}

is_signal = lambda process: process.startswith("hh_")
is_bkg = lambda process: not is_signal(process)
is_boosted_ggf = lambda category: "boosted" in category and "sig_vbf" in category
is_boosted_vbf = lambda category: "boosted" in category and "sig_ggf" in category
hbb_efficiency_params = {
    "eff_hbb_signal_ggf": (is_signal, is_boosted_ggf),
    "eff_hbb_signal_vbf": (is_signal, is_boosted_vbf),
    "eff_hbb_bkg_ggf": (is_bkg, is_boosted_ggf),
    "eff_hbb_bkg_vbf": (is_bkg, is_boosted_vbf),
}

# mapping for each shape uncertainty, which process is used.
# If "all" is included, takes all processes except for the ones specified (starting with !)
processes_per_shape = {
    "jec_Total": ["all"],
    "jer": ["all"],
    "jec_Total_{bjet_cat}": ["all"],
    "jer_{bjet_cat}": ["all"],
    "jec_Total_{campaign}": ["all"],
    "jer_{campaign}": ["all"],
    "jec_Total_{year}": ["all"],
    "jer_{year}": ["all"],
    "btag_hf_{bjet_cat}": ["all"],
    "btag_lf_{bjet_cat}": ["all"],
    "btag_hfstats1_{bjet_cat}": ["all"],
    "btag_hfstats2_{bjet_cat}": ["all"],
    "btag_lfstats1_{bjet_cat}": ["all"],
    "btag_lfstats2_{bjet_cat}": ["all"],
    "btag_cferr1_{bjet_cat}": ["all"],
    "btag_cferr2_{bjet_cat}": ["all"],
    "btag_hf_{year}": ["all"],
    "btag_lf_{year}": ["all"],
    "btag_cferr1_{year}": ["all"],
    "btag_cferr2_{year}": ["all"],
    "btag_hf_{campaign}": ["all"],
    "btag_lf_{campaign}": ["all"],
    "btag_cferr1_{campaign}": ["all"],
    "btag_cferr2_{campaign}": ["all"],
    "btag_hfstats1_{campaign}": ["all"],
    "btag_hfstats2_{campaign}": ["all"],
    "btag_lfstats1_{campaign}": ["all"],
    "btag_lfstats2_{campaign}": ["all"],
    "btag_hf": ["all"],
    "btag_lf": ["all"],
    "btag_hfstats1": ["all"],
    "btag_hfstats2": ["all"],
    "btag_lfstats1": ["all"],
    "btag_lfstats2": ["all"],
    "btag_cferr1": ["all"],
    "btag_cferr2": ["all"],
    "mu_id_sf": ["all"],
    "mu_iso_sf": ["all"],
    "e_sf": ["all"],
    "e_reco_sf": ["all"],
    "trigger_sf": ["all"],
    "mu_id_sf_{campaign}": ["all"],
    "mu_iso_sf_{campaign}": ["all"],
    "e_sf_{campaign}": ["all"],
    "e_reco_sf_{campaign}": ["all"],
    "trigger_sf_{campaign}": ["all"],
    "mu_id_sf_{year}": ["all"],
    "mu_iso_sf_{year}": ["all"],
    "e_sf_{year}": ["all"],
    "e_reco_sf_{year}": ["all"],
    "trigger_sf_{year}": ["all"],
    "minbias_xs": ["all"],
    # "isr": ["all"],
    # "fsr": ["all"],
    # NOTE: "!" did not work to exclude processes
    "isr": ["all", "!h_ggf", "!h_vbf"],  # NOTE: skip h_ggf and h_vbf because PSWeights missing in H->tautau
    # "fsr": ["all", "!h_ggf", "!h_vbf"],  # NOTE: skip h_ggf and h_vbf because PSWeights missing in H->tautau
    "fsr_ttbar": ["tt"],
    "fsr_st": ["st_schannel", "st_twchannel"],
    # "fsr_dy": ["dy", "dy_lf", "dy_hf"],
    # "fsr_w": ["w_lnu"],
    "fsr_V": ["dy", "dy_lf", "dy_hf", "w_lnu"],
    "fsr_VV": ["vv", "ww", "zz", "wz"],
    "fsr_ttV": ["ttw", "ttz"],
    "fsr_H": ["h", "vh", "wh", "zh", "tth", "bbh"],  # NOTE: skip h_ggf and h_vbf because PSWeights missing in H->tautau  # noqa: E501
    # "fsr_h": ["h", "h_ggf", "h_vbf", "vh", "wh", "zh", "tth", "bbh"],
    "top_pt": ["tt"],
    "dy_correction": ["dy", "dy_lf", "dy_hf"],
    # "pdf_shape_{proc}": ["{proc}"],
    # "murf_envelope_{proc}": ["{proc}"],
    "pdf_shape_ttbar": ["tt"],
    "pdf_shape_st": ["st_schannel", "st_twchannel"],  # TODO: there was some bug with "st_tchannel"
    "pdf_shape_dy": ["dy", "dy_lf", "dy_hf"],
    "pdf_shape_w": ["w_lnu"],
    "pdf_shape_VV": ["vv", "ww", "zz", "wz"],
    "pdf_shape_ttV": [
        # "ttw",  # NOTE: ttW has no murf/pdf weights
        "ttz",
    ],
    "pdf_shape_H": ["h", "h_ggf", "h_vbf", "vh", "wh", "zh", "tth", "bbh"],
    # NOTE: do I need a separate systematic per parameter variation?
    "pdf_shape_hh_ggf_hbb_hww": sorted(signals_hh_ggf_hhdecay("hbb_hww")),
    "pdf_shape_hh_ggf_hbb_hzz": sorted(signals_hh_ggf_hhdecay("hbb_hzz")),
    "pdf_shape_hh_ggf_hbb_htt": sorted(signals_hh_ggf_hhdecay("hbb_htt")),
    # "pdf_shape_hh_vbf_hbb_hww": sorted(signals_hh_vbf_hhdecay("hbb_hww")),
    # "pdf_shape_hh_vbf_hbb_hzz": sorted(signals_hh_vbf_hhdecay("hbb_hzz")),
    # "pdf_shape_hh_vbf_hbb_htt": sorted(signals_hh_vbf_hhdecay("hbb_htt")),
    "murf_envelope_ttbar": ["tt"],
    "murf_envelope_st": ["st_schannel", "st_tchannel", "st_twchannel"],
    "murf_envelope_dy": ["dy", "dy_lf", "dy_hf"],
    "murf_envelope_w": ["w_lnu"],
    "murf_envelope_VV": ["vv", "ww", "zz", "wz"],
    "murf_envelope_ttV": [
        # "ttw",  # NOTE: ttW has no murf/pdf weights
        "ttz",
    ],
    "murf_envelope_H": ["h", "h_ggf", "h_vbf", "vh", "wh", "zh", "tth", "bbh"],
    # NOTE: do I need a separate systematic per parameter variation?
    "murf_envelope_hh_ggf_hbb_hww": sorted(signals_hh_ggf_hhdecay("hbb_hww")),
    "murf_envelope_hh_ggf_hbb_hzz": sorted(signals_hh_ggf_hhdecay("hbb_hzz")),
    "murf_envelope_hh_ggf_hbb_htt": sorted(signals_hh_ggf_hhdecay("hbb_htt")),
    # "murf_envelope_hh_vbf_hbb_hww": sorted(signals_hh_vbf_hhdecay("hbb_hww")),
    # "murf_envelope_hh_vbf_hbb_hzz": sorted(signals_hh_vbf_hhdecay("hbb_hzz")),
    # "murf_envelope_hh_vbf_hbb_htt": sorted(signals_hh_vbf_hhdecay("hbb_htt")),
}

remove_processes = {
    # NOTE: empty, because it is better to remove processes on the final datacards
    # "dy_lf": {"category": "sr__2b__ml_sig_vbf", "category_match_mode": all},
    # "st_schannel": {"category": "sr__2b__ml_*", "category_match_mode": all},
    # "w_lnu": {"category": "sr__2b__ml_*", "category_match_mode": all},
    # "ttw": {"category": "sr__2b__ml_dy_m10toinf", "category_match_mode": all},
}

# mapping for each shape uncertainty, which shift source is used
# per default: shape and source have the same name (except for pdf and murf, which are implemented per process)
source_per_shape = {shape: shape for shape in processes_per_shape.keys()}
for shape in processes_per_shape.keys():
    if "pdf_shape" in shape:
        source_per_shape[shape] = "pdf"
    elif "murf_envelope" in shape:
        source_per_shape[shape] = "murf_envelope"
    elif "fsr_" in shape:
        source_per_shape[shape] = "fsr"


# naming convention for different campaigns
campaign_format = {
    "2022preEE": "2022",
    "2022postEE": "2022EE",
    "2023preBPix": "2023",
    "2023postBPix": "2023BPix",
}

# naming scheme for different systematics
custom_uncertainty_format = lambda s: f"CMS_HIG25018_{s}"
rename_systematics = {
    "minbias_xs": "CMS_pileup",
    # "ele_ss": "CMS_res_e_13p6TeV",
    "isr": "ps_isr",
    "top_pt": "top_pt_reweighting",
}
for campaign, campaign_fmt in campaign_format.items():
    year = campaign_fmt[:4]
    rename_systematics.update({
        f"e_sf_{campaign}": f"CMS_eff_e_id_{campaign_fmt}",
        f"e_reco_sf_{campaign}": f"CMS_eff_e_reco_{campaign_fmt}",
        f"mu_id_sf_{campaign}": f"CMS_eff_m_id_{campaign_fmt}",
        f"mu_iso_sf_{campaign}": f"CMS_eff_m_iso_{campaign_fmt}",
        # f"trigger_sf_{campaign}": f"CMS_eff_trigger_{campaign_fmt}",
        f"btag_hf_{campaign}": f"CMS_btag_fullShape_hf_{campaign_fmt}",
        f"btag_lf_{campaign}": f"CMS_btag_fullShape_lf_{campaign_fmt}",
        f"btag_hfstats1_{campaign}": f"CMS_btag_fullShape_hfstats1_{campaign_fmt}",
        f"btag_hfstats2_{campaign}": f"CMS_btag_fullShape_hfstats2_{campaign_fmt}",
        f"btag_lfstats1_{campaign}": f"CMS_btag_fullShape_lfstats1_{campaign_fmt}",
        f"btag_lfstats2_{campaign}": f"CMS_btag_fullShape_lfstats2_{campaign_fmt}",
        f"btag_cferr1_{campaign}": f"CMS_btag_fullShape_cferr1_{campaign_fmt}",
        f"btag_cferr2_{campaign}": f"CMS_btag_fullShape_cferr2_{campaign_fmt}",
        f"jec_Total_{campaign}": f"CMS_scale_j_{campaign_fmt}",
        f"jer_{campaign}": f"CMS_res_j_{campaign_fmt}",
        # analysis-specific
        f"trigger_sf_{campaign}": f"CMS_HIG25018_trigger_sf_{campaign_fmt}",
    })

for process in ["ttbar", "st", "dy", "V", "VV", "ttV", "VVV", "H"]:
    rename_systematics[f"fsr_{process}"] = f"ps_fsr_{process}"

rename_systematics_inverted = {v: k for k, v in rename_systematics.items()}
