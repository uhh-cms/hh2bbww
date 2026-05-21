# coding: utf-8

from hbw.util import bracket_expansion
from hbw.config.defaults_and_groups import set_dl_config_defaults_and_groups


def set_dl_hh_config_defaults_and_groups(config_inst):
    set_dl_config_defaults_and_groups(config_inst)
    config_inst.x.default_dataset = "hh_ggf_hbb_hvv_kl1_kt1_powheg"
    # signal_tag = "qqlnu" if config_inst.has_tag("is_sl") else "2l2nu"
    default_signal_process = "hh_ggf_hbb_hvv_kl1_kt1"
    # signal_generator = "powheg"
    # backgrounds0 = ["other", "h", "ttv", "vv", "w_lnu", "st", "dy_m4to10", "dy_m10to50", "dy_m50toinf", "tt"]
    # backgrounds1 = ["other", "h", "ttv", "vv", "w_lnu", "st", "dy_lf", "dy_hf", "tt"]
    hbbhww_sm = ["hh_ggf_hbb_hww_kl1_kt1", "hh_vbf_hbb_hww_kv1_k2v1_kl1"]
    # hbbhww_variations = [
    #     "hh_ggf_hbb_hww_kl0_kt1",
    #     "hh_ggf_hbb_hww_kl1_kt1",
    #     "hh_ggf_hbb_hww_kl2p45_kt1",
    #     "hh_ggf_hbb_hww_kl5_kt1",
    #     "hh_vbf_hbb_hww_kv1_k2v1_kl1",
    # ]
    # hh_sm = [
    #     "hh_ggf_hbb_hww_kl1_kt1", "hh_vbf_hbb_hww_kv1_k2v1_kl1",
    #     "hh_ggf_hbb_hzz_kl1_kt1", "hh_vbf_hbb_hzz_kv1_k2v1_kl1",
    #     "hh_ggf_hbb_htt_kl1_kt1", "hh_vbf_hbb_htt_kv1_k2v1_kl1",
    # ]
    hh_sm1 = [
        "hh_ggf_kl1_kt1", "hh_vbf_kv1_k2v1_kl1",
    ]

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    config_inst.x.process_groups = {
        # Collection of VBF samples with most shape and rate difference
        "gen_vbf": [
            "hh_vbf_hbb_hww2l2nu_kvm0p758_k2v1p44_klm19p3",
            "hh_vbf_hbb_hww2l2nu_kv1_k2v1_kl1",
            "hh_vbf_hbb_hww2l2nu_kv1_k2v0_kl1",
            "hh_vbf_hbb_hww2l2nu_kvm0p962_k2v0p959_klm1p43",
        ],
        "ml_study": [
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
            "st",
            "tt",
            "dy_m4to10", "dy_m10to50", "dy_m50toinf",
            "w_lnu",
            "vv",
            "h_ggf", "h_vbf", "zh", "wh", "zh_gg", "tth",
        ],
        # Collection of all VBF samples present
        "vbf_only": [
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
        ],
        "all": ["*"],
        "default": ["hh_ggf_hbb_hvv_kl1_kt1", "hh_vbf_hbb_hvv_kv1_k2v1_kl1", "h", "vv", "w_lnu", "st", "dy", "tt"],  # noqa: E501
        "sl": ["hh_ggf_hbb_hvv_kl1_kt1", "hh_vbf_hbb_hvv_kv1_k2v1_kl1", "h", "vv", "w_lnu", "dy", "st", "qcd", "tt"],  # noqa: E501
        "dl": ["hh_ggf_hbb_hvv_kl1_kt1", "hh_vbf_hbb_hvv_kv1_k2v1_kl1", "h", "vv", "w_lnu", "st", "dy", "tt"],  # noqa: E501
        "dl1": [default_signal_process, "h", "ttv", "vv", "w_lnu", "st", "dy", "tt"],
        "dl2": [*hbbhww_sm, "h", "ttv", "vv", "w_lnu", "st", "dy_m4to10", "dy_m10to50", "dy_m50toinf", "tt"],  # noqa: E501
        "dl3": [*hh_sm1, "h", "ttv", "vv", "w_lnu", "st", "dy_m4to10", "dy_m10to50", "dy_m50toinf", "tt"],  # noqa: E501
        "dl4": [*hbbhww_sm, "other", "h", "ttv", "vv", "w_lnu", "st", "dy_lf", "dy_hf", "tt"],  # noqa: E501
        "dl41": [*hbbhww_sm, "other", "h", "ttv", "vv", "w_lnu", "st", "dy", "tt"],  # noqa: E501
        "dl42": [*hbbhww_sm, "other", "h", "ttv", "vv", "w_lnu", "st", "dy_m4to10", "dy_m10to50", "dy_m50toinf", "tt"],  # noqa: E501
        "dl6": [*hh_sm1, "other", "h", "ttv", "vv", "w_lnu", "st", "dy_lf", "dy_hf", "tt"],  # noqa: E501
        # "dl7": ["other", "h", "ttv", "vv", "w_lnu", "st", "dy_lf", "dy_hf", "tt"],  # noqa: E501
        "dl9": [*hbbhww_sm, "hh_other", "other", "h", "ttv", "vv", "w_lnu", "st", "dy_lf", "dy_hf", "ttbb_custom", "tt_custom"],  # noqa: E501
        "dl91": [*hbbhww_sm, "hh_other", "other", "h", "ttv", "vv", "w_lnu", "st", "dy_lf", "dy_hf", "tt"],  # noqa: E501
        "dl92": [*hh_sm1, "hh_other", "other", "h", "ttv", "vv", "w_lnu", "st", "dy_tautau_m10to50", "dy_ee_m10to50", "dy_mumu_m10to50", "dy_tautau_m50toinf", "dy_ee_m50toinf", "dy_mumu_m50toinf", "tt"],  # noqa: E501
    }
    for proc, datasets in config_inst.x.dataset_names.items():
        remove_generator = lambda x: x.replace("_powheg", "").replace("_madgraph", "").replace("_amcatnlo", "").replace("_pythia8", "").replace("4f_", "")  # noqa: E501
        config_inst.x.process_groups[f"datasets_{proc}"] = [remove_generator(dataset) for dataset in datasets]

    for group in ("dl9", "dl91", "dl92", "dl6", "dl4", "dl3", "dl2", "dl1", "dl"):  # noqa: E501
        config_inst.x.process_groups[f"d{group}"] = ["data"] + config_inst.x.process_groups[group]

    # category groups for conveniently looping over certain categories
    # (used during plotting and for rebinning)
    config_inst.x.category_groups = {
        "sl": ["sr__1e", "sr__1mu"],
        "sl_resolved": ["sr__1e__resolved", "sr__1mu__resolved"],
        "sl_much": ["sr__1mu", "sr__1mu__1b", "sr__1mu__2b"],
        "sl_ech": ["sr__1e", "sr__1e__1b", "sr__1e__2b"],
        "sl_much_resolved": ["sr__1mu__resolved", "sr__1mu__resolved__1b", "sr__1mu__resolved__2b"],
        "sl_ech_resolved": ["sr__1e__resolved", "sr__1e__resolved__1b", "sr__1e__resolved__2b"],
        "sl_much_boosted": ["sr__1mu__boosted"],
        "sl_ech_boosted": ["sr__1e__boosted"],
        "dl": ["sr", "dycr", "ttcr", "sr__1b", "sr__2b", "dycr__1b", "dycr__2b", "ttcr__1b", "ttcr__2b"],
        "dl_preml_incl": bracket_expansion(["incl", "{,2e__,2mu__,emu__}resolved{,__1b,__2b}"]),
        "dl_preml_small": bracket_expansion(["incl", "{sr,ttcr,dycr}{,__2e,__2mu,__emu}__resolved{,__1b,__2b}"]),
        "dl_preml_large": bracket_expansion(["incl", "{,sr__,ttcr__,dycr__}{,2e__,2mu__,emu__}resolved{,__1b,__2b}"]),
        "dl_preml_1": bracket_expansion(["incl", "{,sr,ttcr,dycr}__{,2e,2mu,emu}"]),
        "dl_preml_boosted": bracket_expansion(["{,sr__,ttcr__,dycr__}{,2e__,2mu__,emu__}boosted"]),
        "dl_ttcr": ["ttcr", "ttcr__1b", "ttcr__2b", "ttcr__2e", "ttcr__2mu", "ttcr__emu"],
        "dl_dycr": ["dycr", "dycr__1b", "dycr__2b", "dycr__2e", "dycr__2mu", "dycr__emu"],
        "dl_sr": ["sr", "sr__1b", "sr__2b", "sr__2e", "sr__2mu", "sr__emu"],
        "dl_resolved": ["sr__resolved", "sr__2e__resolved", "sr__2mu__resolved", "sr__emu__resolved"],
        "dl_2much": ["sr__2mu", "sr__2mu__1b", "sr__2mu__2b", "dycr__2mu", "dycr__2mu__1b", "dycr__2mu__2b", "ttcr__2mu", "ttcr__2mu__1b", "ttcr__2mu__2b"],  # noqa: E501
        "dl_2ech": ["sr__2e", "sr__2e__1b", "sr__2e__2b", "dycr__2e", "dycr__2e__1b", "dycr__2e__2b", "ttcr__2e", "ttcr__2e__1b", "ttcr__2e__2b"],  # noqa: E501
        "dl_emuch": ["sr__emu", "sr__emu__1b", "sr__emu__2b", "dycr__emu", "dycr__emu__1b", "dycr__emu__2b", "ttcr__emu", "ttcr__emu__1b", "ttcr__emu__2b"],  # noqa: E501
        "dl_2much_resolved": ["sr__2mu__resolved", "sr__2mu__resolved__1b", "sr__2mu__resolved__2b"],
        "dl_2ech_resolved": ["sr__2e__resolved", "sr__2e__resolved__1b", "sr__2e__resolved__2b"],
        "dl_emuch_resolved": ["sr__emu__resolved", "sr__emu__resolved__1b", "sr__emu__resolved__2b"],
        "dl_2much_boosted": ["sr__2mu__boosted"],
        "dl_2ech_boosted": ["sr__2e__boosted"],
        "dl_emuch_boosted": ["sr__emu__boosted"],
        "default": ["incl", "sr__1e", "sr__1mu"],
        "test": ["incl", "sr__1e"],
        "dilep": ["incl", "sr__2e", "sr__2mu", "sr__emu"],
        # Single lepton
        "SR_sl": (
            "sr__1e__1b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1", "sr__1mu__1b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
            "sr__1e__2b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1", "sr__1mu__2b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
        ),
        "vbfSR_sl": (
            "sr__1e__1b__ml_hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1", "sr__1mu__1b__ml_hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1",
            "sr__1e__2b__ml_hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1", "sr__1mu__2b__ml_hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1",
        ),
        "SR_sl_resolved": (
            "sr__1e__resolved__1b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
            "sr__1mu__resolved__1b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
            "sr__1e__resolved__2b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
            "sr__1mu__resolved__2b__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
        ),
        "vbfSR_sl_resolved": (
            "sr__1e__resolved__1b__ml_hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1",
            "sr__1mu__resolved__1b__ml_hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1",
            "sr__1e__resolved__2b__ml_hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1",
            "sr__1mu__resolved__2b__ml_hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1",
        ),
        "SR_sl_boosted": (
            "sr__1e__boosted__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1", "sr__1mu__boosted__ml_hh_ggf_hbb_hvvqqlnu_kl1_kt1",
        ),
        "vbfSR_sl_boosted": (
            "sr__1e__ml_boosted_hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1",
            "sr__1mu__ml_boosted_hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1",
        ),
        "BR_sl": (
            "sr__1e__ml_tt", "sr__1e__ml_st", "sr__1e__ml_v_lep",
            "sr__1mu__ml_tt", "sr__1mu__ml_st", "sr__1mu__ml_v_lep",
        ),
        # Dilepton
        "SR_bjets_incl": bracket_expansion(["sr__ml_{signal_ggf2,sig_ggf,hh_ggf_hbb_hvv2l2nu_kl1_kt1,hh_ggf_kl1_kt1}"]),
        "vbfSR_bjets_incl": bracket_expansion(["sr__ml_{signal_vbf2,sig_vbf,hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1,hh_vbf_kv1_k2v1_kl1}"]),  # noqa: E501
        "SR_dl": bracket_expansion(["sr__{1b,2b}__ml_{signal_ggf2,sig_ggf,hh_ggf_hbb_hvv2l2nu_kl1_kt1,hh_ggf_kl1_kt1}"]),  # noqa: E501
        "vbfSR_dl": bracket_expansion(["sr__{1b,2b}__ml_{signal_vbf2,sig_vbf,hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1,hh_vbf_kv1_k2v1_kl1}"]),  # noqa: E501
        "SR_dl_resolved": bracket_expansion(["sr__resolved__{1b,2b}__ml_{signal_ggf2,sig_ggf,hh_ggf_hbb_hvv2l2nu_kl1_kt1,hh_ggf_kl1_kt1}"]),  # noqa: E501
        "SR_hhh_resolved": bracket_expansion(["sr__resolved__{3b,4b}__ml_{hhh_signal,tt_ml,tth,tthh_4b}", "sr__resolved__2b__ml_{hhh_signal,tt_ml,tth,tthh_4b,dy_st}"]),  # noqa: 
        "vbfSR_dl_resolved": bracket_expansion(["sr__resolved__{1b,2b}__ml_{signal_vbf2,sig_vbf,hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1,hh_vbf_kv1_k2v1_kl1}"]),  # noqa: E501
        # "SR_1b_dl": bracket_expansion(["sr__1b__ml_{signal_ggf2,sig_ggf,hh_ggf_hbb_hvv2l2nu_kl1_kt1,hh_ggf_kl1_kt1}"]),  # noqa: E501
        # "vbfSR_1b_dl": bracket_expansion(["sr__1b__ml_{signal_vbf2,sig_vbf,hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1,hh_vbf_kv1_k2v1_kl1}"]),  # noqa: E501
        # "SR_2b_dl_resolved": bracket_expansion(["sr__resolved__2b__ml_{signal_ggf2,sig_ggf,hh_ggf_hbb_hvv2l2nu_kl1_kt1,hh_ggf_kl1_kt1}"]),  # noqa: E501
        # "vbfSR_2b_dl_resolved": bracket_expansion(["sr__resolved__2b__ml_{signal_vbf2,sig_vbf,hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1,hh_vbf_kv1_k2v1_kl1}"]),  # noqa: E501
        "SR_dl_boosted": bracket_expansion(["sr__boosted__ml_{signal_ggf2,sig_ggf,hh_ggf_hbb_hvv2l2nu_kl1_kt1,hh_ggf_kl1_kt1}"]),  # noqa: E501
        "vbfSR_dl_boosted": bracket_expansion(["sr__boosted__ml_{signal_vbf2,sig_vbf,hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1,hh_vbf_kv1_k2v1_kl1}"]),  # noqa: E501
        "BR_dl": bracket_expansion(["sr__{resolved__1b,resolved__2b,boosted,1b,2b}__ml_{bkg,tt,st,dy,dy_m10toinf,h}"]),
        "BR_bjets_incl": bracket_expansion(["sr__ml_{tt,st,dy,dy_m10toinf,h}"]),
        "hhh_sr": bracket_expansion(["sr__resolved__{2b,3b,4b}__ml_{sig_hhh,hhh_signal,hhh_4b2w_2l2nu_c30_d40}", "sr__{2b,3b,4b}__ml_sig_all"]),  # noqa: E501
        "hhh_bkg": bracket_expansion(["sr__{2,3,4}b__ml_{tt,st,dy,h,hh,hh_bkg,tthh_4b,tt_custom,ttbb_custom,tt_ml,hh_custom,tth}", "sr__resolved__2b__ml_{tt,st,dy,h,hh_bkg,tthh_4b,tt_custom,ttbb_custom,tt_ml,hh_custom,tth}", "sr__resolved__3b__ml_{tt,st,dy,h,hh_bkg,tthh_4b,tt_custom,ttbb_custom,tt_ml,hh_custom,tth}", "sr__resolved__4b__ml_{tt,st,dy,h,hh_bkg,tthh_4b,tt_custom,ttbb_custom,tt_ml,hh_custom,tth}"]),  # noqa: E501
    }

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    from hbw.ml.derived.ml_dl_dih import input_features as ml_inputs
    config_inst.x.variable_groups = {
        "gen_features": ["gen_hbw.lep0.pt", "gen_hbw.lep1.pt", "gen_hbw.dilep.pt", "gen_hbw.dilep.mass", "gen_hbw.hh.mass"],   # noqa: E501
        "gen_vbf": ["vbfpair.deta", "vbfpair.mass", "gen_sec1_eta", "gen_sec2_eta", "gen_sec1_pt", "gen_sec2_pt"],
        "mli": ["mli_*"],
        "pas": bracket_expansion([
            "mli_{mbb,mbbllMET,bb_pt,b1_pt}_rebinned3",
            "mli_{mll,ll_pt,n_jet}",
            "rebinlogit_mlscore.sig_{ggf,vbf}_binary",
            "mlscore.sig_{ggf,vbf}_binary",
        ]),
        "iso": bracket_expansion(["lepton{0,1}_{pfreliso,minipfreliso,mvatth}"]),
        "sl": ["n_*", "electron_*", "muon_*", "met_*", "jet*", "bjet*", "ht"],
        "sl_resolved": ["n_*", "electron_*", "muon_*", "met_*", "jet*", "bjet*", "ht"],
        "sl_boosted": ["n_*", "electron_*", "muon_*", "met_*", "fatjet_*"],
        "ml_inputs": ml_inputs.v2,  # should correspond to our currently used ML input features
        "ml_inputs_discrete": ml_inputs.v2_discrete + bracket_expansion([
            "mli_{b1,b2,j1,j2}_{pt,eta,discrete_b_score,b_score}",
            "mli_n_btag",
            "mli_{b,l}_discrete_b_score_sum",
        ]),
        "ml_outputs": ["mlscore.*", "rebinlogit_mlscore.sig*binary"],
        "basic_kin": bracket_expansion([
            "{lepton0,lepton1,jet0,fatjet0}_{pt,eta,phi}",
            # "met_{pt,phi}",  # TODO: apply MetCorr to these variables
        ]),
        "dl": bracket_expansion([
            "n_{jet,jet_pt30,bjet,btag,electron,muon,fatjet,hbbjet,vetotau}",
            "lepton{0,1}_{pt,eta,phi,pfreliso,minipfreliso}",  # ,mvatth}",
            "met_{pt,phi}",
            "incljets_{pt,eta}",
            "jet{0,1,2,3}_{pt,eta,phi,mass,btagpnetb}",
            "bjet{0,1}_{pt,eta,phi,mass,btagpnetb}",
            "ht", "lt", "mll", "ptll", "npvs",
        ]),
        "dl_eta_studies": bracket_expansion([
            "n_{jet,jet_pt30,bjet,btag}",
            "lepton{0,1}_{pt,eta}",
            "met_{pt,phi}",
            "jet{0,1,2}_{pt,eta,phi,mass,btagpnetb}",
            "bjet{0,1}_{pt,eta,phi,mass,btagpnetb}",
            "ht", "mll", "ptll",
            "barreljet{0,1,2}_{pt,eta}",
            "endcapjet{0,1,2}_{pt,eta}",
            "barrellep{0,1}_pt",
            "endcaplep{0,1}_pt",
        ]),
        "dl_resolved": ["n_*", "electron_*", "muon_*", "met_*", "jet*", "bjet*", "ht", "lt", "mll", "ptll"],
        "dl_boosted": ["n_*", "electron_*", "muon_*", "met_*", "fatjet_*", "lt", "mll", "ptll"],
        "default": ["n_jet", "n_muon", "n_electron", "ht", "m_bb", "deltaR_bb", "jet1_pt"],  # n_deepjet, ....
        "test": ["n_jet", "n_electron", "jet1_pt"],
        "cutflow": ["cf_jet1_pt", "cf_jet4_pt", "cf_n_jet", "cf_n_electron", "cf_n_muon"],  # cf_n_deepjet
        "dilep": [
            "n_jet", "n_muon", "n_electron", "ht", "m_bb", "m_ll", "deltaR_bb", "deltaR_ll",
            "ll_pt", "bb_pt", "E_miss", "delta_Phi", "MT", "min_dr_lljj",
            "m_lljjMET", "channel_id", "n_bjet", "wp_score", "charge", "m_ll_check",
        ],
    }

    # add all groups from ml inputs to variable groups
    for key, variables in ml_inputs.items():
        config_inst.x.variable_groups[f"ml_inputs_{key}"] = variables

    # plotting settings groups
    # (used in plotting)
    # cms_label = "wip"
    cms_label = "pw"
    config_inst.x.general_settings_groups = {
        "test1": {"p1": True, "p2": 5, "p3": "text", "skip_legend": True},
        "default_norm": {"shape_norm": True, "yscale": "log"},
        "dpostfit_merged": {
            "remove_negative": True,
            "whitespace_fraction": 0.35,
            "cms_label": f"{cms_label}",
            "yscale": "log",
            "hide_signal_errors": True,
            # "lumi": "62",  # NOTE: hard-coded for now (to be removed/changed when running on other years)
            "magnitudes": 5.5,
            # "blinding_threshold": 0.008,
        },
        "postfit_merged": {
            "remove_negative": True,
            "whitespace_fraction": 0.35,
            "cms_label": f"sim{cms_label}",
            "yscale": "log",
            "hide_signal_errors": True,
            # "lumi": "109",  # NOTE: hard-coded for now (to be removed/changed when running on other years)
            "magnitudes": 5.5,
            # "blinding_threshold": 0.008,
        },
        "dpostfit": {
            "remove_negative": True,
            # "whitespace_fraction": 0.40,
            "whitespace_fraction": 0.44,
            "cms_label": f"{cms_label}",
            "yscale": "log",
            "hide_signal_errors": True,
            # "lumi": "62",  # NOTE: hard-coded for now (to be removed/changed when running on other years)
            # "blinding_threshold": 0.008,
        },
        "postfit": {
            "remove_negative": True,
            "whitespace_fraction": 0.44,
            "cms_label": f"sim{cms_label}",
            "yscale": "log",
            "hide_signal_errors": True,
            # "lumi": "62",  # NOTE: hard-coded for now (to be removed/changed when running on other years)
            # "blinding_threshold": 0.008,
        },
        "data_mc_plots": {
            "remove_negative": True,
            # "custom_style_config": "default",  # NOTE: does not work in combination with group
            "whitespace_fraction": 0.4,
            "cms_label": f"{cms_label}",
            "yscale": "log",
            "blinding_threshold": 0.008,
        },
        "data_mc_plots_not_blinded": {
            "remove_negative": True,
            # "custom_style_config": "default",  # NOTE: does not work in combination with group
            "whitespace_fraction": 0.4,
            "cms_label": f"{cms_label}",
            "yscale": "log",
        },
        "more_magnitudes": {
            "remove_negative": True,
            # "custom_style_config": "default",  # NOTE: does not work in combination with group
            "whitespace_fraction": 0.2,
            "cms_label": f"{cms_label}",
            "yscale": "log",
            "blinding_threshold": 0.008,
            "magnitudes": 8,
        },
        "data_mc_plots_blind_conservative": {
            "remove_negative": True,
            # "custom_style_config": "default",  # NOTE: does not work in combination with group
            "whitespace_fraction": 0.4,
            "cms_label": f"{cms_label}",
            "yscale": "log",
            "blinding_threshold": 0.004,
        },
        "unstacked": {
            "remove_negative": True,
            "whitespace_fraction": 0.4,
            "cms_label": f"sim{cms_label}",
            "yscale": "log",
            "shape_norm": True,
        },
    }

    config_inst.x.process_settings_groups = {
        "default": {default_signal_process: {"scale": 2000, "unstack": True}},
        "unstack_all": {proc.name: {"unstack": True} for proc, _, _ in config_inst.walk_processes()},
        "unstack_signal": {proc.name: {"unstack": True} for proc in config_inst.processes if "HH" in proc.name},
        "scale_signal": {
            proc.name: {"unstack": True, "scale": 10000}
            for proc, _, _ in config_inst.walk_processes() if proc.has_tag("is_signal")
        },
        "scale_signal1": {
            proc.name: {"unstack": True, "scale": "stack"}
            for proc, _, _ in config_inst.walk_processes() if proc.has_tag("is_signal")
        },
        "data_split_in_era": {
            proc.name: {"unstack": True}
            for proc, _, _ in config_inst.walk_processes()
        },
        "dilep": {
            "hh_vbf_hbb_hww2l2nu": {"scale": 90000, "unstack": True},
            "hh_ggf_hbb_hww2l2nu": {"scale": 10000, "unstack": True},
        },
        "dileptest": {
            "hh_ggf_hbb_hvv2l2nu_kl1_kt1": {"scale": 10000, "unstack": True},
        },
        "control": {
            "hh_ggf_hbb_hvvqqlnu_kl0_kt1": {"scale": 90000, "unstack": True},
            "hh_ggf_hbb_hvvqqlnu_kl1_kt1": {"scale": 90000, "unstack": True},
            "hh_ggf_hbb_hvvqqlnu_kl2p45_kt1": {"scale": 90000, "unstack": True},
            "hh_ggf_hbb_hvvqqlnu_kl5_kt1": {"scale": 90000, "unstack": True},
        },
    }

    config_inst.x.variable_settings_groups = {
        "none": {},
        "test": {
            "mli_mbb": {"rebin": 2, "label": "test"},
            "mli_mjj": {"rebin": 2},
        },
        "boosted_rebin": {
            var: {"rebin": 4}
            for var in (ml_inputs.v2 + [
                "mli_full_vbf_deta",
                "mli_full_vbf_mass",
                "mli_ht_alljets",
                "mli_maxdr_jj_alljets",  # likely bad modelled
            ])
            if not var.startswith("mli_n_") and not var == "mli_mixed_channel"
        },
        "rebin_ml_scores100": {
            # var: {"rebin": 100}
            var: {"rebin": 4}
            for var in [
                "rebinlogit_mlscore.sig_ggf_binary",
                "rebinlogit_mlscore.sig_vbf_binary",
                "mlscore.max_score",
                "mlscore.sig_ggf_binary",
                "mlscore.sig_vbf_binary",
                "mlscore.sig_ggf",
                "mlscore.sig_vbf",
                "mlscore.tt",
                "mlscore.st",
                "mlscore.dy_m10toinf",
                "mlscore.dy",
                "mlscore.h",
            ]
        },
    }

    # groups are defined via config.x.category_groups
    config_inst.x.default_bins_per_category = {
        # Single lepton
        # "SR_sl": 10,
        # "vbfSR_sl": 5,
        # "BR_sl": 3,
        # "SR_sl_resolved": 10,
        # "SR_sl_boosted": 5,
        # "vbfSR_sl_resolved": 5,
        # "vbfSR_sl_boosted": 3,
        # Dilepton
        "BR_dl": 1,
        "sr_boosted_bkg": 1,
        "sr__resolved__1b__ml_sig_ggf": 12,
        "sr__resolved__2b__ml_sig_ggf": 8,
        "sr__resolved__1b__ml_sig_vbf": 10,
        "sr__resolved__2b__ml_sig_vbf": 8,
        "sr__1b__ml_sig_ggf": 10,
        "sr__2b__ml_sig_ggf": 6,
        "sr__1b__ml_sig_vbf": 8,
        "sr__2b__ml_sig_vbf": 6,
        "sr__boosted__ml_sig_ggf": 3,
        "sr__boosted__ml_sig_vbf": 3,
        "sr__boosted": 3,
        "hhh_bkg": 1,
        "hhh_sr": 10,
    }

    is_signal_sm = lambda proc_name: "kl1_kt1" in proc_name or "kv1_k2v1_kl1" in proc_name
    is_signal_sm_ggf = lambda proc_name: "kl1_kt1" in proc_name
    is_signal_sm_vbf = lambda proc_name: "kv1_k2v1_kl1" in proc_name
    is_signal_hhh = lambda proc_name: "hhh" in proc_name
    # is_gghh_sm = lambda proc_name: "kl1_kt1" in proc_name
    # is_qqhh_sm = lambda proc_name: "kv1_k2v1_kl1" in proc_name
    # is_signal_ggf_kl1 = lambda proc_name: "kl1_kt1" in proc_name and "hh_ggf" in proc_name
    # is_signal_vbf_kl1 = lambda proc_name: "kv1_k2v1_kl1" in proc_name and "hh_vbf" in proc_name
    is_background = lambda proc_name: (
        "hbb_hvv" not in proc_name and "hbb_hww" not in proc_name and
        "hbb_hzz" not in proc_name and "hbb_htt" not in proc_name
    )
    is_background_hhh = lambda proc_name: ("hhh" not in proc_name)

    config_inst.x.inference_category_rebin_processes = {
        # Single lepton
        "SR_sl": is_signal_sm_ggf,
        "vbfSR_sl": is_signal_sm_vbf,
        "SR_sl_resolved": is_signal_sm,
        "SR_sl_boosted": is_signal_sm,
        "vbfSR_sl_resolved": is_signal_sm,
        "vbfSR_sl_boosted": is_signal_sm,
        "BR_sl": is_background,
        # Dilepton
        "SR_dl": is_signal_sm_ggf,
        "vbfSR_dl": is_signal_sm_vbf,
        "BR_dl": is_background,
        "SR_bjets_incl": is_signal_sm_ggf,
        "vbfSR_bjets_incl": is_signal_sm_vbf,
        "BR_bjets_incl": is_background,
        "SR_dl_resolved": is_signal_sm_ggf,
        "SR_dl_boosted": is_signal_sm_ggf,
        "vbfSR_dl_resolved": is_signal_sm_vbf,
        "vbfSR_dl_boosted": is_signal_sm_vbf,
        "sr__1b": is_signal_sm_ggf,
        "sr__2b": is_signal_sm_ggf,
        "sr__boosted": is_signal_sm_vbf,
        "sr_boosted_bkg": is_background,
        "hhh_sr": is_signal_hhh,
        "hhh_bkg": is_background_hhh,
    }
