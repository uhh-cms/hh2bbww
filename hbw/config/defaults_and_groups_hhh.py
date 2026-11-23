# coding: utf-8

from hbw.util import bracket_expansion
from hbw.config.defaults_and_groups import set_dl_config_defaults_and_groups


def set_dl_hhh_config_defaults_and_groups(config_inst):
    set_dl_config_defaults_and_groups(config_inst)

    config_inst.x.default_dataset = "hhh_4b2w_2l2nu_c30_d4_custom"
    # signal_tag = "qqlnu" if config_inst.has_tag("is_sl") else "2l2nu"
    default_signal_process = "hhh_4b2w_2l2nu_c30_d40"
    # signal_generator = "custom"

    config_inst.x.process_groups = {
        # "d": ["hhh_4b2w_2l2nu_c30_d40", "tthh", "vhh_4b", "other", "h_other", "tth", "ttv", "vv", "w_lnu", "st", "dy", "ttb_custom", "tt2b_custom", "tt_bb_custom", "tt_custom"],  # noqa: E501
        "dl0": ["hhh_4b2w_2l2nu_c30_d40", "tthh", "vhh_4b", "other", "h_other", "tth", "ttv", "vv", "w_lnu", "st", "dy", "ttb_custom", "tt2b_custom", "tt_bb_custom", "tt_custom"],  # noqa: E501
    }
    for proc, datasets in config_inst.x.dataset_names.items():
        def remove_generator(x):
            return x.replace("_powheg", "").replace("_madgraph", "").replace("_amcatnlo", "").replace("_pythia8", "").replace("4f_", "")  # noqa: E501
        config_inst.x.process_groups[f"datasets_{proc}"] = [
            remove_generator(dataset) for dataset in datasets]

    for group in ("dl0",):  # noqa: E501
        config_inst.x.process_groups[f"d{group}"] = [
            "data"] + config_inst.x.process_groups[group]

    # category groups for conveniently looping over certain categories
    # (used during plotting and for rebinning)
    config_inst.x.category_groups = {
        "sr_bcats": ["sr__2b", "sr__3b", "sr__4b"],
        "ml_cats": bracket_expansion([
            "sr__resolved__{3b,4b}__ml_{hhh_signal,tthh_4b,tt_ml,tth}",
            "sr__resolved__2b__ml_{hhh_signal,st,dy,tt_ml,tth,tthh_4b}",
        ]),
        "hhh_sr": bracket_expansion([
            "sr__{resolved,resolved_glo}__{2b,2b_1l,1b_1tb,3b,4b}__ml_{sig_hhh,hhh_signal,hhh_4b2w_2l2nu_c30_d40}",
            "sr__{2b,3b,4b}__ml_{sig_all,hhh_signal}",
            "sr__{boosted,boosted_low,boosted_loose,boosted_glo}{,__ml_hhh_signal}",
        ]),
        "hhh_bkg": bracket_expansion([
            "sr__{2b,3b,4b,2b_1l,1b_1tb,}__ml_{tt,st,dy,h,hh,hh_bkg,tthh_4b,tthh,ttbb_custom,tt_ml,hh_custom,tth}",
            "sr__{resolved,resolved_glo}__{2b,2b_1l,1b_1tb}__ml_{tt,st,dy,h,hh_bkg,tthh_4b,tthh,,ttbb_custom,tt_ml,hh_custom,tth}",  # noqa: E501
            "sr__{resolved,resolved_glo}__3b__ml_{tt,st,dy,h,hh_bkg,tthh_4b,tthh,,ttbb_custom,tt_ml,hh_custom,tth}",
            "sr__{resolved,resolved_glo}__4b__ml_{tt,st,dy,h,hh_bkg,tthh_4b,tthh,,ttbb_custom,tt_ml,hh_custom,tth}",
        ]),
        "tt_bkg": bracket_expansion(["sr__3b__ml_tt_custom"]),
    }

    from hbw.inference.inf_dl_trih import systematics as systs
    config_inst.x.shift_sources = {
        "jer": ["nominal", "jer_up", "jer_down"],
        "hhh_up": systs.hhh_shape_ffn_full,
        # "shape_hhh_up": [f"syst_up" for syst in systs.hhh_shape_ffn_full],
    }

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    from hbw.ml.derived.ml_dl_dih import input_features as ml_inputs
    from hbw.ml.derived.ml_dl_trih import input_features as ml_input_trih
    config_inst.x.variable_groups = {
        "hhh_ml_inputs": ml_input_trih.expanded_hhh_inputs,
        "kinematics": bracket_expansion([
            "{lepton0,lepton1,jet0,fatjet0}_{pt,eta,phi}",
            "mli_discrete_b_score_sum",
            "mli_hb_candidate{1,2,3,4}_discrete_b_score",
        ]),
        "checkJME": bracket_expansion([
            "{lepton0,lepton1,jet0,jet1,jet2,jet3}_{pt,eta,phi}", "n_jet", "n_btag", "n_jet_pt25", "mli_min_dr_lb", "mli_met_pt",  # noqa: E501
        ]),
        "dctr_test": [
            # new input features:
            "mli_n_jet",
            "mli_n_btag",
            "mli_n_btag",
            "mli_n_jet",
            "mli_ht",
            "mli_ht_lf",
            "mli_ht_hf",
            "mli_dr_hbb",
            "mli_min_dr_jj",
            "mli_ll_pt",
            "mli_discrete_b_score_sum",
            "mli_hb_candidate1_discrete_b_score",
            "mli_hb_candidate1_pt",
            "mli_hb_candidate1_eta",

            "mli_hb_candidate2_discrete_b_score",
            "mli_hb_candidate2_pt",
            "mli_hb_candidate2_eta",

            "mli_hb_candidate3_discrete_b_score",
            "mli_hb_candidate3_pt",
            "mli_hb_candidate3_eta",

            "mli_hb_candidate4_discrete_b_score",
            "mli_hb_candidate4_pt",
            "mli_hb_candidate4_eta",
        ],
    }

    # add all groups from ml inputs to variable groups
    for key, variables in ml_inputs.items():
        config_inst.x.variable_groups[f"ml_inputs_{key}"] = variables

    # plotting settings groups
    # (used in plotting)
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
            # NOTE: hard-coded for now (to be removed/changed when running on other years)
            "lumi": "62",
            "magnitudes": 5.5,
        },
        "more_legend": {
            "whitespace_fraction": 0.45,
        },
        "remove_neg": {
            "remove_negative": True,
        },
        "data_mc_plots": {
            "remove_negative": True,
            "whitespace_fraction": 0.4,
            "cms_label": f"{cms_label}",
            "yscale": "log",
            "blinding_threshold": 0.0000003,
        },
        "gatja": {
            "hide_stat_errors": True,
        },
    }

    config_inst.x.process_settings_groups = {
        "default": {default_signal_process: {"scale": 2000, "unstack": True}},
        "unstack_all": {proc.name: {"unstack": True} for proc, _, _ in config_inst.walk_processes()},
        "unstack_signal": {proc.name: {"unstack": True} for proc in config_inst.processes if "HHH" in proc.name},
    }

    # groups are defined via config.x.category_groups
    config_inst.x.default_bins_per_category = {
        "hhh_bkg": 1,
        "tt_bkg": 1,
        "hhh_sr": 10,
    }

    def is_signal_hhh(proc_name):
        return "hhh_4b2w_2l2nu_c30_d40" in proc_name

    def is_background_hhh(proc_name):
        return ("hhh" not in proc_name)

    config_inst.x.inference_category_rebin_processes = {
        "hhh_sr": is_signal_hhh,
        "hhh_bkg": is_background_hhh,
        "tt_bkg": is_background_hhh,
    }
