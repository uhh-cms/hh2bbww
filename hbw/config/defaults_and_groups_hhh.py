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
        "dl11": ["hhh_4b2w_2l2nu_c30_d40", "tthh_4b", "vhh_4b", "other", "h", "ttv", "vv", "w_lnu", "st", "dy", "tt"],  # noqa: E501
        "dl12": ["hhh_4b2w_2l2nu_c30_d40", "tthh_4b", "vhh_4b", "other", "h", "ttv", "vv", "w_lnu", "st", "dy", "tt_cc", "tt_lf", "ttbb_b", "ttbb_2b", "ttbb_bb"],  # noqa: E501
        "dl15": ["hhh_4b2w_2l2nu_c30_d40", "tthh_4b", "vhh_4b", "other", "h", "ttv", "vv", "w_lnu", "st", "dy", "ttbb_custom", "tt_custom"],  # noqa: E501
        "dl20": ["hhh_4b2w_2l2nu_c30_d40", "tthh_4b", "vhh_4b", "other", "h", "ttv", "vv", "w_lnu", "st", "dy", "ttb_custom", "tt2b_custom", "tt_bb_custom", "tt_custom"],  # noqa: E501
        "dl17": ["hhh", "tthh_4b", "vhh_4b", "hh_ggf", "hh_vbf", "other", "h", "tth", "ttv", "vv", "w_lnu", "st", "dy", "ttbb_custom", "tt_custom"],  # noqa: E501
        "dl18": ["hhh_4b2w_2l2nu_c30_d40", "tthh_4b", "vhh_4b", "hh_ggf_hbb_hvv_kl1_kt1", "hh_vbf_hbb_hvv_kv1_k2v1_kl1", "other", "h", "tth", "ttv", "vv", "w_lnu", "dy", "ttbb_custom", "tt_custom"],  # noqa: E501
        "dl15C": ["tthh_4b", "vhh_4b", "other", "h", "ttv", "vv", "w_lnu", "st", "dy", "ttbb_custom", "tt_custom"],  # noqa: E501
        "dl15B": ["hhh_4b2w_2l2nu_c30_d40", "tthh_4b", "vhh_4b", "other", "h", "ttv", "vv", "w_lnu", "st", "dy", "tt"],  # noqa: E501
        "dl16": ["hhh_4b2w_2l2nu_c30_d40", "tthh_4b", "vhh_4b", "other", "h", "ttv", "vv", "w_lnu", "st", "dy", "ttbb_dl_1b", "ttbb_sl_1b", "ttbb_fh_1b", "tt_dl_nonb", "tt_sl_nonb", "tt_fh_nonb"],  # noqa: E501
        "dl7": ["hh_vbf_hbb_hvv2l2nu_kvm0p962_k2v0p959_klm1p43", "other", "h", "ttv", "vv", "w_lnu", "dy", "st", "tt"],  # noqa: E501
    }
    for proc, datasets in config_inst.x.dataset_names.items():
        remove_generator = lambda x: x.replace("_powheg", "").replace("_madgraph", "").replace("_amcatnlo", "").replace("_pythia8", "").replace("4f_", "")  # noqa: E501
        config_inst.x.process_groups[f"datasets_{proc}"] = [remove_generator(dataset) for dataset in datasets]

    for group in ("dl1", ):  # noqa: E501
        config_inst.x.process_groups[f"d{group}"] = ["data"] + config_inst.x.process_groups[group]

    # category groups for conveniently looping over certain categories
    # (used during plotting and for rebinning)
    config_inst.x.category_groups = {
        "sr_bcats": ["sr__2b", "sr__3b", "sr__4b"],
        "ml_cats": bracket_expansion(["sr__resolved__{3b,4b}__ml_{hhh_signal,tthh_4b,tt_ml,tth}", "sr__resolved__2b__ml_{hhh_signal,st,dy,tt_ml,tth,tthh_4b}"]),  # noqa: E501
        "hhh_sr": bracket_expansion(["sr__resolved__{2b,3b,4b}__ml_{sig_hhh,hhh_signal,hhh_4b2w_2l2nu_c30_d40}", "sr__{2b,3b,4b}__ml_sig_all"]),  # noqa: E501
        "hhh_bkg": bracket_expansion(["sr__{2,3,4}b__ml_{tt,st,dy,h,hh,hh_bkg,tthh_4b,tt_custom,ttbb_custom,tt_ml,hh_custom,tth}", "sr__resolved__2b__ml_{tt,st,dy,h,hh_bkg,tthh_4b,tt_custom,ttbb_custom,tt_ml,hh_custom,tth}", "sr__resolved__3b__ml_{tt,st,dy,h,hh_bkg,tthh_4b,tt_custom,ttbb_custom,tt_ml,hh_custom,tth}", "sr__resolved__4b__ml_{tt,st,dy,h,hh_bkg,tthh_4b,tt_custom,ttbb_custom,tt_ml,hh_custom,tth}"]),  # noqa: E501
    }

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    from hbw.ml.derived.ml_dl_dih import input_features as ml_inputs
    from hbw.ml.derived.ml_dl_trih import input_features as ml_input_trih
    config_inst.x.variable_groups = {
        "hhh_ml_inputs": ml_input_trih.expanded_hhh_inputs,
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
            "lumi": "62",  # NOTE: hard-coded for now (to be removed/changed when running on other years)
            "magnitudes": 5.5,
            # "blinding_threshold": 0.008,
        },
        "data_mc_plots": {
            "remove_negative": True,
            # "custom_style_config": "default",  # NOTE: does not work in combination with group
            "whitespace_fraction": 0.4,
            "cms_label": f"{cms_label}",
            "yscale": "log",
            "hide_signal_errors": True,
            # "hide_stat_errors": True,
            # "blinding_threshold": 0.00006,
            "blinding_threshold": 0.00003,  # NOTE: good for hhh 2b
            # "blinding_threshold": 0.00005,  # NOTE: good for 3b I think
            # "blinding_threshold": 0.0001,  # NOTE: good for 4b with cat strategy
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
        "hhh_sr": 10,
    }

    is_signal_hhh = lambda proc_name: "hhh_4b2w_2l2nu_c30_d40" in proc_name
    is_background_hhh = lambda proc_name: ("hhh" not in proc_name)

    config_inst.x.inference_category_rebin_processes = {
        "hhh_sr": is_signal_hhh,
        "hhh_bkg": is_background_hhh,
    }
