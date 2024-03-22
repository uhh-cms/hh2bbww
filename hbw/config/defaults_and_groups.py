# coding: utf-8

import law

from columnflow.inference import InferenceModel
from columnflow.tasks.framework.base import RESOLVE_DEFAULT


def default_selector(cls, container, task_params):
    if container.has_tag("is_sl"):
        selector = "sl"
    elif container.has_tag("is_dl"):
        selector = "dl"

    return selector


def default_ml_model(cls, container, task_params):
    """ Function that chooses the default_ml_model based on the inference_model if given """
    # for most tasks, do not use any default ml model
    default_ml_model = None

    # set default ml_model when task is part of the MLTraining pipeline
    # NOTE: default_ml_model does not work for the MLTraining task
    if hasattr(cls, "ml_model"):
        # TODO: we might want to distinguish between multiple default ML models (sl vs dl)
        default_ml_model = "dense_default"

    # check if task is using an inference model
    # if that is the case, use the default ml_model set in the inference model
    if getattr(cls, "inference_model", None):
        inference_model = task_params.get("inference_model", None)

        # if inference model is not set, assume it's the container default
        if inference_model in (None, law.NO_STR, RESOLVE_DEFAULT):
            inference_model = container.x.default_inference_model

        # get the default_ml_model from the inference_model_inst
        inference_model_inst = InferenceModel.get_cls(inference_model)
        default_ml_model = getattr(inference_model_inst, "ml_model_name", default_ml_model)

    return default_ml_model


def ml_inputs_producer(cls, container, task_params):
    if container.has_tag("is_sl") and not container.has_tag("is_resonant"):
        ml_inputs = "ml_inputs"
    if container.has_tag("is_dl"):
        ml_inputs = "dl_ml_inputs"
    if container.has_tag("is_sl") and container.has_tag("is_resonant"):
        ml_inputs = "sl_res_ml_inputs"
    return ml_inputs


def default_producers(cls, container, task_params):
    """ Default producers chosen based on the Inference model and the ML Model """

    # per default, use the ml_inputs and event_weights
    default_producers = [ml_inputs_producer(cls, container, task_params), "event_weights"]

    if hasattr(cls, "ml_model"):
        # do no further resolve the ML categorizer when this task is part of the MLTraining pipeline
        return default_producers

    # check if a mlmodel has been set
    ml_model = task_params.get("ml_models", None)

    # only consider 1 ml_model
    if ml_model and isinstance(ml_model, (list, tuple)):
        ml_model = ml_model[0]

    # try and get the default ml model if not set
    if ml_model in (None, law.NO_STR, RESOLVE_DEFAULT):
        ml_model = default_ml_model(cls, container, task_params)

    # if a ML model is set, and the task is not part of the MLTraining pipeline,
    # use the ml categorization producer
    if ml_model not in (None, law.NO_STR, RESOLVE_DEFAULT, tuple()):
        # NOTE: this producer needs to be added as the last element! otherwise, category_ids will be overwritten
        default_producers.append(f"ml_{ml_model}")

    # if we're running the inference_model, we don't need the ml_inputs
    # NOTE: we cannot skip ml_inputs, because it is needed for cf.MLEvaluation
    # if "inference_model" in task_params.keys():
    #     default_producers.remove("ml_inputs")

    return default_producers


def set_config_defaults_and_groups(config_inst):
    """ Configuration function that sets all the defaults and groups in the config_inst """

    # define the default dataset and process based on the analysis tags
    signal_tag = "sl" if config_inst.has_tag("is_sl") else "dl"
    default_signal_process = f"ggHH_kl_1_kt_1_{signal_tag}_hbbhww"
    signal_generator = "powheg"

    if config_inst.has_tag("resonant"):
        signal_tag = f"res_{signal_tag}"
        # for resonant, rely on the law.cfg to define the default signal process (NOTE: might change in the future)
        default_signal_process = law.config.get_expanded("analysis", "default_dataset")
        signal_generator = "madgraph"

    #
    # Defaults
    #

    # TODO: the default dataset is currently still being set up by the law.cfg
    config_inst.x.default_dataset = default_signal_dataset = f"{default_signal_process}_{signal_generator}"
    config_inst.x.default_calibrator = "skip_jecunc"
    config_inst.x.default_selector = default_selector
    config_inst.x.default_producer = default_producers
    config_inst.x.default_ml_model = default_ml_model
    config_inst.x.default_inference_model = "default"
    config_inst.x.default_categories = ["incl"]
    config_inst.x.default_variables = ["jet1_pt"]

    #
    # Groups
    #

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    config_inst.x.process_groups = {
        "all": ["*"],
        "default": [default_signal_process, "tt", "st", "w_lnu", "dy_lep"],
        "dilep": ["ggHH_kl_*", "tt", "st", "w_lnu", "dy_lep"],
        "with_qcd": [default_signal_process, "tt", "qcd", "st", "w_lnu", "dy_lep"],
        "much": [default_signal_process, "tt", "qcd_mu", "st", "w_lnu", "dy_lep"],
        "ech": [default_signal_process, "tt", "qcd_ele", "st", "w_lnu", "dy_lep"],
        "inference": ["ggHH_*", "tt", "st", "w_lnu", "dy_lep", "qcd_*"],
        "k2v": ["qqHH_*", "tt", "st", "w_lnu", "dy_lep", "qcd_*"],
        "ml": [default_signal_process, "tt", "st", "w_lnu", "dy_lep"],
        "ml_test": [default_signal_process, "st", "w_lnu"],
        "test": [default_signal_process, "tt_sl"],
        "small": [default_signal_process, "tt", "st"],
        "bkg": ["tt", "st", "w_lnu", "dy_lep"],
        "signal": ["ggHH_*", "qqHH_*"], "gghh": ["ggHH_*"], "qqhh": ["qqHH_*"],
    }
    config_inst.x.process_groups["dmuch"] = ["data_mu"] + config_inst.x.process_groups["much"]
    config_inst.x.process_groups["dech"] = ["data_e"] + config_inst.x.process_groups["ech"]

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    config_inst.x.dataset_groups = {
        "all": ["*"],
        "default": [default_signal_dataset, "tt_*", "qcd_*", "st_*", "dy_*", "w_lnu_*"],
        "inference": ["ggHH_*", "tt_*", "qcd_*", "st_*", "dy_*", "w_lnu_*"],
        "test": [default_signal_dataset, "tt_sl_powheg"],
        "small": [default_signal_dataset, "tt_*", "st_*"],
        "bkg": ["tt_*", "st_*", "w_lnu_*", "dy_*"],
        "tt": ["tt_*"], "st": ["st_*"], "w": ["w_lnu_*"], "dy": ["dy_*"],
        "qcd": ["qcd_*"], "qcd_mu": ["qcd_mu*"], "qcd_ele": ["qcd_em*", "qcd_bctoe*"],
        "signal": ["ggHH_*", "qqHH_*"], "gghh": ["ggHH_*"], "qqhh": ["qqHH_*"],
        "ml": ["ggHH_kl_1*", "tt_*", "st_*", "dy_*", "w_lnu_*"],
        "dilep": ["tt_*", "st_*", "dy_*", "ggHH_*", "w_lnu_*"],
    }

    # category groups for conveniently looping over certain categories
    # (used during plotting and for rebinning)
    config_inst.x.category_groups = {
        "much": ["1mu", "1mu__resolved", "1mu__boosted"],
        "ech": ["1e", "1e__resolved", "1e__boosted"],
        "default": ["incl", "1e", "1mu"],
        "test": ["incl", "1e"],
        "dilep": ["incl", "2e", "2mu", "emu"],
        "SR": ("1e__ml_ggHH_kl_1_kt_1_sl_hbbhww", "1mu__ml_ggHH_kl_1_kt_1_sl_hbbhww"),
        "vbfSR": ("1e__ml_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww", "1mu__ml_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww"),
        "SR_resolved": ("1e__ml_resolved_ggHH_kl_1_kt_1_sl_hbbhww", "1mu__ml_resolved_ggHH_kl_1_kt_1_sl_hbbhww"),
        "SR_boosted": ("1e__ml_boosted_ggHH_kl_1_kt_1_sl_hbbhww", "1mu__ml_boosted_ggHH_kl_1_kt_1_sl_hbbhww"),
        "vbfSR_resolved": ("1e__ml_resolved_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww", "1mu__ml_resolved_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww"),  # noqa
        "vbfSR_boosted": ("1e__ml_boosted_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww", "1mu__ml_boosted_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww"),  # noqa
        "BR": ("1e__ml_tt", "1e__ml_st", "1e__ml_v_lep", "1mu__ml_tt", "1mu__ml_st", "1mu__ml_v_lep"),
        "SR_dl": (
            "2e__1b__ml_ggHH_sig",
            "2mu__1b__ml_ggHH_sig",
            "emu__1b__ml_ggHH_sig",
            "2e__2b__ml_ggHH_sig",
            "2mu__2b__ml_ggHH_sig",
            "emu__2b__ml_ggHH_sig",
            "2e__1b__ml_ggHH_sig_all",
            "2mu__1b__ml_ggHH_sig_all",
            "emu__1b__ml_ggHH_sig_all",
            "2e__2b__ml_ggHH_sig_all",
            "2mu__2b__ml_ggHH_sig_all",
            "emu__2b__ml_ggHH_sig_all",
        ),
        "BR_dl": (
            "2e__1b__ml_tt",
            "2e__1b__ml_st",
            "emu__1b__ml_tt",
            "emu__1b__ml_st",
            "2mu__1b__ml_tt",
            "2mu__1b__ml_st",
            "2e__2b__ml_tt",
            "2e__2b__ml_st",
            "emu__2b__ml_tt",
            "emu__2b__ml_st",
            "2mu__2b__ml_tt",
            "2mu__2b__ml_st",
            "2e__1b__ml_dy_lep",
            "emu__1b__ml_dy_lep",
            "2mu__1b__ml_dy_lep",
            "2e__2b__ml_dy_lep",
            "emu__2b__ml_dy_lep",
            "2mu__2b__ml_dy_lep",
        ),
    }

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    config_inst.x.variable_groups = {
        "default": ["n_jet", "n_muon", "n_electron", "ht", "m_bb", "deltaR_bb", "jet1_pt"],  # n_deepjet, ....
        "test": ["n_jet", "n_electron", "jet1_pt"],
        "cutflow": ["cf_jet1_pt", "cf_jet4_pt", "cf_n_jet", "cf_n_electron", "cf_n_muon"],  # cf_n_deepjet
        "dilep": [
            "n_jet", "n_muon", "n_electron", "ht", "m_bb", "m_ll", "deltaR_bb", "deltaR_ll",
            "ll_pt", "bb_pt", "E_miss", "delta_Phi", "MT", "min_dr_lljj",
            "m_lljjMET", "channel_id", "n_bjet", "wp_score", "m_ll_check",
        ],
        "control": [
            "n_jet", "n_fatjet", "n_electron", "n_muon",
            "jet1_pt", "jet1_eta", "jet1_phi", "jet1_btagDeepFlavB",   # "jet1_btagDeepB",
            "jet2_pt", "jet2_eta", "jet2_phi", "jet2_btagDeepFlavB",   # "jet2_btagDeepB",
            "jet3_pt", "jet3_eta", "jet3_phi", "jet3_btagDeepFlavB",   # "jet3_btagDeepB",
            "jet4_pt", "jet4_eta", "jet4_phi", "jet4_btagDeepFlavB",   # "jet4_btagDeepB",
            "fatjet1_pt", "fatjet1_eta", "fatjet1_phi", "fatjet1_btagHbb", "fatjet1_deepTagMD_HbbvsQCD",
            "fatjet1_mass", "fatjet1_msoftdrop", "fatjet1_tau1", "fatjet1_tau2", "fatjet1_tau21",
            "fatjet2_pt", "fatjet2_eta", "fatjet2_phi", "fatjet2_btagHbb", "fatjet2_deepTagMD_HbbvsQCD",
            "fatjet2_mass", "fatjet2_msoftdrop", "fatjet2_tau1", "fatjet2_tau2", "fatjet2_tau21",
            "electron_pt", "electron_eta", "electron_phi", "muon_pt", "muon_eta", "muon_phi",
        ],
    }

    # shift groups for conveniently looping over certain shifts
    # (used during plotting)
    config_inst.x.shift_groups = {
        "jer": ["nominal", "jer_up", "jer_down"],
    }

    # selector step groups for conveniently looping over certain steps
    # (used in cutflow tasks)
    # NOTE: this could be added as part of the selector init itself
    config_inst.x.selector_step_groups = {
        "resolved": ["Trigger", "Lepton", "VetoLepton", "Jet", "Bjet", "VetoTau"],
        "boosted": ["Trigger", "Lepton", "VetoLepton", "FatJet", "Boosted"],
        "default": ["Lepton", "VetoLepton", "Jet", "Bjet", "Trigger"],
        "thesis": ["Lepton", "Muon", "Jet", "Trigger", "Bjet"],  # reproduce master thesis cuts for checks
        "test": ["Lepton", "Jet", "Bjet"],
        "dilep": ["Trigger", "Bjet", "Lepton"],
    }

    # plotting settings groups
    # (used in plotting)
    config_inst.x.general_settings_groups = {
        "test1": {"p1": True, "p2": 5, "p3": "text", "skip_legend": True},
        "default_norm": {"shape_norm": True, "yscale": "log"},
    }
    config_inst.x.process_settings_groups = {
        "default": {default_signal_process: {"scale": 2000, "unstack": True}},
        "unstack_all": {proc.name: {"unstack": True} for proc in config_inst.processes},
        "unstack_signal": {proc.name: {"unstack": True} for proc in config_inst.processes if "HH" in proc.name},
        "scale_signal": {
            proc.name: {"unstack": True, "scale": 10000}
            for proc in config_inst.processes if "HH" in proc.name
        },
        "dilep": {
            "ggHH_kl_0_kt_1_dl_hbbhww": {"scale": 10000, "unstack": True},
            "ggHH_kl_1_kt_1_dl_hbbhww": {"scale": 10000, "unstack": True},
            "ggHH_kl_2p45_kt_1_dl_hbbhww": {"scale": 10000, "unstack": True},
            "ggHH_kl_5_kt_1_dl_hbbhww": {"scale": 10000, "unstack": True},
            "ggHH_sig_all": {"scale": 1000, "unstack": True},
        },
        "dileptest": {
            "ggHH_kl_1_kt_1_dl_hbbhww": {"scale": 10000, "unstack": True},
        },
        "control": {
            "ggHH_kl_0_kt_1_sl_hbbhww": {"scale": 90000, "unstack": True},
            "ggHH_kl_1_kt_1_sl_hbbhww": {"scale": 90000, "unstack": True},
            "ggHH_kl_2p45_kt_1_sl_hbbhww": {"scale": 90000, "unstack": True},
            "ggHH_kl_5_kt_1_sl_hbbhww": {"scale": 90000, "unstack": True},
        },
    }
    # when drawing DY as a line, use a different type of yellow
    config_inst.x.process_settings_groups["unstack_all"].update({"dy_lep": {"unstack": True, "color": "#e6d800"}})

    config_inst.x.variable_settings_groups = {
        "test": {
            "mli_mbb": {"rebin": 2, "label": "test"},
            "mli_mjj": {"rebin": 2},
        },
    }

    # groups for custom plot styling
    config_inst.x.custom_style_config_groups = {
        "example": {
            "legend_cfg": {"title": "my custom legend title", "ncols": 2},
            "ax_cfg": {"ylabel": "my ylabel", "xlim": (0, 100)},
            "rax_cfg": {"ylabel": "some other ylabel"},
            "annotate_cfg": {"text": "category label usually here"},
        },
    }

    # CSP (calibrator, selector, producer) groups
    config_inst.x.producer_groups = {
        "mli": ["ml_inputs", "event_weights"],
        "mlo": ["ml_dense_default", "event_weights"],
        "cols": ["mli", "features"],
    }

    # groups are defined via config.x.category_groups
    config_inst.x.default_bins_per_category = {
        "SR": 10,
        "SR_dl": 10,
        # "vbfSR": 5,
        "BR_dl": 3,
        "BR": 3,
        "SR_resolved": 10,
        "SR_boosted": 5,
        "vbfSR_resolved": 5,
        "vbfSR_boosted": 3,
        # "SR_dl": 10,
        # "BR_dl": 3,
        # "1e__ml_ggHH_kl_1_kt_1_sl_hbbhww": 10,
        # "1e__ml_tt": 3,
        # "1e__ml_st": 3,
        # "1e__ml_v_lep": 3,
        # "1mu__ml_ggHH_kl_1_kt_1_sl_hbbhww": 10,
        # "1mu__ml_tt": 3,
        # "1mu__ml_st": 3,
        # "1mu__ml_v_lep": 3,
    }

    config_inst.x.inference_category_rebin_processes = {
        "SR": ("ggHH_kl_1_kt_1_dl_hbbhww"),
        "SR_dl": ("ggHH_kl_1_kt_0_dl_hbbhww", "ggHH_kl_1_kt_1_dl_hbbhww", "ggHH_kl_2p45_kt_1_dl_hbbhww", "ggHH_kl_5_kt_1_dl_hbbhww"),
        # "vbfSR": ("ggHH_kl_1_kt_1_sl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww"),
        "BR": lambda proc_name: "hbbhww" not in proc_name,
        "BR_dl": lambda proc_name: "hbbhww" not in proc_name,
        # "SR_dl": ("ggHH_kl_5_kt_1_dl_hbbhww",),
        # "BR_dl": lambda proc_name: "hbbhww" not in proc_name,
        # "1e__ml_ggHH_kl_1_kt_1_sl_hbbhww": ("ggHH_kl_1_kt_1_sl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww"),
        # "1e__ml_tt": lambda proc_name: "hbbhww" not in proc_name,
        # "1e__ml_st": lambda proc_name: "hbbhww" not in proc_name,
        # "1e__ml_v_lep": lambda proc_name: "hbbhww" not in proc_name,
        # "1mu__ml_ggHH_kl_1_kt_1_sl_hbbhww":  ("ggHH_kl_1_kt_1_sl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww"),
        # "1mu__ml_tt": lambda proc_name: "hbbhww" not in proc_name,
        # "1mu__ml_st": lambda proc_name: "hbbhww" not in proc_name,
        # "1mu__ml_v_lep": lambda proc_name: "hbbhww" not in proc_name,
    }
