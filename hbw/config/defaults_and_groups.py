# coding: utf-8

from columnflow.inference import InferenceModel


def default_ml_model(container, **task_params):
    """ Function that chooses the default_ml_model based on the inference_model if given """
    # for most tasks, do not use any default ml model
    default_ml_model = None

    # the ml_model parameter is only used by `MLTraining` and `MLEvaluation`, therefore use some default
    # NOTE: default_ml_model does not work for the MLTraining task
    if "ml_model" in task_params.keys():
        default_ml_model = "dense_default"

    # check if task is using an inference model; if that's the case, use the default set in the model
    if "inference_model" in task_params.keys():
        inference_model = task_params.get("inference_model", None)

        # if inference model is not set, assume it's the container default
        if inference_model in (None, "NO_STR"):
            inference_model = container.x.default_inference_model

        # get the default_ml_model from the inference_model_inst
        inference_model_inst = InferenceModel.get_cls(inference_model)
        default_ml_model = getattr(inference_model_inst, "ml_model_name", default_ml_model)

    return default_ml_model


def default_producers(container, **task_params):
    """ Default producers chosen based on the Inference model and the ML Model """

    # per default, use the ml_inputs and event_weights
    default_producers = ["ml_inputs", "event_weights"]

    # check if a ml_model has been set
    ml_model = task_params.get("mlmodel", None) or task_params.get("mlmodels", None)

    # only consider 1 ml_model
    if isinstance(ml_model, (list, tuple)):
        ml_model = ml_model[0]

    # try and get the default ml model if not set
    if ml_model in (None, "NO_STR"):
        ml_model = default_ml_model(container, **task_params)

    # "ml_model" is only in the parameters for MLTraining or MLEvaluation (otherwise it's "ml_models")
    is_ml_task = "ml_model" in task_params

    # if a ML model is set, and the task is neither MLTraining nor MLEvaluation,
    # use the ml categorization producer
    if ml_model not in (None, "NO_STR", tuple()) and not is_ml_task:
        default_producers.insert(0, f"ml_{ml_model}")

    # if we're running the inference_model, we don't need the ml_inputs
    # NOTE: we cannot skip ml_inputs, because it is needed for cf.MLEvaluation
    # if "inference_model" in task_params.keys():
    #     default_producers.remove("ml_inputs")

    return default_producers


def set_config_defaults_and_groups(config_inst):
    """ Configuration function that sets all the defaults and groups in the config_inst """

    # default calibrator, selector, producer, ml model and inference model
    config_inst.x.default_calibrator = "skip_jecunc"
    config_inst.x.default_selector = "default"
    # config_inst.x.default_producer = "ml_inputs"
    config_inst.x.default_producer = default_producers
    # config_inst.x.default_ml_model = "default"
    config_inst.x.default_ml_model = default_ml_model
    config_inst.x.default_inference_model = "default"
    config_inst.x.default_categories = ["incl"]

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    config_inst.x.process_groups = {
        "all": ["*"],
        "default": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "st", "w_lnu", "dy_lep"],
        "with_qcd": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "qcd", "st", "w_lnu", "dy_lep"],
        "much": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "qcd_mu", "st", "w_lnu", "dy_lep"],
        "ech": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "qcd_ele", "st", "w_lnu", "dy_lep"],
        "inference": ["ggHH_*", "tt", "st", "w_lnu", "dy_lep", "qcd_*"],
        "ml": ["ggHH_kl_1_*", "tt", "st", "w_lnu", "dy_lep"],
        "ml_test": ["ggHH_kl_1_*", "st", "w_lnu"],
        "test": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt_sl"],
        "small": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "st"],
        "bkg": ["tt", "st", "w_lnu", "dy_lep"],
        "signal": ["ggHH_*", "qqHH"], "gghh": ["ggHH_*"], "qqhh": ["qqHH_*"],
    }
    config_inst.x.process_groups["dmuch"] = ["data_mu"] + config_inst.x.process_groups["much"]
    config_inst.x.process_groups["dech"] = ["data_e"] + config_inst.x.process_groups["ech"]

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    config_inst.x.dataset_groups = {
        "all": ["*"],
        "default": ["ggHH_kl_1*", "tt_*", "qcd_*", "st_*", "dy_*", "w_lnu_*"],
        "inference": ["ggHH_*", "tt_*", "qcd_*", "st_*", "dy_*", "w_lnu_*"],
        "test": ["ggHH_kl_1*", "tt_sl_powheg"],
        "small": ["ggHH_kl_1*", "tt_*", "st_*"],
        "bkg": ["tt_*", "st_*", "w_lnu_*", "dy_*"],
        "tt": ["tt_*"], "st": ["st_*"], "w": ["w_lnu_*"], "dy": ["dy_*"],
        "qcd": ["qcd_*"], "qcd_mu": ["qcd_mu*"], "qcd_ele": ["qcd_em*", "qcd_bctoe*"],
        "signal": ["ggHH_*", "qqHH_*"], "gghh": ["ggHH_*"], "qqhh": ["qqHH_*"],
        "ml": ["ggHH_kl_1*", "tt_*", "st_*", "dy_*", "w_lnu_*"],
    }

    # category groups for conveniently looping over certain categories
    # (used during plotting)
    config_inst.x.category_groups = {
        "much": ["1mu", "1mu__resolved", "1mu__boosted"],
        "ech": ["1e", "1e__resolved", "1e__boosted"],
        "default": ["incl", "1e", "1mu"],
        "test": ["incl", "1e"],
    }

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    config_inst.x.variable_groups = {
        "default": ["n_jet", "n_muon", "n_electron", "ht", "m_bb", "deltaR_bb", "jet1_pt"],  # n_deepjet, ....
        "test": ["n_jet", "n_electron", "jet1_pt"],
        "cutflow": ["cf_jet1_pt", "cf_jet4_pt", "cf_n_jet", "cf_n_electron", "cf_n_muon"],  # cf_n_deepjet
    }

    # shift groups for conveniently looping over certain shifts
    # (used during plotting)
    config_inst.x.shift_groups = {
        "jer": ["nominal", "jer_up", "jer_down"],
    }

    # selector step groups for conveniently looping over certain steps
    # (used in cutflow tasks)
    config_inst.x.selector_step_groups = {
        "resolved": ["Trigger", "Lepton", "VetoLepton", "Jet", "Bjet", "VetoTau"],
        "boosted": ["Trigger", "Lepton", "VetoLepton", "FatJet", "Boosted"],
        "default": ["Lepton", "VetoLepton", "Jet", "Bjet", "Trigger"],
        "thesis": ["Lepton", "Muon", "Jet", "Trigger", "Bjet"],  # reproduce master thesis cuts for checks
        "test": ["Lepton", "Jet", "Bjet"],
    }

    # plotting settings groups
    config_inst.x.general_settings_groups = {
        "test1": {"p1": True, "p2": 5, "p3": "text", "skip_legend": True},
        "default_norm": {"shape_norm": True, "yscale": "log"},
    }
    config_inst.x.process_settings_groups = {
        "default": {"ggHH_kl_1_kt_1_sl_hbbhww": {"scale": 2000, "unstack": True}},
        "unstack_all": {proc.name: {"unstack": True} for proc in config_inst.processes},
        "unstack_signal": {proc.name: {"unstack": True} for proc in config_inst.processes if "HH" in proc.name},
        "scale_signal": {
            proc.name: {"unstack": True, "scale": 10000}
            for proc in config_inst.processes if "HH" in proc.name
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
