# coding: utf-8

import law

from columnflow.inference import InferenceModel
from columnflow.tasks.framework.base import RESOLVE_DEFAULT
from hbw.util import bracket_expansion


def default_calibrator(container):
    # return ["with_b_reg", "fatjet"]
    return ["ak4", "fatjet", "ele"]


def default_selector(container):
    if container.has_tag("is_sl"):
        selector = "sl1"
    elif container.has_tag("is_dl"):
        selector = "dl1"

    return selector


def ml_inputs_producer(container):
    if container.has_tag("is_sl") and not container.has_tag("is_resonant"):
        ml_inputs = "sl_ml_inputs"
    if container.has_tag("is_dl"):
        ml_inputs = "dl_ml_inputs"
    if container.has_tag("is_sl") and container.has_tag("is_resonant"):
        ml_inputs = "sl_res_ml_inputs"
    return ml_inputs


def default_ml_model(cls, container, task_params):
    """ Function that chooses the default_ml_model based on the inference_model if given """
    # for most tasks, do not use any default ml model
    default_ml_model = ()

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

        # get the default_ml_model from the inference_model_cls
        inference_model_cls = InferenceModel.get_cls(inference_model)
        default_ml_model = getattr(inference_model_cls, "ml_model_name", default_ml_model)

    return default_ml_model


def default_producers(cls, container, task_params):
    """ Default producers chosen based on the Inference model and the ML Model """

    # per default, use the ml_inputs and event_weights
    default_producers = ["event_weights", "pre_ml_cats", ml_inputs_producer(container)]

    if hasattr(cls, "ml_model"):
        # do no further resolve the ML categorizer when this task is part of the MLTraining pipeline
        default_producers.remove("pre_ml_cats")
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
    # use the ml categorization producer instead of the default categorization producer
    if ml_model not in (None, law.NO_STR, RESOLVE_DEFAULT, tuple()):
        default_producers.remove("pre_ml_cats")
        # NOTE: this producer needs to be added as the last element! otherwise, category_ids will be overwritten
        default_producers.append(f"cats_ml_{ml_model}")

    return default_producers


def set_sl_config_defaults_and_groups(config_inst):
    """ Configuration function that sets all the defaults and groups in the config_inst """
    year = config_inst.campaign.x.year

    # define the default dataset and process based on the analysis tags
    signal_tag = "qqlnu" if config_inst.has_tag("is_sl") else "2l2nu"
    default_signal_process = "hh_ggf_hbb_hvv_kl1_kt1"
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
    config_inst.x.default_calibrator = default_calibrator(config_inst)
    config_inst.x.default_selector = default_selector(config_inst)
    config_inst.x.default_reducer = "default"
    config_inst.x.ml_inputs_producer = ml_inputs_producer(config_inst)
    config_inst.x.default_producer = default_producers
    config_inst.x.default_hist_producer = "default"
    # config_inst.x.default_hist_producer = "with_trigger_weight"
    config_inst.x.default_ml_model = default_ml_model
    config_inst.x.default_inference_model = "default" if year == 2017 else "sl_22"
    config_inst.x.default_categories = ["incl", "sr", "dycr", "ttcr"]
    config_inst.x.default_variables = ["jet0_pt", "mll", "n_jet", "ptll", "lepton0_pt", "lepton1_pt"]

    # general_settings default needs to be tuple (or dict) to be resolved correctly
    config_inst.x.default_general_settings = ("data_mc_plots",)
    config_inst.x.default_custom_style_config = "default"

    #
    # Groups
    #

    backgrounds0 = ["h", "ttv", "vv", "w_lnu", "st", "dy_m4to10", "dy_m10to50", "dy_m50toinf", "tt"]
    backgrounds1 = ["h", "ttv", "vv", "w_lnu", "st", "dy", "tt"]
    hbbhww_sm = ["hh_ggf_hbb_hww_kl1_kt1", "hh_vbf_hbb_hww_kv1_k2v1_kl1"]
    hh_sm = [
        "hh_ggf_hbb_hww_kl1_kt1", "hh_vbf_hbb_hww_kv1_k2v1_kl1", "hh_ggf_hbb_hzz_kl1_kt1", "hh_ggf_hbb_htt_kl1_kt1",
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
        "dl3": [default_signal_process, "h", "ttv", "vv", "w_lnu", "st", "dy_m10to50", "dy_m50toinf", "tt"],  # noqa: E501
        "dlmu": ["data_mu", default_signal_process, "h", "ttv", "vv", "w_lnu", "st", "dy_m4to10", "dy_m10to50", "dy_m50toinf", "tt"],  # noqa: E501
        "dleg": ["data_egamma", default_signal_process, "h", "ttv", "vv", "w_lnu", "st", "dy_m4to10", "dy_m10to50", "dy_m50toinf", "tt"],  # noqa: E501
        "dlmajor": [default_signal_process, "st", "dy", "tt"],
        "2much": [default_signal_process, "h", "ttv", "vv", "w_lnu", "st", "dy_m4to10", "dy_m10to50", "dy_m50toinf", "tt"],  # noqa: E501
        "2ech": [default_signal_process, "h", "ttv", "vv", "w_lnu", "st", "dy_m4to10", "dy_m10to50", "dy_m50toinf", "tt"],  # noqa: E501
        "emuch": [default_signal_process, "h", "ttv", "vv", "w_lnu", "st", "dy_m4to10", "dy_m10to50", "dy_m50toinf", "tt"],  # noqa: E501
        "inference": ["hh_ggf_*", "tt", "st", "w_lnu", "dy", "qcd_*"],
        "postfit": [*hbbhww_sm, *backgrounds1],
        "k2v": ["hh_vbf_*", "tt", "st", "w_lnu", "dy", "qcd_*"],
        "ml": [default_signal_process, "tt", "st", "w_lnu", "dy"],
        "ml_test": [default_signal_process, "st", "w_lnu"],
        "mldl": ["hh_ggf_hbb_hvv2l2nu_kl1_kt1", "tt", "st", "dy"],
        "mlsl": ["hh_ggf_hbb_hvvqqlnu_kl1_kt1", "tt", "st", "w_lnu", "dy"],
        "test": [default_signal_process, "tt_sl"],
        "small": [default_signal_process, "tt", "st"],
        "bkg": ["tt", "st", "w_lnu", "dy"],
        # signal groups
        "signal": ["hh_ggf_*", "hh_vbf_*"],
        "hbv": ["hh_ggf_hbb_hvv_kl1_kt1", "hh_vbf_hbb_hvv_kv1_k2v1_kl1"],
        "hbw": ["hh_ggf_hbb_hww_kl1_kt1", "hh_vbf_hbb_hww_kv1_k2v1_kl1"],
        "hbz": ["hh_ggf_hbb_hzz_kl1_kt1", "hh_vbf_hbb_hzz_kv1_k2v1_kl1"],
        "hbv_ggf": ["hh_ggf_hbb_hvv_kl*_kt1"], "hbv_vbf": ["hh_vbf_hbb_hvv_*"],
        "hbv_ggf_dl": ["hh_ggf_hbb_hvv2l2nu_kl*_kt1"],
        "hbv_ggf_sl": ["hh_ggf_hbb_hvvqqlnu_kl*_kt1"],
        "hbv_vbf_dl": ["hh_vbf_hbb_hvv2l2nu_*"],
        "hbv_vbf_sl": ["hh_vbf_hbb_hvvqqlnu_*"],
        "hbw_ggf": ["hh_ggf_hbb_hww_kl*_kt1"], "hbw_vbf": ["hh_vbf_hbb_hww_*"],
        "hbw_ggf_dl": ["hh_ggf_hbb_hww2l2nu_kl*_kt1"],
        "hbw_ggf_sl": ["hh_ggf_hbb_hwwqqlnu_kl*_kt1"],
        "hbw_vbf_dl": ["hh_vbf_hbb_hww2l2nu_*"],
        "hbw_vbf_sl": ["hh_vbf_hbb_hwwqqlnu_*"],
        "hbz_ggf": ["hh_ggf_hbb_hzz_kl*_kt1"], "hbz_vbf": ["hh_vbf_hbb_hzz_*"],
        "hbz_ggf_dl": ["hh_ggf_hbb_hzz2l2nu_kl*_kt1"],
        "hbz_ggf_sl": ["hh_ggf_hbb_hzzqqlnu_kl*_kt1"],
        "hbz_vbf_dl": ["hh_vbf_hbb_hzz2l2nu_*"],
        "hbz_vbf_sl": ["hh_vbf_hbb_hzzqqlnu_*"],
        # background groups (separated for plotting)
        "dy_m": ["dy_m4to10", "dy_m10to50", "dy_m50toinf"],
        # background groups (for yield tables)
        "table": [*hbbhww_sm, *backgrounds0, "data", "background"],
        "table0": [*hh_sm, *backgrounds0, "data", "background"],
        "table1": [*hbbhww_sm, *backgrounds1, "data", "background"],
        "dy_all": ["dy", "dy_m4to10", "dy_m10to50", "dy_m50toinf", "dy_m50toinf_0j", "dy_m50toinf_1j", "dy_m50toinf_2j"],  # noqa: E501
        "tt_all": ["tt", "tt_dl", "tt_sl", "tt_fh"],
        "st_all": ["st", "st_schannel", "st_tchannel", "st_twchannel"],
        "h_all": ["h", "h_ggf", "h_vbf", "vh", "zh", "zh_gg", "wh", "tth", "ttvh"],
    }
    for proc, datasets in config_inst.x.dataset_names.items():
        remove_generator = lambda x: x.replace("_powheg", "").replace("_madgraph", "").replace("_amcatnlo", "").replace("_pythia8", "").replace("4f_", "")  # noqa: E501
        config_inst.x.process_groups[f"datasets_{proc}"] = [remove_generator(dataset) for dataset in datasets]

    for group in ("dl3", "dl2", "dl1", "dl", "2much", "2ech", "emuch"):
        # thanks to double counting removal, we can (and should) now use all datasets in each channel
        config_inst.x.process_groups[f"d{group}"] = ["data"] + config_inst.x.process_groups[group]

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    config_inst.x.dataset_groups = {
        "all": ["*"],
        "test": [default_signal_dataset, "tt_sl_powheg"],
        "small": [default_signal_dataset, "tt_*", "st_*"],
        "bkg": ["tt_*", "st_*", "w_lnu_*", "dy_*"],
        "tt": ["tt_*"], "st": ["st_*"], "w": ["w_lnu_*"], "dy": ["dy_*"],
        "qcd": ["qcd_*"], "qcd_mu": ["qcd_mu*"], "qcd_ele": ["qcd_em*", "qcd_bctoe*"],
        "signal": ["hh_ggf_*", "hh_vbf_*"], "hh_ggf": ["hh_ggf_*"], "hh_vbf": ["hh_vbf_*"],
        "ml": ["hh_ggf*kl1_kt1", "tt_*", "st_*", "dy_*", "w_lnu_*"],
        "dilep": ["tt_*", "st_*", "dy_*", "w_lnu_*", "hh_ggf_*"],
        "h": ["h_ggf_*", "h_vbf_*", "zh_*", "wph_*", "wmh_*", "tth_*", "ttzh_*", "ttwh_*"],
    }
    if config_inst.name == "l22post":
        config_inst.x.dataset_groups["test123"] = ["tt_dl_powheg", "tt_sl_powheg"]
    elif config_inst.name == "l22pre":
        config_inst.x.dataset_groups["test123"] = ["tt_dl_powheg"]

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
        "default": ["incl", "sr__1e", "sr__1mu"],
        "test": ["incl", "sr__1e"],
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
        "BR_dl": bracket_expansion(["sr__{1b,2b}__ml_{tt,st,dy,h}"]),
        "SR_sig_sl": bracket_expansion(["sr__{1e,1mu}__{1b,2b}__ml_{sig_ggf,sig_vbf}"]),
        "SR_bkg_sl": bracket_expansion(["sr__{1e,1mu}__{1b,2b}__ml_{tt,st,w_lnu}"]),
    }

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    config_inst.x.variable_groups = {
        "gen_vbf": ["vbfpair.deta", "vbfpair.mass", "gen_sec1_eta", "gen_sec2_eta", "gen_sec1_pt", "gen_sec2_pt"],
        "mli": ["mli_*"],
        "iso": bracket_expansion(["lepton{0,1}_{pfreliso,minipfreliso,mvatth}"]),
        "sl": ["n_*", "electron_*", "muon_*", "met_*", "jet*", "bjet*", "ht"],
        "sl_resolved": ["n_*", "electron_*", "muon_*", "met_*", "jet*", "bjet*", "ht"],
        "sl_boosted": ["n_*", "electron_*", "muon_*", "met_*", "fatjet_*"],
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
        "dilep": ["Jet", "Bjet", "Lepton", "Trigger"],
    }

    # plotting settings groups
    # (used in plotting)
    config_inst.x.general_settings_groups = {
        "test1": {"p1": True, "p2": 5, "p3": "text", "skip_legend": True},
        "default_norm": {"shape_norm": True, "yscale": "log"},
        "postfit": {
            "whitespace_fraction": 0.4,
            "cms_label": "simpw",
            "yscale": "log",
            "hide_signal_errors": True,
        },
        "data_mc_plots": {
            # "custom_style_config": "default",  # NOTE: does not work in combination with group
            "whitespace_fraction": 0.4,
            "cms_label": "pw",
            "yscale": "log",
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
        "test": {
            "mli_mbb": {"rebin": 2, "label": "test"},
            "mli_mjj": {"rebin": 2},
        },
    }

    # groups for custom plot styling
    config_inst.x.custom_style_config_groups = {
        "default": {
            "legend_cfg": {
                "ncols": 2,
                "fontsize": 16,
                "bbox_to_anchor": (0., 0., 1., 1.),
            },
            "annotate_cfg": {
                "xy": (0.05, 0.95),
                "xycoords": "axes fraction",
                "fontsize": 16,
            },
        },
        "legend_single_col": {
            "legend_cfg": {"ncols": 1, "fontsize": 20},
        },
        "small_legend": {
            "legend_cfg": {"ncols": 2, "fontsize": 16},
        },
        "no_cat_label": {
            "legend_cfg": {"ncols": 2, "fontsize": 20},
            "annotate_cfg": {"text": ""},
        },
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
        # Single lepton
        "SR_sl": 10,
        "vbfSR_sl": 5,
        "BR_sl": 3,
        "SR_sl_resolved": 10,
        "SR_sl_boosted": 5,
        "vbfSR_sl_resolved": 5,
        "vbfSR_sl_boosted": 3,
        "SR_sig_sl": 10,
        "SR_bkg_sl": 3,
    }

    is_signal_sm = lambda proc_name: "kl1_kt1" in proc_name or "kv1_k2v1_kl1" in proc_name
    is_signal_sm_ggf = lambda proc_name: "kl1_kt1" in proc_name
    is_signal_sm_vbf = lambda proc_name: "kv1_k2v1_kl1" in proc_name
    # is_gghh_sm = lambda proc_name: "kl1_kt1" in proc_name
    # is_qqhh_sm = lambda proc_name: "kv1_k2v1_kl1" in proc_name
    # is_signal_ggf_kl1 = lambda proc_name: "kl1_kt1" in proc_name and "hh_ggf" in proc_name
    # is_signal_vbf_kl1 = lambda proc_name: "kv1_k2v1_kl1" in proc_name and "hh_vbf" in proc_name
    is_background = lambda proc_name: (
        "hbb_hvv" not in proc_name and "hbb_hww" not in proc_name and "hbb_hzz" not in proc_name
    )

    config_inst.x.inference_category_rebin_processes = {
        # Single lepton
        "SR_sl": is_signal_sm_ggf,
        "vbfSR_sl": is_signal_sm_vbf,
        "SR_sl_resolved": is_signal_sm,
        "SR_sl_boosted": is_signal_sm,
        "vbfSR_sl_resolved": is_signal_sm,
        "vbfSR_sl_boosted": is_signal_sm,
        "BR_sl": is_background,
        "SR_sig_sl": is_signal_sm,
        "SR_bkg_sl": is_background,
    }
