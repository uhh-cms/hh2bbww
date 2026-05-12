# coding: utf-8

import law

from columnflow.inference import InferenceModel
from columnflow.tasks.framework.base import RESOLVE_DEFAULT


def default_calibrator(container):
    default_calibrators = law.config.get_expanded("analysis", "default_calibrators", ())
    if isinstance(default_calibrators, str):
        return default_calibrators.split(",")
    else:
        return ["ak4", "ak8", "ele"]


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
        if container.has_tag("is_hh"):
            ml_inputs = "dl_ml_inputs"
        elif container.has_tag("is_hhh"):
            ml_inputs = "hhh_dl_ml_inputs"
    if container.has_tag("is_sl") and container.has_tag("is_resonant"):
        ml_inputs = "sl_res_ml_inputs"
    return ml_inputs


def default_hist_producer(container):
    if container.has_tag("is_dl"):
        if container.has_tag("is_hh"):
            hist_producer = "with_trigger_weight"
        elif container.has_tag("is_hhh"):
            hist_producer = "hhh_default"
    return hist_producer


def default_ml_model(cls, container, task_params):
    """ Function that chooses the default_ml_model based on the inference_model if given """
    # for most tasks, do not use any default ml model
    default_ml_model = law.config.get_expanded("analysis", "default_ml_models", ())
    if isinstance(default_ml_model, str):
        default_ml_model = default_ml_model.split(",")

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

    # try and get the default ml model if not set
    if ml_model in (None, law.NO_STR, RESOLVE_DEFAULT):
        ml_model = default_ml_model(cls, container, task_params)

    # only consider 1 ml_model
    if ml_model and isinstance(ml_model, (list, tuple)):
        ml_model = ml_model[0]

    # if a ML model is set, and the task is not part of the MLTraining pipeline,
    # use the ml categorization producer instead of the default categorization producer
    if ml_model not in (None, law.NO_STR, RESOLVE_DEFAULT, tuple()):
        default_producers.remove("pre_ml_cats")
        # NOTE: this producer needs to be added as the last element! otherwise, category_ids will be overwritten
        default_producers.append(f"cats_ml_{ml_model}")

    return default_producers


def set_dl_config_defaults_and_groups(config_inst):
    """ Configuration function that sets all the defaults and groups in the config_inst """
    # define the default dataset and process based on the analysis tags

    #
    # Defaults
    #

    config_inst.x.default_calibrator = default_calibrator(config_inst)
    config_inst.x.default_selector = default_selector(config_inst)
    config_inst.x.default_reducer = "default"
    config_inst.x.ml_inputs_producer = ml_inputs_producer(config_inst)
    config_inst.x.default_producer = default_producers
    config_inst.x.default_hist_producer = default_hist_producer(config_inst)
    config_inst.x.default_ml_model = default_ml_model
    config_inst.x.default_inference_model = "default_unblind"
    config_inst.x.default_categories = ["incl", "sr", "dycr", "ttcr"]
    config_inst.x.default_variables = ["jet0_pt", "mll", "n_jet", "ptll", "lepton0_pt", "lepton1_pt"]
    config_inst.x.default_general_settings = ("data_mc_plots_not_blinded",)
    config_inst.x.default_custom_style_config = "default"

    #
    # Groups
    #

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    config_inst.x.dataset_groups = {
        "all": ["*"],
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

    # shift groups for conveniently looping over certain shifts
    # (used during plotting)
    config_inst.x.shift_groups = {
        "jer": ["nominal", "jer_up", "jer_down"],
        # TODO this is just a workaround to call cf.PlotShiftedVariables with a group or shift-sources
        "all_up": [
            # # theory unc.
            "pdf_up",
            "murf_envelope_up",
            "isr_up",
            "fsr_up",
            "top_pt_up",
            "dy_correction_up",
            # # experimental unc.
            # "lumi_13p6TeV_2022_up",
            # b-tagging
            "btag_hf_up",
            "btag_lf_up",
            "btag_hfstats1_up",
            "btag_hfstats2_up",
            "btag_lfstats1_up",
            "btag_lfstats2_up",
            "btag_cferr1_up",
            "btag_cferr2_up",
            # other experimental unc.
            "mu_id_sf_up",
            "mu_iso_sf_up",
            "e_sf_up",
            "e_reco_sf_up",
            "trigger_sf_up",
            "minbias_xs_up",
            # jerc
            "jer_up",
            "jec_Total_up",
        ],
        "theory_up": [
            "pdf_up",
            "murf_envelope_up",
            "isr_up",
            "fsr_up",
            "top_pt_up",
        ],
        "btag_up": [
            "btag_hf_up",
            "btag_lf_up",
            "btag_hfstats1_up",
            "btag_hfstats2_up",
            "btag_lfstats1_up",
            "btag_lfstats2_up",
            "btag_cferr1_up",
            "btag_cferr2_up",
        ],
        "experimental_up": [
            "mu_id_sf_up",
            "mu_iso_sf_up",
            "e_sf_up",
            "e_reco_sf_up",
            "trigger_sf_up",
            "minbias_xs_up",
            "dy_correction_up",
        ],
        "jerc_up": [
            "jer_up",
            "jec_Total_up",
        ],
    }
    config_inst.x.shift_groups["shapes_up"] = [
        *config_inst.x.shift_groups["theory_up"],
        *config_inst.x.shift_groups["btag_up"],
        *config_inst.x.shift_groups["experimental_up"],
    ]
    for shift_groups in ("all", "theory", "btag", "experimental", "jerc"):
        config_inst.x.shift_groups[shift_groups + "_down"] = [
            shift.replace("_up", "_down") for shift in config_inst.x.shift_groups[shift_groups + "_up"]
        ]
        config_inst.x.shift_groups[shift_groups] = (
            config_inst.x.shift_groups[shift_groups + "_up"] +
            config_inst.x.shift_groups[shift_groups + "_down"]
        )

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

    def reorder_mll(ax, handles, labels, n_cols):
        empty_handle = ax.plot([], label="", linestyle="None")[0]

        hh_idxs = []
        data_idx = None
        new_handles = []
        new_labels = []
        for i, label in enumerate(labels):
            if "HH" in label:
                hh_idxs.append(i)
            elif "data" in label.lower():
                data_idx = i
            else:
                new_handles.append(handles[i])
                new_labels.append(labels[i])

        new_handles.insert(0, handles[data_idx])
        new_labels.insert(0, labels[data_idx])
        for idx in hh_idxs[::-1]:
            new_handles.insert(0, handles[idx])
            new_labels.insert(0, labels[idx])
        for i in range(3):
            new_handles.insert(0, empty_handle)
            new_labels.insert(0, "")

        handles[:] = new_handles
        labels[:] = new_labels

    def reorder_data_first(ax, handles, labels, n_cols):
        """Reorder legend entries to put 'data' first"""
        # Find the index of the data entry
        data_idx = None
        empty_label_idx = None
        for i, label in enumerate(labels):
            # if empty_label_idx is None and label == "":
            if label == "":
                empty_label_idx = i
            if data_idx is None and "data" in label.lower():
                data_idx = i

        if data_idx is not None and data_idx != 0:
            # Move data to first position
            data_handle = handles.pop(data_idx)
            data_label = labels.pop(data_idx)

            handles.pop(empty_label_idx)  # remove empty entry added via cf_entries_per_column
            labels.pop(empty_label_idx)

            # insert data at the top of column 1 (index 0)
            handles.insert(0, data_handle)
            labels.insert(0, data_label)

    def reorder_data_pos(ax, handles, labels, n_cols, label_pos=0):
        """Reorder legend entries to put 'data' first"""
        # Find the index of the data entry
        data_idx = None
        for i, label in enumerate(labels):
            if "data" in label.lower():
                data_idx = i
                break

        if data_idx is not None and data_idx != 0:
            # Move data to first position
            data_handle = handles.pop(data_idx)
            data_label = labels.pop(data_idx)
            handles.pop(0)  # remove empty entry added via cf_entries_per_column
            labels.pop(0)
            # # insert data at the bottom of column 1 (index 2)
            handles.insert(label_pos, data_handle)
            labels.insert(label_pos, data_label)

            # handles.insert(2, data_handle)
            # labels.insert(2, data_label)

    # groups for custom plot styling
    config_inst.x.custom_style_config_groups = {
        "dpostfit_merged": {
            "gridspec_cfg": {
                "left": 0.08,
                "right": 0.98,
                "top": 0.95,
                # "bottom": 0.05,
                "bottom": 0.1,
            },
            "subplots_cfg": {
                "figsize": (15, 10),
            },
            "legend_cfg": {
                "ncols": 5,
                "cf_entries_per_column": [0, 3, 3, 3, 4],  # start with empty col, then move data to front using cf_update_handles_labels  # noqa: E501
                "cf_update_handles_labels": reorder_data_pos,
                "fontsize": 20,
                "bbox_to_anchor": (0., 0., 1., 1.),
            },
            "annotate_cfg": {
                "xy": (0.03, 0.95),
                "xycoords": "axes fraction",
                "fontsize": 24,
            },
            "ax_cfg": {
                # "ylabel_fontsize": 30,
                # "xlabel_fontsize": 30,
                # "ylim": (2e-1, 6e7),
                "ylim": (2e-1, 1e8),
            },
            "rax_cfg": {
                # "ylabel_fontsize": 30,
                # "xlabel_fontsize": 30,
                "ylim": (0.30, 1.70),
                "ylabel": "Data / Bkg.",
                "xlabel": "Bin number",
            },
            "cms_label_cfg": {
                "fontsize": 24,
            },
        },
        "postfit_merged": {
            "gridspec_cfg": {
                "left": 0.08,
                "right": 0.98,
                "top": 0.95,
                "bottom": 0.05,
            },
            "subplots_cfg": {
                "figsize": (15, 10),
            },
            "legend_cfg": {
                "ncols": 4,
                "fontsize": 20,
                "bbox_to_anchor": (0., 0., 1., 1.),
            },
            "annotate_cfg": {
                "xy": (0.03, 0.95),
                "xycoords": "axes fraction",
                "fontsize": 24,
            },
            "rax_cfg": {
                "ylim": (0.30, 1.70),
                "ylabel": "Data / Bkg.",
            },
            "cms_label_cfg": {
                "fontsize": 24,
            },
        },
        "shifts": {
            "gridspec_cfg": {
                "left": 0.08,
                "right": 0.98,
                "top": 0.95,
                "bottom": 0.05,
            },
            "legend_cfg": {
                "ncols": 1,
                # "fontsize": ,
                "bbox_to_anchor": (0., 0., 1., 1.),
            },
            "rax_cfg": {
                "ylim": (0.95, 1.05),
            },
        },
        "dpostfit": {
            "legend_cfg": {
                "ncols": 2,
                "fontsize": 20,
                "bbox_to_anchor": (0., 0., 1., 1.),
                "cf_entries_per_column": [5, 8],
                "cf_update_handles_labels": reorder_data_first,
            },
            "annotate_cfg": {
                "xy": (0.03, 0.95),
                "xycoords": "axes fraction",
                "fontsize": 22,
            },
            "ax_cfg": {
                "xlabel_fontsize": 30,
                "ylabel_fontsize": 30,
            },
            "rax_cfg": {
                "ylabel_fontsize": 30,
                "xlabel_fontsize": 30,
                "ylabel": "Data / Bkg.",
            },
            "cms_label_cfg": {
                "fontsize": 24,
            },
        },
        "dpostfit_mll": {
            "legend_cfg": {
                "ncols": 3,
                "fontsize": 18,
                "bbox_to_anchor": (0., 0., 1., 1.),
                "cf_update_handles_labels": reorder_mll,
            },
            "annotate_cfg": {
                "xy": (0.03, 0.95),
                "xycoords": "axes fraction",
                "fontsize": 22,
            },
            "ax_cfg": {
                "xlabel_fontsize": 30,
                "ylabel_fontsize": 30,
            },
            "rax_cfg": {
                "ylabel_fontsize": 30,
                "xlabel_fontsize": 30,
                "ylabel": "Data / Bkg.",
            },
            "cms_label_cfg": {
                "fontsize": 24,
            },
        },
        "dpostfit_nosig": {
            "legend_cfg": {
                "ncols": 2,
                "fontsize": 20,
                "bbox_to_anchor": (0., 0., 1., 1.),
                "cf_entries_per_column": [4, 7],
                "cf_update_handles_labels": reorder_data_first,
            },
            "annotate_cfg": {
                "xy": (0.03, 0.95),
                "xycoords": "axes fraction",
                "fontsize": 22,
            },
            "ax_cfg": {
                "xlabel_fontsize": 30,
                "ylabel_fontsize": 30,
            },
            "rax_cfg": {
                "ylabel_fontsize": 30,
                "xlabel_fontsize": 30,
                "ylabel": "Data / Bkg.",
            },
            "cms_label_cfg": {
                "fontsize": 24,
            },
        },
        "dpostfit0": {
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
            "rax_cfg": {
                "ylabel": "Data / Bkg.",
            },
        },
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
        "default_rax40": {
            "legend_cfg": {
                "ncols": 2,
                "fontsize": 16,
                "bbox_to_anchor": (0., 0., 1., 1.),
            },
            "rax_cfg": {
                "ylim": (0.60, 1.40),
            },
            "annotate_cfg": {
                "xy": (0.05, 0.95),
                "xycoords": "axes fraction",
                "fontsize": 16,
            },
        },
        "default_rax60": {
            "legend_cfg": {
                "ncols": 2,
                "fontsize": 16,
                "bbox_to_anchor": (0., 0., 1., 1.),
            },
            "rax_cfg": {
                "ylim": (0.40, 1.60),
            },
            "annotate_cfg": {
                "xy": (0.05, 0.95),
                "xycoords": "axes fraction",
                "fontsize": 16,
            },
        },
        "default_rax75": {
            "legend_cfg": {
                "ncols": 2,
                "fontsize": 16,
                "bbox_to_anchor": (0., 0., 1., 1.),
            },
            # "ax_cfg": {
            #     "ylim": (-10, 10),
            # },
            "rax_cfg": {
                "ylim": (0.25, 1.75),
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
