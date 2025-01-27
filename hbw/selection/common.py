# coding: utf-8

"""
Selection modules for HH(bbWW) that are used for both SL and DL.
"""

from __future__ import annotations

from collections import defaultdict
from columnflow.types import Callable

import law
from columnflow.util import maybe_import, DotDict
from columnflow.columnar_util import EMPTY_FLOAT, get_ak_routes
from columnflow.production.util import attach_coffea_behavior

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.met_filters import met_filters
from columnflow.selection.cms.json_filter import json_filter
from columnflow.selection.cms.jets import jet_veto_map
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.categories import category_ids
from hbw.production.process_ids import hbw_process_ids
from columnflow.production.cms.seeds import deterministic_seeds

from hbw.selection.gen import hard_gen_particles
from hbw.production.weights import event_weights_to_normalize, large_weights_killer
from hbw.selection.stats import hbw_selection_step_stats, hbw_increment_stats
from hbw.selection.hists import hbw_selection_hists


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


@selector(
    uses={"*"},
    exposed=True,
)
def check_columns(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    # hists: dict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    routes = get_ak_routes(events)  # noqa
    from hbw.util import debugger
    debugger()
    raise Exception("This Selector is only for checking nano content")


def get_met_filters(self: Selector):
    """ custom function to skip met filter for our Run2 EOY signal samples """
    met_filters = self.config_inst.x.met_filters

    if getattr(self, "dataset_inst", None) and self.dataset_inst.has_tag("is_eoy"):
        # remove filter for EOY sample
        try:
            met_filters.remove("Flag.BadPFMuonDzFilter")
        except (KeyError, AttributeError):
            pass

    return met_filters


hbw_met_filters = met_filters.derive("hbw_met_filters", cls_dict=dict(get_met_filters=get_met_filters))


@selector(
    uses={
        jet_veto_map,
        hbw_met_filters, json_filter, "PV.npvsGood",
        hbw_process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,
    },
    produces={
        hbw_met_filters, json_filter,
        hbw_process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,
    },
    exposed=False,
)
def pre_selection(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """ Methods that are called for both SL and DL before calling the selection modules """

    # temporary fix for optional types from Calibration (e.g. events.Jet.pt --> ?float32)
    # TODO: remove as soon as possible as it might lead to weird bugs when there are none entries in inputs
    events = ak.fill_none(events, EMPTY_FLOAT)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # run deterministic seeds when no Calibrator has been requested
    if not self.task.calibrators:
        events = self[deterministic_seeds](events, **kwargs)

    # mc weight
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
        events = self[large_weights_killer](events, **kwargs)

    if self.dataset_inst.is_mc:
        # get hard gen particles
        events, results = self[hard_gen_particles](events, results, **kwargs)

    # create process ids
    events = self[hbw_process_ids](events, **kwargs)

    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # apply some general quality criteria on events
    results.steps["good_vertex"] = events.PV.npvsGood >= 1
    events, met_results = self[hbw_met_filters](events, **kwargs)  # produces "met_filter" step
    results += met_results
    if self.dataset_inst.is_data:
        events, json_results = self[json_filter](events, **kwargs)  # produces "json" step
        results += json_results
    else:
        results.steps["json"] = ak.Array(np.ones(len(events), dtype=bool))

    # apply jet veto map
    events, jet_veto_results = self[jet_veto_map](events, **kwargs)
    results += jet_veto_results

    # combine quality criteria into a single step
    results.steps["cleanup"] = (
        results.steps.jet_veto_map &
        results.steps.good_vertex &
        results.steps.met_filter &
        results.steps.json
    )

    return events, results


@pre_selection.init
def pre_selection_init(self: Selector) -> None:
    if self.task and not self.task.calibrators:
        self.uses.add(deterministic_seeds)
        self.produces.add(deterministic_seeds)

    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return

    self.uses.update({hard_gen_particles})
    self.produces.update({hard_gen_particles})


@selector(
    uses={
        category_ids, hbw_increment_stats, hbw_selection_step_stats,
        hbw_selection_hists,
    },
    produces={
        category_ids, hbw_increment_stats, hbw_selection_step_stats,
        hbw_selection_hists,
    },
    exposed=False,
)
def post_selection(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: defaultdict,
    hists: dict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """ Methods that are called for both SL and DL after calling the selection modules """

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # produce event weights
    if self.dataset_inst.is_mc:
        events = self[event_weights_to_normalize](events, results=results, **kwargs)

    # increment stats
    events = self[hbw_selection_step_stats](events, results, stats, **kwargs)
    events = self[hbw_increment_stats](events, results, stats, **kwargs)
    events = self[hbw_selection_hists](events, results, hists, **kwargs)

    def log_fraction(stats_key: str, msg: str | None = None):
        if not stats.get(stats_key):
            return
        if not msg:
            msg = "Fraction of {stats_key}"
        logger.info(f"{msg}: {(100 * stats[stats_key] / stats['num_events']):.2f}%")

    log_fraction("num_negative_weights", "Fraction of negative weights")
    log_fraction("num_pu_0", "Fraction of events with pu_weight == 0")
    log_fraction("num_pu_100", "Fraction of events with pu_weight >= 100")

    # temporary fix for optional types from Calibration (e.g. events.Jet.pt --> ?float32)
    # TODO: remove as soon as possible as it might lead to weird bugs when there are none entries in inputs
    events = ak.fill_none(events, EMPTY_FLOAT)

    logger.info(f"Selected {ak.sum(results.event)} from {len(events)} events")
    return events, results


@post_selection.init
def post_selection_init(self: Selector) -> None:

    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return

    self.uses.update({event_weights_to_normalize})
    self.produces.update({event_weights_to_normalize})


configurable_attributes = {
    # lepton selection
    "mu_pt": float,
    "ele_pt": float,
    "mu2_pt": float,
    "ele2_pt": float,
    # trigger selection
    "trigger": dict,  # dict[str, list[str]]
    "trigger_config_func": Callable,
    # jet selection
    "jet_pt": float,
    "n_jet": int,
    # bjet selection
    "n_btag": int,
}


def configure_selector(self: Selector):
    """
    Helper to configure the selector with the configurable attributes.

    This function should be called in the main Selector's __init__ method.
    The configuration is stored in the main config_inst under the key "selector_config".
    The configurable attributes and their expected types are defined in the `configurable_attributes` dict.

    The configuration is performed as follows:
    - The confugation is performed once per config_inst as soon as a config_inst is available. This is ensured by
        checking if the config_inst has the aux key "selector_config".
    - If the attribute is a Callable, it is called. This function can directly set config attributes or
        return the value to be set. When None is returned, the attribute is skipped.
    - If the attribute is None, it is skipped.
    - If the attribute is not of the expected type, a TypeError is raised.
    - If the attribute is already set in the config_inst, a warning is issued.
    - The configuration is stored in the config_inst under the key "selector_config".
    - The configuration is also stored in the config_inst under the key of the attribute name (to be removed).
    - The btag_sf, btag_column, and btag_wp are set based on the b_tagger attribute.
    """
    # ensure calling just once
    if not hasattr(self, "config_inst") or self.config_inst.has_aux("selector_config"):
        return

    self.config_inst.set_aux("selector_config", DotDict())

    for attr_name, attr_type in configurable_attributes.items():
        if not hasattr(self, attr_name):
            continue

        attr = getattr(self, attr_name)

        if isinstance(attr, Callable):
            logger.info(f"Calling config function '{attr_name}'")
            attr = attr()

        if attr is None:
            continue

        if not isinstance(attr, attr_type):
            raise TypeError(f"Attribute '{attr_name}' must be of type '{attr_type}' or None")

        if attr_name in self.config_inst.aux and attr != self.config_inst.get_aux(attr_name):
            logger.info(
                f"Selector {self.cls_name} is overwriting config attribute '{attr_name}' "
                f"(replaces '{self.config_inst.get_aux(attr_name)}' with '{attr}')",
            )

        # TODO: only use the "selector_config" and remove the direct config aux
        self.config_inst.set_aux(attr_name, attr)
        self.config_inst.x.selector_config[attr_name] = attr
