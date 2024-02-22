# coding: utf-8

"""
Selection modules for HH(bbWW) that are used for both SL and DL.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Tuple

import law
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, get_ak_routes
from columnflow.production.util import attach_coffea_behavior

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.met_filters import met_filters
from columnflow.selection.cms.json_filter import json_filter
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.categories import category_ids
from columnflow.production.processes import process_ids
from columnflow.production.cms.seeds import deterministic_seeds

from hbw.selection.gen import hard_gen_particles
from hbw.production.weights import event_weights_to_normalize, large_weights_killer
from hbw.selection.stats import hbw_selection_step_stats, hbw_increment_stats

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
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
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
        hbw_met_filters, json_filter, "PV.npvsGood",
        process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,
    },
    produces={
        hbw_met_filters, json_filter,
        process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,
    },
    exposed=False,
)
def pre_selection(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """ Methods that are called for both SL and DL before calling the selection modules """

    # temporary fix for optional types from Calibration (e.g. events.Jet.pt --> ?float32)
    # TODO: remove as soon as possible as it might lead to weird bugs when there are none entries in inputs
    events = ak.fill_none(events, EMPTY_FLOAT)

    # run deterministic seeds when no Calibrator has been requested
    if not self.task.calibrators:
        events = self[deterministic_seeds](events, **kwargs)

    # mc weight
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
        events = self[large_weights_killer](events, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # apply some general quality criteria on events
    results.steps["good_vertex"] = events.PV.npvsGood >= 1
    events, met_results = self[hbw_met_filters](events, **kwargs)  # produces "met_filter" step
    results += met_results
    if self.dataset_inst.is_data:
        events, json_results = self[json_filter](events, **kwargs)  # produces "json" step
        results += json_results
    else:
        results.steps["json"] = ak.Array(np.ones(len(events), dtype=bool))

    # combine quality criteria into a single step
    results.steps["cleanup"] = (
        results.steps.good_vertex & results.steps.met_filter & results.steps.json
    )

    return events, results


@pre_selection.init
def pre_selection_init(self: Selector) -> None:
    if self.task and not self.task.calibrators:
        self.uses.add(deterministic_seeds)
        self.produces.add(deterministic_seeds)


@selector(
    uses={
        category_ids, hbw_increment_stats, hbw_selection_step_stats,
    },
    produces={
        category_ids, hbw_increment_stats, hbw_selection_step_stats,
    },
    exposed=False,
)
def post_selection(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """ Methods that are called for both SL and DL after calling the selection modules """

    if self.dataset_inst.is_mc:
        events, results = self[hard_gen_particles](events, results, **kwargs)

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # produce event weights
    if self.dataset_inst.is_mc:
        events = self[event_weights_to_normalize](events, results=results, **kwargs)

    # increment stats
    self[hbw_selection_step_stats](events, results, stats, **kwargs)
    self[hbw_increment_stats](events, results, stats, **kwargs)

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

    self.uses.update({event_weights_to_normalize, hard_gen_particles})
    self.produces.update({event_weights_to_normalize, hard_gen_particles})
