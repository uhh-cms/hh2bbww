# coding: utf-8

"""
Selection modules for HH(bbWW) to identify events that should be considered missing.
"""

from __future__ import annotations

import law
from columnflow.util import maybe_import
from columnflow.columnar_util import fill_at, full_like, has_ak_column

from columnflow.selection import Selector, SelectionResult, selector

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@selector
def get_outlier_scale_weights(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    **kwargs,
) -> ak.Array:
    """
    Helper to identify bad events that should be considered missing altogether
    """
    bad_mask = full_like(events.event, False, dtype=bool)
    results.steps["no_sel_mask"] = ~bad_mask
    if self.dataset_inst.is_data:
        # no bad data events
        return events, results
    if self.dataset_inst.has_tag("no_lhe_weights"):
        logger.debug("Dataset has tag 'no_lhe_weights', skipping checks for bad events")
        # at the moment, we only check for bad events from LHE weights
        return events, results

    if not has_ak_column(events, "LHEScaleWeight"):
        logger.debug("Dataset does not have LHEScaleWeight column, skipping checks for bad events")
        # when ScaleWeights are not present, we cannot check for bad events
        return events, results

    # drop events for which we expect lhe infos but that lack them
    # see https://cms-talk.web.cern.ch/t/lhe-weight-vector-empty-for-certain-events/97636/3
    if self.dataset_inst.has_tag("partial_lhe_weights"):
        logger.debug("Dataset has tag 'partial_lhe_weights', dropping events with empty LHEScaleWeight")
        n_weights = ak.num(events.LHEScaleWeight, axis=1)
        bad_lhe_mask = (n_weights != 8) & (n_weights != 9)
        if ak.any(bad_lhe_mask):
            bad_mask = bad_mask | bad_lhe_mask
            frac = ak.mean(bad_lhe_mask)
            logger.warning(
                f"found {ak.sum(bad_lhe_mask)} events ({frac * 100:.1f}%) with bad LHEScaleWeight",
            )
    if self.dataset_inst.has_tag(["has_lhe_weights", "partial_lhe_weights"], mode=any):
        logger.debug("Dataset has tag 'has_lhe_weights' or 'partial_lhe_weights', checking for bad LHEScaleWeights")
        # check if the LHE weights are all finite (assuming bad LHEScaleWeights also means bad LHEPdfWeights)
        bad_lhe_mask = ak.any(~np.isfinite(events.LHEScaleWeight), axis=1)
        if ak.any(bad_lhe_mask):
            bad_mask = bad_mask | bad_lhe_mask
            frac = ak.mean(bad_lhe_mask)
            logger.warning(
                f"found {ak.sum(bad_lhe_mask)} events ({frac * 100:.1f}%) with non-finite LHEScaleWeights; "
                "setting all LHE scale and pdf weights to 1",
            )

            # set LHEScaleWeight and LHEPdfWeight to 1 for these events
            for col in ("LHEScaleWeight", "LHEPdfWeight"):
                if has_ak_column(events, col):
                    # set weights to 1 for these events to avoid issues with nan values downstream
                    events = fill_at(events, bad_lhe_mask, col, ak.ones_like(events[col]))
                    # events = fill_at(events, bad_lhe_mask, col, full_like(events[col], 1.0))

    # define "no selection" step as all events that are not considered bad from this function
    no_sel = ~bad_mask
    results.steps["no_sel_mask"] = no_sel

    return events, results


@selector
def extend_bad_events(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    **kwargs,
) -> ak.Array:
    """
    Helper to identify bad events that should be considered missing altogether
    """
    if self.dataset_inst.is_data or self.dataset_inst.has_tag("no_lhe_weights"):
        return events, results

    if has_ak_column(events, "pdf_weight"):
        bad_mask = (events.pdf_weight == 0)

    results.steps["no_sel_mask"] = results.steps.no_sel_mask & (~bad_mask)
    return events, results
