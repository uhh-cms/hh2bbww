# coding: utf-8

"""
Stat-related methods.
"""

from collections import defaultdict, OrderedDict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.production.cms.btag import btag_weights
from hbw.production.weights import event_weights_to_normalize
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={increment_stats, btag_weights, event_weights_to_normalize},
)
def hbw_increment_stats(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: dict,
    **kwargs,
) -> ak.Array:
    # collect important information from the results
    event_mask = results.main.event
    event_mask_no_bjet = results.steps.all_but_bjet
    n_jets = results.x.n_central_jets

    # weight map definition
    weight_map = {
        # "num" operations
        "num_events": Ellipsis,  # all events
        "num_events_selected": event_mask,  # selected events only
        "num_events_selected_no_bjet": event_mask_no_bjet,
        # "sum" operations
        "sum_mc_weight": events.mc_weight,  # weights of all events
        "sum_mc_weight_selected": (events.mc_weight, event_mask),  # weights of selected events
        "sum_mc_weight_no_bjet": (events.mc_weight, event_mask_no_bjet),
        "sum_mc_weight_selected_no_bjet": (events.mc_weight, event_mask_no_bjet),
    }

    weight_columns = list(
        set(self[event_weights_to_normalize].produced_columns) |
        set(self[btag_weights].produced_columns),
    )
    weight_columns = sorted([col.string_nano_column for col in weight_columns])

    # mc weight times correction weight (with variations) without any selection
    for name in weight_columns:
        if "weight" not in name:
            # skip non-weight columns here
            continue

        weight_map[f"sum_mc_weight_{name}"] = (events.mc_weight * events[name], Ellipsis)

        # weights for selected events
        weight_map[f"sum_mc_weight_{name}_selected"] = (events.mc_weight * events[name], event_mask)

        if name.startswith("btag_weight"):
            # weights for selected events, excluding the bjet selection
            weight_map[f"sum_mc_weight_{name}_selected_no_bjet"] = (
                (events.mc_weight * events[name], event_mask_no_bjet)
            )

    group_map = {
        "process": {
            "values": events.process_id,
            "mask_fn": (lambda v: events.process_id == v),
        },
        "njet": {
            "values": results.x.n_central_jets,
            "mask_fn": (lambda v: n_jets == v),
        },
    }

    group_combinations = [("process", "njet")]

    self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        group_combinations=group_combinations,
        **kwargs,
    )

    return events


@selector(
    uses={btag_weights, event_weights_to_normalize},
)
def increment_stats_old(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: dict,
    **kwargs,
) -> ak.Array:
    """
    Unexposed selector that does not actually select objects but instead increments selection
    *stats* in-place based on all input *events* and the final selection *mask*.
    """
    # get event masks
    event_mask = results.main.event
    event_mask_no_bjet = results.steps.all_but_bjet

    # increment plain counts
    stats["num_events"] += len(events)
    stats["num_events_selected"] += ak.sum(event_mask, axis=0)

    # get a list of unique jet multiplicities present in the chunk
    unique_process_ids = np.unique(events.process_id)
    unique_n_jets = []
    if results.has_aux("n_central_jets"):
        unique_n_jets = np.unique(results.x.n_central_jets)

    # create a map of entry names to (weight, mask) pairs that will be written to stats
    weight_map = OrderedDict()
    if self.dataset_inst.is_mc:
        # mc weight for all events
        weight_map["mc_weight"] = (events.mc_weight, Ellipsis)

        # mc weight for selected events
        weight_map["mc_weight_selected"] = (events.mc_weight, event_mask)

        # mc weight for selected events, excluding the bjet selection
        weight_map["mc_weight_selected_no_bjet"] = (events.mc_weight, event_mask_no_bjet)

        # mc weight times correction weight (with variations) without any selection
        for name in sorted(self[event_weights_to_normalize].produced_columns):
            if "weight" not in name or "btag" in name:
                # skip btag and non-weight columns here
                continue
            weight_map[f"mc_weight_{name}"] = (events.mc_weight * events[name], Ellipsis)

        # btag weights
        for name in sorted(self[btag_weights].produces):
            if not name.startswith("btag_weight"):
                continue

            # weights for all events
            weight_map[name] = (events[name], Ellipsis)

            # weights for selected events
            weight_map[f"{name}_selected"] = (events[name], event_mask)

            # weights for selected events, excluding the bjet selection
            weight_map[f"{name}_selected_no_bjet"] = (events[name], event_mask_no_bjet)

            # mc weight times btag weight for selected events, excluding the bjet selection
            weight_map[f"mc_weight_{name}_selected_no_bjet"] = (events.mc_weight * events[name], event_mask_no_bjet)

    # get and store the weights
    for name, (weights, mask) in weight_map.items():
        joinable_mask = True if mask is Ellipsis else mask

        # sum for all processes
        stats[f"sum_{name}"] += ak.sum(weights[mask])

        # sums per process id and again per jet multiplicity
        stats.setdefault(f"sum_{name}_per_process", defaultdict(float))
        stats.setdefault(f"sum_{name}_per_process_and_njet", defaultdict(lambda: defaultdict(float)))
        for p in unique_process_ids:
            stats[f"sum_{name}_per_process"][int(p)] += ak.sum(
                weights[(events.process_id == p) & joinable_mask],
            )
            for n in unique_n_jets:
                stats[f"sum_{name}_per_process_and_njet"][int(p)][int(n)] += ak.sum(
                    weights[
                        (events.process_id == p) &
                        (results.x.n_central_jets == n) &
                        joinable_mask
                    ],
                )

    return events
