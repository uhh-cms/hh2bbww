# coding: utf-8

""" production methods regarding ml stats """

from __future__ import annotations

import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.ml import MLModel
from columnflow.columnar_util import set_ak_column
from columnflow.selection.stats import increment_stats


ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={increment_stats, "normalization_weight", "process_id"},
    # produces={"dummy"},
)
def ml_preparation(
    self: Producer,
    events: ak.Array,
    stats: dict = {},
    fold_indices: ak.Array | None = None,
    ml_model_inst: MLModel | None = None,
    **kwargs,
) -> ak.Array:
    """
    Producer that is run as part of PrepareMLEvents to collect relevant stats
    """

    weight = events["normalization_weight"]
    stats["num_events"] += len(events)
    stats["sum_weights"] += ak.sum(weight, axis=0)

    weight_map = {
        "num_events": Ellipsis,  # all events
    }
    if self.task.dataset_inst.is_mc:
        weight_map["sum_weights"] = weight
        weight_map["sum_abs_weights"] = (weight, weight > 0)
        weight_map["sum_pos_weights"] = np.abs(weight)

    group_map = {
        "process": {
            "values": events.process_id,
            "mask_fn": (lambda v: events.process_id == v),
        },
        "fold": {
            "values": fold_indices,
            "mask_fn": (lambda v: fold_indices == v),
        },
    }

    group_combinations = [("process", "fold")]

    self[increment_stats](
        events,
        None,  # SelectionResult that is not required
        stats,
        weight_map=weight_map,
        group_map=group_map,
        group_combinations=group_combinations,
        **kwargs,
    )

    return events
