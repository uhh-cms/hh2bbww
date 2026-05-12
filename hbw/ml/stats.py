# coding: utf-8

""" production methods regarding ml stats """

from __future__ import annotations

import functools

import law

from columnflow.production import Producer, producer
from columnflow.categorization import Categorizer
from columnflow.util import maybe_import
from columnflow.ml import MLModel
from columnflow.columnar_util import set_ak_column
from columnflow.selection.stats import increment_stats
from hbw.categorization.categories import catid_sr, catid_mll_low_narrow, catid_hhh_sr, catid_geq3b, catid_geq4b, catid_eq2b, catid_eq3b  # noqa
from hbw.util import IF_SL, IF_DL, IF_MC
from hbw.weight.hp_dih import default_hist_producer
from hbw.weight.hp_trih import default_hist_producer_trih


ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

logger = law.logger.get_logger(__name__)


def del_sub_proc_stats(
    stats: dict,
    proc: str,
) -> np.ndarray:
    """
    Function deletes dict keys which are not part of the requested process

    :param stats: Dictionaire containing ML stats for each process.
    :param proc: String of the process.
    :param sub_id: List of ids of sub processes that should be reatined (!).
    """
    item_list = list(stats.weight_map.keys())
    for item in item_list:
        stats[item].pop()


@producer(
    uses={IF_SL(catid_sr), IF_DL(catid_mll_low_narrow), IF_DL(catid_hhh_sr), increment_stats, "process_id", "fold_indices"},  # noqa E501
    produces={IF_MC("event_weight")},
    extra_categorizer=None,
)
def prepml(
    self: Producer,
    events: ak.Array,
    task: law.Task,
    stats: dict = {},
    fold_indices: ak.Array | None = None,
    ml_model_inst: MLModel | None = None,
    **kwargs,
) -> ak.Array:
    """
    Producer that is run as part of PrepareMLEvents to collect relevant stats
    """

    if self.config_inst.has_tag("is_sl"):
        sr_categorizer = catid_sr
    elif self.config_inst.has_tag("is_dl"):
        if self.config_inst.has_tag("is_hh"):
            sr_categorizer = catid_mll_low_narrow
            default_hp = default_hist_producer
        elif self.config_inst.has_tag("is_hhh"):
            sr_categorizer = catid_hhh_sr
            default_hp = default_hist_producer_trih
        else:
            raise Exception(f"config {self.config_inst.name} needs either the 'is_hh' or 'is_hhh' tag")
    else:
        raise Exception(f"config {self.config_inst.name} needs either the 'is_sl' or 'is_dl' tag")

    if task.task_family == "cf.PrepareMLEvents":
        # pass category mask to only use events that belong to the main "signal region"
        # NOTE: we could also just require the pre_ml_cats Producer here

        events, mask = self[sr_categorizer](events, **kwargs)
        logger.info(f"Select {ak.sum(mask)} from {len(events)} events for MLTraining using {sr_categorizer.cls_name}")
        events = events[mask]

        if self.extra_categorizer:
            for cat_cls in self.categorizers_cls:
                # apply additional categorizer if specified
                events, mask = self[cat_cls](events, **kwargs)
                logger.info(f"Select {ak.sum(mask)} from {len(events)} events using {cat_cls.cls_name}")
                events = events[mask]

    weight_map = {
        "num_events": Ellipsis,  # all events
    }

    if task.dataset_inst.is_mc:
        # full event weight
        events, weight = self[default_hp](events, task, **kwargs)
        events = set_ak_column_f32(events, "event_weight", weight)
        stats["sum_weights"] += float(ak.sum(weight, axis=0))
        weight_map["sum_weights"] = weight
        weight_map["sum_pos_weights"] = (weight, weight > 0)
        weight_map["sum_abs_weights"] = np.abs(weight)
        weight_map["num_events_pos_weights"] = weight > 0

        # normalization weight only
        norm_weight = events["stitched_normalization_weight"]
        stats["sum_norm_weights"] += float(ak.sum(norm_weight, axis=0))
        weight_map["sum_norm_weights"] = norm_weight
        weight_map["sum_pos_norm_weights"] = (norm_weight, norm_weight > 0)
        weight_map["sum_abs_norm_weights"] = np.abs(norm_weight)

    group_map = {
        "process": {
            "values": events.process_id,
            "mask_fn": (lambda v: events.process_id == v),
        },
        "fold": {
            "values": events.fold_indices,
            "mask_fn": (lambda v: events.fold_indices == v),
            "combinations_only": True,
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

    key_list = list(weight_map.keys())
    for key in key_list:
        stats.pop(key, None)
        # TODO: pop 'num_fold_events'

    return events


@prepml.init
def prepml_init(self):
    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return

    if self.config_inst.has_tag("is_dl"):
        if self.config_inst.has_tag("is_hh"):
            default_hp = default_hist_producer
        elif self.config_inst.has_tag("is_hhh"):
            default_hp = default_hist_producer_trih
        else:
            raise Exception(f"config {self.config_inst.name} needs either the 'is_hh' or 'is_hhh' tag")
    else:
        raise Exception(f"config {self.config_inst.name} needs 'is_dl' tag")

    self.uses.add("stitched_normalization_weight")
    self.uses.add(default_hp)

    if self.extra_categorizer:
        self.categorizers_cls = []
        for cls_name in law.util.make_list(self.extra_categorizer):
            if not Categorizer.has_cls(cls_name):
                logger.warning(
                    f"Extra categorizer {cls_name} not found, skipping it in {self.cls_name}.",
                )
                continue
            cat_cls = Categorizer.get_cls(cls_name)
            self.categorizers_cls.append(cat_cls)
            self.uses.add(cat_cls)


prepml_eq2b = prepml.derive("prepml_eq2b", cls_dict={"extra_categorizer": "catid_eq2b"})
prepml_eq3b = prepml.derive("prepml_eq3b", cls_dict={"extra_categorizer": "catid_eq3b"})
prepml_geq4b = prepml.derive("prepml_geq4b", cls_dict={"extra_categorizer": "catid_geq4b"})
prepml_geq3b = prepml.derive("prepml_geq3b", cls_dict={"extra_categorizer": "catid_geq3b"})
prepml_met40 = prepml.derive("prepml_met40", cls_dict={"extra_categorizer": "mask_fn_met_geq40"})
prepml_fatjet = prepml.derive("prepml_fatjet", cls_dict={"extra_categorizer": "catid_fatjet"})
prepml_2j = prepml.derive("prepml_2j", cls_dict={"extra_categorizer": "catid_2njet"})
# prepml_hhh_sr = prepml.derive("prepml_hhh_sr", cls_dict={"extra_categorizer": "mask_fn_hhh_sr"})
prepml_sr = prepml.derive("prepml_hhh_sr", cls_dict={"extra_categorizer": "mask_fn_hhh_sr"})
