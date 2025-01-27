# coding: utf-8

"""
Column production methods related trigger studies
"""


from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.categories import category_ids

from hbw.config.categories import add_categories_production
from hbw.weight.default import default_weight_producer
from hbw.production.weights import event_weights

np = maybe_import("numpy")
ak = maybe_import("awkward")


# produce trigger column for triggers included in the NanoAOD
@producer(
    produces={"trig_ids"},
    channel=["mm", "ee", "mixed"],
    version=10,
)
def trigger_prod(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces column filled for each event with the triggers triggering the event.
    This column can then be used to fill a Histogram where each bin corresponds to a certain trigger.
    """

    # TODO: check if trigger were fired by unprescaled L1 seed
    trig_ids = ak.Array([["allEvents"]] * len(events))

    for channel in self.channel:

        channel_trigger = ak.Array([0] * len(events))

        # label for stepwise combination of triggers
        comb_label = ""
        for trigger in self.config_inst.x.triggers:
            # build trigger combination for channel
            if channel in trigger.x.channels:
                comb_label += trigger.hlt_field + "+"
                channel_trigger = channel_trigger | events.HLT[trigger.hlt_field]

                # add stepwise trigger combination
                trig_passed = ak.where(channel_trigger, [[comb_label[:-1]]], [[]])
                trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

        # add trigger selection for the channel
        trig_passed = ak.where(channel_trigger, [[channel]], [[]])
        trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    # add individual triggers
    for trigger in self.config_inst.x.triggers:
        trig_passed = ak.where(events.HLT[trigger.hlt_field], [[trigger.hlt_field]], [[]])
        trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    events = set_ak_column(events, "trig_ids", trig_ids)

    return events


# initialize the trigger producer, triggers can be set in the trigger config
@trigger_prod.init
def trigger_prod_init(self: Producer) -> None:

    for trigger in self.config_inst.x.triggers:
        self.uses.add(f"HLT.{trigger.hlt_field}")


# producers for single channels
mu_trigger_prod = trigger_prod.derive("mu_trigger_prod", cls_dict={"channel": ["mu"]})
ele_trigger_prod = trigger_prod.derive("ele_trigger_prod", cls_dict={"channel": ["e"]})


# add categories used for trigger studies
@producer(
    uses=category_ids,
    produces=category_ids,
    version=3,
)
def trig_cats(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Reproduces the category ids to include the trigger categories
    """

    events = self[category_ids](events, **kwargs)

    return events


@trig_cats.init
def trig_cats_init(self: Producer) -> None:

    add_categories_production(self.config_inst)


# always plot trig_weights with '--weight-producer no_weights', otherwise the weights themselves are weighted
@producer(
    uses={
        default_weight_producer,
        event_weights,
    },
    produces={
        "trig_weights",
    },
    version=1,
)
def trig_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces a weight column to check the event weights
    """

    events = self[event_weights](events, **kwargs)
    events, weights = self[default_weight_producer](events, **kwargs)
    events = set_ak_column(events, "trig_weights", weights)

    return events
