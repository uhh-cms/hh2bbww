# coding: utf-8

"""
Column production methods related trigger studies
"""


from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.categories import category_ids

from hbw.config.categories import add_categories_production
from hbw.weight.default import default_hist_producer
from hbw.production.weights import event_weights

np = maybe_import("numpy")
ak = maybe_import("awkward")


def check_l1_seeds(self: Producer, events: ak.Array, trigger) -> ak.Array:
    """
    Check if the unprescaled L1 seeds of a given trigger have fired
    """
    l1_seeds_fired = ak.Array([False] * len(events))

    for l1_seed in self.config_inst.x.hlt_L1_seeds[trigger]:
        l1_seeds_fired = l1_seeds_fired | events.L1[l1_seed]

    return l1_seeds_fired


# produce trigger column for triggers included in the NanoAOD
@producer(
    produces={"trig_ids"},
    channel=["mm", "ee", "mixed"],
    version=16,
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
        channel_trigger2 = ak.Array([0] * len(events))  # w/o check for unprescaled L1 seeds

        # label for stepwise combination of triggers
        comb_label = ""
        for trigger in self.config_inst.x.triggers:
            # build trigger combination for channel
            if channel in trigger.x.channels:
                comb_label += trigger.hlt_field + "+"
                channel_trigger = channel_trigger | (events.HLT[trigger.hlt_field]
                                                     & check_l1_seeds(self, events, trigger.hlt_field))
                channel_trigger2 = channel_trigger2 | (events.HLT[trigger.hlt_field])

                # add stepwise trigger combination
                if "+" in comb_label[:-1]:
                    trig_passed = ak.where(channel_trigger, [[comb_label[:-1]]], [[]])
                    trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

        # add trigger selection for the channel
        trig_passed = ak.where(channel_trigger, [[channel]], [[]])
        trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

        trig_passed = ak.where(channel_trigger2, [[f"{channel}_prescL1"]], [[]])
        trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    # add individual triggers
    for trigger in self.config_inst.x.triggers:
        trig_passed = ak.where(events.HLT[trigger.hlt_field] & check_l1_seeds(self, events, trigger.hlt_field),
                               [[trigger.hlt_field]], [[]])
        trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    # add sequential combination of triggers for dpg talk
    mixed_trigger_sequence = {
        "emu_dilep": ["Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL", "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ"],
        "emu_single": ["Ele30_WPTight_Gsf", "IsoMu24"],
        "emu_electronjet": ["Ele50_CaloIdVT_GsfTrkIdT_PFJet165"],
        # "emu_alt_mix": ["Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
        #                 "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        #                 "Ele30_WPTight_Gsf",
        #                 "IsoMu24",
        #                 "Ele115_CaloIdVT_GsfTrkIdT"]
    }
    ee_trigger_sequence = {
        "ee_dilep": ["Ele23_Ele12_CaloIdL_TrackIdL_IsoVL", "DoubleEle33_CaloIdL_MW"],
        "ee_single": ["Ele30_WPTight_Gsf"],
        "ee_electronjet": ["Ele50_CaloIdVT_GsfTrkIdT_PFJet165"],
    }
    mm_trigger_sequence = {
        "mm_dilep": ["Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8"],
        "mm_single": ["IsoMu24"],
    }
    trigger_sequence = {
        "ee": ee_trigger_sequence,
        "mm": mm_trigger_sequence,
        "mixed": mixed_trigger_sequence
    }

    for channel in self.channel:
        seq_trigger = ak.Array([0] * len(events))
        seq_label = ""
        for label, triggers in trigger_sequence[channel].items():
            triggers_mask = ak.Array([0] * len(events))
            for trigger in triggers:
                triggers_mask = triggers_mask | (events.HLT[trigger] & check_l1_seeds(self, events, trigger))
                seq_trigger = seq_trigger | (events.HLT[trigger] & check_l1_seeds(self, events, trigger))

            trig_passed = ak.where(triggers_mask, [[label]], [[]])
            trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

            seq_label += label + "+"
            if "+" in seq_label[:-1]:
                trig_passed = ak.where(seq_trigger, [[seq_label[:-1]]], [[]])
                trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    events = set_ak_column(events, "trig_ids", trig_ids)

    return events


# initialize the trigger producer, triggers can be set in the trigger config
@trigger_prod.init
def trigger_prod_init(self: Producer) -> None:

    for trigger in self.config_inst.x("triggers", []):
        self.uses.add(f"HLT.{trigger.hlt_field}")
    # self.uses.add("HLT.Ele115_CaloIdVT_GsfTrkIdT")

    for hlt_path, l1_seeds in self.config_inst.x.hlt_L1_seeds.items():
        for l1_seed in l1_seeds:
            if f"L1.{l1_seeds}" not in self.uses:
                self.uses.add(f"L1.{l1_seed}")


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
        default_hist_producer,
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
    events, weights = self[default_hist_producer](events, **kwargs)
    events = set_ak_column(events, "trig_weights", weights)

    return events


# produce trigger columns for debugging
@producer(
    produces={"trig_ids"},
    channel=["mm", "ee", "mixed"],
    version=1,
)
def trigger_prod_db(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces column filled for each event with the triggers triggering the event.
    This column can then be used to fill a Histogram where each bin corresponds to a certain trigger.
    """

    # TODO: check if trigger were fired by unprescaled L1 seed
    trig_ids = ak.Array([["allEvents"]] * len(events))

    # add individual triggers
    for trigger in self.config_inst.x.triggers:
        trig_passed = ak.where(events.HLT[trigger.hlt_field], [[trigger.hlt_field]], [[]])
        trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    trig_passed = ak.where(events.HLT["Ele115_CaloIdVT_GsfTrkIdT"], [["Ele115_CaloIdVT_GsfTrkIdT"]], [[]])
    trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    events = set_ak_column(events, "trig_ids", trig_ids)

    return events


# initialize the trigger producer, triggers can be set in the trigger config
@trigger_prod_db.init
def trigger_prod_db_init(self: Producer) -> None:

    for trigger in self.config_inst.x("triggers", []):
        self.uses.add(f"HLT.{trigger.hlt_field}")
    self.uses.add("HLT.Ele115_CaloIdVT_GsfTrkIdT")
