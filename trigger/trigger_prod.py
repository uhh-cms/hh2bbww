# coding: utf-8

"""
Column production methods related trigger studies
"""


from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column


np = maybe_import("numpy")
ak = maybe_import("awkward")


########################################################################################################################
# Single lepton trigger producer
########################################################################################################################

# produce trigger columns for single lepton channel
@producer(
    produces={"trig_ids"},
    channel=["m", "e"],
    version=10,
)
def trigger_prod_sl(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces column filled for each event with the triggers triggering the event.
    This column can then be used to fill a Histogram where each bin corresponds to a certain trigger.
    This producer is only to explore the different possible triggers and their combinations.
    """

    # TODO: check if trigger were fired by unprescaled L1 seed
    trig_ids = ak.Array([["allEvents"]] * len(events))

    for channel in self.channel:
        # add individual triggers
        for trigger in self.config_inst.x.sl_triggers[f"{self.config_inst.x.year}"][channel]:
            if not ak.any(trig_ids == trigger):  # avoid double counting
                trig_passed = ak.where(events.HLT[trigger], [[trigger]], [[]])
                trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    combinations22 = {
        "ele30ele28": [
            "Ele30_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150"
        ],
        "ele30_quadjet": [
            "Ele30_WPTight_Gsf",
            "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65"
        ],
        "ele30e28e15": [
            "Ele30_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150",
            "Ele15_IsoVVVL_PFHT450"
        ],
        "ele30e28e15_quadjet": [
            "Ele30_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150",
            "Ele15_IsoVVVL_PFHT450",
            "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65"
        ],
        "ele30e15_quadjet": [
            "Ele30_WPTight_Gsf",
            "Ele15_IsoVVVL_PFHT450",
            "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65"
        ],
        "ele30e28e15e50_quadjet": [
            "Ele30_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150",
            "Ele15_IsoVVVL_PFHT450",
            "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
            "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65"
        ],
        "mu24m50": [
            "IsoMu24",
            "Mu50"
        ],
        "mu24m50_quadjet": [
            "IsoMu24",
            "Mu50",
            "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65"
        ],
        "mu24m50m15_quadjet": [
            "IsoMu24",
            "Mu50",
            "Mu15_IsoVVVL_PFHT450",
            "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65"
        ]
    }
    combinations23 = {
        "e30e28": [
            "Ele30_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150",
        ],
        "e30e28e15": [
            "Ele30_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150",
            "Ele15_IsoVVVL_PFHT450",
        ],
        "e30e28e15e50": [
            "Ele30_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150",
            "Ele15_IsoVVVL_PFHT450",
            "Ele50_CaloIdVT_GsfTrkIdT_PFJet165"
        ],
        "e30e28e15e50_quad340": [
            "Ele30_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150",
            "Ele15_IsoVVVL_PFHT450",
            "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70"
        ],
        "e30e28e15e50_quad280": [
            "Ele30_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150",
            "Ele15_IsoVVVL_PFHT450",
            "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55"
        ],
        "exEle30": [
            "Ele28_eta2p1_WPTight_Gsf_HT150",
            "Ele15_IsoVVVL_PFHT450",
            "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55"
        ],
        "exEle28": [
            "Ele30_WPTight_Gsf",
            "Ele15_IsoVVVL_PFHT450",
            "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55"
        ],
        "exEle15": [
            "Ele30_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150",
            "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55"
        ],
        "exEle50": [
            "Ele30_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150",
            "Ele15_IsoVVVL_PFHT450",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55"
        ],
        "exeQuad280": [
            "Ele30_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150",
            "Ele15_IsoVVVL_PFHT450",
            "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70",
        ],
        "exeQuad340": [
            "Ele30_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150",
            "Ele15_IsoVVVL_PFHT450",
            "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55",
        ],
        "m24m50": [
            "IsoMu24",
            "Mu50"
        ],
        "m24m50_quad340": [
            "IsoMu24",
            "Mu50",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70"
        ],
        "m24m50_quad280": [
            "IsoMu24",
            "Mu50",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55"
        ],
        "m24m50m15_quad280": [
            "IsoMu24",
            "Mu50",
            "Mu15_IsoVVVL_PFHT450",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55"
        ],
        "m24m50m15_quad340": [
            "IsoMu24",
            "Mu50",
            "Mu15_IsoVVVL_PFHT450",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70"
        ],
        "m24m50m15_quad280_340": [
            "IsoMu24",
            "Mu50",
            "Mu15_IsoVVVL_PFHT450",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70"
        ],
        "m24m50m15_quad280_340_Mu3MET": [
            "IsoMu24",
            "Mu50",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55",
            "Mu15_IsoVVVL_PFHT450",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70",
            "Mu3er1p5_PFJet100er2p5_PFMETNoMu100_PFMHTNoMu100_IDTight"
        ],
        "exIsoMu24": [
            "Mu50",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55",
            "Mu15_IsoVVVL_PFHT450",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70"
        ],
        "exMu50": [
            "IsoMu24",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55",
            "Mu15_IsoVVVL_PFHT450",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70"
        ],
        "exMu15": [
            "IsoMu24",
            "Mu50",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70"
        ],
        "exQuad280": [
            "IsoMu24",
            "Mu50",
            "Mu15_IsoVVVL_PFHT450",
            "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70",
        ],
        "exQuad340": [
            "IsoMu24",
            "Mu50",
            "Mu15_IsoVVVL_PFHT450",
            "PFHT280_QuadPFJet30_PNet2BTagMean0p55",
        ],
    }
    combinations = {
        2022: combinations22,
        2023: combinations23
    }

    for key in combinations[self.config_inst.x.year].keys():
        trig_passed = ak.Array([0] * len(events))
        for trigger in combinations[self.config_inst.x.year][key]:
            trig_passed = trig_passed | (events.HLT[trigger])
        trig_passed = ak.where(trig_passed, [[key]], [[]])
        trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    events = set_ak_column(events, "trig_ids", trig_ids)

    return events


# initialize the trigger producer, triggers can be set in the trigger config
@trigger_prod_sl.init
def trigger_prod_sl_init(self: Producer) -> None:

    for channel in self.config_inst.x.sl_triggers[f"{self.config_inst.x.year}"]:
        for trigger in self.config_inst.x.sl_triggers[f"{self.config_inst.x.year}"][channel]:
            self.uses.add(f"HLT.{trigger}")


########################################################################################################################
# Dilepton trigger producer
########################################################################################################################

# dilepton trigger prod
@producer(
    produces={"trig_ids"},
    version=1,
)
def trigger_prod_dl(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Uses the trigger_ids produced during the selection to build additional combinations of triggers.
    Produces column filled for each event with the triggers triggering the event.
    This column can then be used to fill a Histogram where each bin corresponds to a certain trigger.
    """
    # build sequential combinations of triggers, ids can be found in the trigger config
    mixed_trigger_sequence = {
        "emu_dilep": [301, 401],
        "emu_single": [201, 101],
        "emu_electronjet": [203],
    }
    ee_trigger_sequence = {
        "ee_dilep": [202, 204],
        "ee_single": [201],
        "ee_electronjet": [203],
    }
    mm_trigger_sequence = {
        "mm_dilep": [102],
        "mm_single": [101],
    }
    trigger_sequence = {
        "ee": ee_trigger_sequence,
        "mm": mm_trigger_sequence,
        "mixed": mixed_trigger_sequence
    }

    # initialize the trigger ids column filled with the labels
    trig_ids = ak.Array([["allEvents"]] * len(events))
    # add individual triggers
    for trigger in self.config_inst.x.triggers:
        trig_passed = ak.any(events.trigger_ids == trigger.id, axis=1)
        trig_ids = ak.concatenate([trig_ids, ak.where(trig_passed, [[trigger.hlt_field]], [[]])], axis=1)

    # add sequential combinations of triggers
    for channel in ["ee", "mm", "mixed"]:
        seq_trigger = events.run * 0  # initialize with zeros
        seq_label = ""
        for label, trigger_ids in trigger_sequence[channel].items():
            trigger_mask = events.run * 0
            for trigger_id in trigger_ids:
                trigger_mask = trigger_mask | ak.any(events.trigger_ids == trigger_id, axis=1)
                seq_trigger = seq_trigger | ak.any(events.trigger_ids == trigger_id, axis=1)

            trig_passed = ak.where(trigger_mask, [[label]], [[]])
            trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

            seq_label += label + "+"
            if "+" in seq_label[:-1]:
                trig_passed = ak.where(seq_trigger, [[seq_label[:-1]]], [[]])
                trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

        # add the complete channel selection
        trig_passed = ak.where(seq_trigger, [[f"{channel}"]], [[]])
        trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    events = set_ak_column(events, "trig_ids", trig_ids)

    return events


# initialize the dilepton trigger producer, triggers can be set in the trigger config
@trigger_prod_dl.init
def trigger_prod_dl_init(self: Producer) -> None:
    self.uses.add("trigger_ids")


# dilepton trigger prod without sequential combinations to reduce memory usage
@producer(
    produces={"trig_ids"},
    version=1,
)
def trigger_prod_dls(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Uses the trigger_ids produced during the selection to build additional combinations of triggers.
    Produces column filled for each event with the triggers triggering the event.
    This column can then be used to fill a Histogram where each bin corresponds to a certain trigger.
    """
    # build sequential combinations of triggers, ids can be found in the trigger config
    mixed_trigger_sequence = {
        "emu_dilep": [301, 401],
        "emu_single": [201, 101],
        "emu_electronjet": [203],
    }
    ee_trigger_sequence = {
        "ee_dilep": [202, 204],
        "ee_single": [201],
        "ee_electronjet": [203],
    }
    mm_trigger_sequence = {
        "mm_dilep": [102],
        "mm_single": [101],
    }
    trigger_sequence = {
        "ee": ee_trigger_sequence,
        "mm": mm_trigger_sequence,
        "mixed": mixed_trigger_sequence
    }

    # initialize the trigger ids column filled with the labels
    trig_ids = ak.Array([["allEvents"]] * len(events))
    # add individual triggers
    for trigger in self.config_inst.x.triggers:
        trig_passed = ak.any(events.trigger_ids == trigger.id, axis=1)
        trig_ids = ak.concatenate([trig_ids, ak.where(trig_passed, [[trigger.hlt_field]], [[]])], axis=1)

    # add sequential combinations of triggers
    for channel in ["ee", "mm", "mixed"]:
        seq_trigger = events.run * 0  # initialize with zeros
        seq_label = ""
        for label, trigger_ids in trigger_sequence[channel].items():
            trigger_mask = events.run * 0
            for trigger_id in trigger_ids:
                trigger_mask = trigger_mask | ak.any(events.trigger_ids == trigger_id, axis=1)
                seq_trigger = seq_trigger | ak.any(events.trigger_ids == trigger_id, axis=1)

            # trig_passed = ak.where(trigger_mask, [[label]], [[]])
            # trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

            seq_label += label + "+"
            # if "+" in seq_label[:-1]:
            #     # trig_passed = ak.where(seq_trigger, [[seq_label[:-1]]], [[]])
            #     # trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

        # add the complete channel selection
        trig_passed = ak.where(seq_trigger, [[f"{channel}"]], [[]])
        trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    events = set_ak_column(events, "trig_ids", trig_ids)

    return events


# initialize the dilepton trigger producer, triggers can be set in the trigger config
@trigger_prod_dls.init
def trigger_prod_dls_init(self: Producer) -> None:
    self.uses.add("trigger_ids")


########################################################################################################################
# Simple trigger producer for debugging
########################################################################################################################

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

    trig_ids = ak.Array([["allEvents"]] * len(events))

    # add individual triggers
    for trigger in self.config_inst.x.triggers:
        trig_passed = ak.where(events.HLT[trigger.hlt_field], [[trigger.hlt_field]], [[]])
        trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    events = set_ak_column(events, "trig_ids", trig_ids)

    return events


# initialize the trigger producer, triggers can be set in the trigger config
@trigger_prod_db.init
def trigger_prod_db_init(self: Producer) -> None:

    for trigger in self.config_inst.x("triggers", []):
        self.uses.add(f"HLT.{trigger.hlt_field}")
