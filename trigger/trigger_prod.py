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

    year = self.config_inst.campaign.x.year

    # TODO: check if trigger were fired by unprescaled L1 seed
    trig_ids = ak.Array([["allEvents"]] * len(events))

    for channel in self.channel:
        # add individual triggers
        for trigger in self.config_inst.x.sl_triggers[f"{year}"][channel]:
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

    for key in combinations[year].keys():
        trig_passed = ak.Array([0] * len(events))
        for trigger in combinations[year][key]:
            trig_passed = trig_passed | (events.HLT[trigger])
        trig_passed = ak.where(trig_passed, [[key]], [[]])
        trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    events = set_ak_column(events, "trig_ids", trig_ids)

    return events


# initialize the trigger producer, triggers can be set in the trigger config
@trigger_prod_sl.init
def trigger_prod_sl_init(self: Producer) -> None:
    year = self.config_inst.campaign.x.year
    for channel in self.config_inst.x.sl_triggers[f"{year}"]:
        for trigger in self.config_inst.x.sl_triggers[f"{year}"][channel]:
            self.uses.add(f"HLT.{trigger}")


# sl trigger prod using the single lepton trigger ids
@producer(
    produces={"trig_ids"},
    channel=["m", "e"],
    version=1,
)
def trigger_prod_sl_ids(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Uses the trigger_ids produced during the selection to build additional combinations of triggers.
    """
    combinations = {
        "e30_e28_e15_quadjet": [201, 202, 205, 501],
        "e30_e28_e15": [201, 202, 205],
        "e30_e15_quadjet": [201, 205, 501],
        "e30_e28_quadjet": [201, 202, 501],
        "e15_e28_quadjet": [202, 205, 501],
        "e30_quadjet": [201, 501],
        "e30_e15": [201, 205],
        "e30_e28": [201, 202],
        "m24_m50_m15_quadjet": [101, 103, 104, 501],
        "m24_m50_m15": [101, 103, 104],
        "m24_m50_quadjet": [101, 103, 501],
        "m24_m15_quadjet": [101, 104, 501],
        "m15_m50_quadjet": [103, 104, 501],
        "m24_quadjet": [101, 501],
        "m24_m50": [101, 103],
        "m24_m15": [101, 104],
    }

    # initialize the trigger ids column filled with the labels
    trig_ids = ak.Array([["allEvents"]] * len(events))
    # add individual triggers
    for trigger in self.config_inst.x.triggers:
        trig_passed = ak.any(events.trigger_ids == trigger.id, axis=1)
        trig_ids = ak.concatenate([trig_ids, ak.where(trig_passed, [[trigger.hlt_field]], [[]])], axis=1)

    # add combinations of triggers
    for label, trigger_ids in combinations.items():
        trigger_mask = events.run * 0
        for trigger_id in trigger_ids:
            trigger_mask = trigger_mask | ak.any(events.trigger_ids == trigger_id, axis=1)
        trig_passed = ak.where(trigger_mask, [[label]], [[]])
        trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    events = set_ak_column(events, "trig_ids", trig_ids)

    return events


@trigger_prod_sl_ids.init
def trigger_prod_sl_ids_init(self: Producer) -> None:
    """
    Initialize the single lepton trigger producer, triggers can be set in the trigger config.
    """
    self.uses.add("trigger_ids")


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
        "emu_old": [301, 401, 201, 101],
    }
    ee_trigger_sequence = {
        "ee_dilep": [202, 204],
        "ee_single": [201],
        "ee_electronjet": [203],
        "ee_old": [202, 201],
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

    mixed_trigger_sequence = {
        "emu_dilep": ["Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL", "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ"],
        "emu_single": ["Ele30_WPTight_Gsf", "IsoMu24"],
        "emu_electronjet": ["Ele50_CaloIdVT_GsfTrkIdT_PFJet165"],
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
                triggers_mask = triggers_mask | (events.HLT[trigger])
                seq_trigger = seq_trigger | (events.HLT[trigger])

            trig_passed = ak.where(triggers_mask, [[label]], [[]])
            trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

            seq_label += label + "+"
            if "+" in seq_label[:-1]:
                trig_passed = ak.where(seq_trigger, [[seq_label[:-1]]], [[]])
                trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    events = set_ak_column(events, "trig_ids", trig_ids)

    return events


# initialize the trigger producer, triggers can be set in the trigger config
@trigger_prod_db.init
def trigger_prod_db_init(self: Producer) -> None:

    for trigger in self.config_inst.x("triggers", []):
        self.uses.add(f"HLT.{trigger.hlt_field}")
    self.uses.add("run")


# checks to see what the individual trigger checks do
@producer(
    produces={"trig_ids"},
    version=1,
)
def trigger_check(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Checks the individual trigger checks and returns a column with the results.
    This is only for debugging purposes.
    """

    from hbw.util import debugger
    debugger()

    return events


# initialize the trigger check producer
@trigger_check.init
def trigger_check_init(self: Producer) -> None:

    for trigger in self.config_inst.x.triggers:
        self.uses.add(f"HLT.{trigger.hlt_field}")
