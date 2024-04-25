# coding: utf-8
"""
Producer for synchronization exercises.
"""
from columnflow.production import Producer, producer

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from hbw.production.prepare_objects import custom_collections

np = maybe_import("numpy")
ak = maybe_import("awkward")

int_cols = {
    "event_nr", "run_nr", "ls",
    "nAK4", "nAK4_btag", "nAK8_btag", "nFakeMuon", "nFakeElectron", "nTightMuon", "nTightElectron",
}


def get_columns_to_store(config_inst):
    columns_to_store = {
        "event_nr": lambda events: events.event,
        "run_nr": lambda events: events.run,
        "ls": lambda events: events.luminosityBlock,
        "nAK4": lambda events: ak.sum(events.Jet.pt > 0, axis=1),
        "nAK4_btag": lambda events: (
            ak.sum(events.Jet[config_inst.x.btag_column] > config_inst.x.btag_wp_score, axis=-1)
        ),
        "nAK8_btag": lambda events: ak.sum(events.HbbJet.pt > 0, axis=1),
        "nFakeElectron": lambda events: ak.sum(events.Electron.pt > 0, axis=1),
        "nTightElectron": lambda events: ak.sum(events.Electron.is_tight, axis=1),
        "nFakeMuon": lambda events: ak.sum(events.Muon.pt > 0, axis=1),
        "nTightMuon": lambda events: ak.sum(events.Muon.is_tight, axis=1),
        "lepton0_pt": lambda events: events.Lepton.pt[:, 0],
        "lepton0_eta": lambda events: events.Lepton.eta[:, 0],
        "lepton0_phi": lambda events: events.Lepton.phi[:, 0],
        # "lepton0_relIso": lambda events: events.Lepton.jetRelIso[:, 0],
        # "lepton0_pdgId": lambda events: events.Lepton.pdgId[:, 0],
        "lepton1_pt": lambda events: events.Lepton.pt[:, 1],
        "lepton1_eta": lambda events: events.Lepton.eta[:, 1],
        "lepton1_phi": lambda events: events.Lepton.phi[:, 1],
        # "lepton1_relIso": lambda events: events.Lepton.jetRelIso[:, 1],
        # "lepton1_pdgId": lambda events: events.Lepton.pdgId[:, 1],
        "ak4jet0_pt": lambda events: events.Bjet.pt[:, 0],
        "ak4jet0_eta": lambda events: events.Bjet.eta[:, 0],
        "ak4jet0_phi": lambda events: events.Bjet.phi[:, 0],
        "ak4jet0_btagDeepFlavB": lambda events: events.Bjet.btagDeepFlavB[:, 0],
        # "ak4jet0_btagPNetB": lambda events: events.Bjet.btagPNetB[:, 0],
        "ak4jet1_pt": lambda events: events.Bjet.pt[:, 1],
        "ak4jet1_eta": lambda events: events.Bjet.eta[:, 1],
        "ak4jet1_phi": lambda events: events.Bjet.phi[:, 1],
        "ak4jet1_btagDeepFlavB": lambda events: events.Bjet.btagDeepFlavB[:, 1],
        # "ak4jet1_btagPNetB": lambda events: events.Bjet.btagPNetB[:, 1],
        "ak4jet2_pt": lambda events: events.Lightjet.pt[:, 0],
        "ak4jet2_eta": lambda events: events.Lightjet.eta[:, 0],
        "ak4jet2_phi": lambda events: events.Lightjet.phi[:, 0],
        "ak4jet2_btagDeepFlavB": lambda events: events.Lightjet.btagDeepFlavB[:, 0],
        # "ak4jet2_btagPNetB": lambda events: events.Lightjet.btagPNetB[:, 0],
        "ak4jet3_pt": lambda events: events.Lightjet.pt[:, 1],
        "ak4jet3_eta": lambda events: events.Lightjet.eta[:, 1],
        "ak4jet3_phi": lambda events: events.Lightjet.phi[:, 1],
        "ak4jet3_btagDeepFlavB": lambda events: events.Lightjet.btagDeepFlavB[:, 1],
        # "ak4jet3_btagPNetB": lambda events: events.Lightjet.btagPNetB[:, 1],
        "ak8jet0_pt": lambda events: events.HbbJet.pt[:, 0],
        "ak8jet0_eta": lambda events: events.HbbJet.eta[:, 0],
        "ak8jet0_phi": lambda events: events.HbbJet.phi[:, 0],
        "ak8jet0_msoftdrop": lambda events: events.HbbJet.msoftdrop[:, 0],
        "met_pt": lambda events: events.MET.pt,
        "met_phi": lambda events: events.MET.phi,
        "mc_weight": lambda events: events.mc_weight,
        "normalized_pu_weight": lambda events: events.pu_weight,
        "normalized_btag_weight": lambda events: events.normalized_btag_weight,
        "normalized_njet_btag_weight": lambda events: events.normalized_njet_btag_weight,
        "pdf_weight": lambda events: events.pdf_weight,
        "murf_envelope_weight": lambda events: events.murf_envelope_weight,
    }
    return columns_to_store


def pad_to(events, obj, length):
    # this will not be the correct behavior, but it still works fine for this usecase
    empty_obj = {col: -9999 for col in events[obj].fields}
    return ak.fill_none(ak.pad_none(events[obj], length), empty_obj)


def get_flat_array(events, config_inst):
    columns_to_store = get_columns_to_store(config_inst)
    arrays = {column_key: column_func(events) for column_key, column_func in columns_to_store.items()}

    return ak.Array(arrays)

@producer(uses={"*"}, produces={"*"}, exposed=True)
def sync_exercise(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # prepare object behavior
    collections = custom_collections.copy()
    collections["Electron"] = {
        "type_name": "Muon",  # make Electron to a Muon to allow concatenating them
        "check_attr": "metric_table",
        "skip_fields": "*Idx*G",
    }
    events = attach_coffea_behavior.call_func(None, events, collections=collections)

    # object padding
    lepton = ak.concatenate([events.Muon, events.Electron], axis=-1)
    events = set_ak_column(events, "Lepton", lepton[ak.argsort(lepton.pt, ascending=False)])
    events = set_ak_column(events, "Lepton", pad_to(events, "Lepton", 2))
    events = set_ak_column(events, "Bjet", pad_to(events, "Bjet", 2))
    events = set_ak_column(events, "Lightjet", pad_to(events, "Lightjet", 2))
    events = set_ak_column(events, "HbbJet", pad_to(events, "HbbJet", 1))

    # transform to flat array (copy of events)
    columns_to_store = get_columns_to_store(self.config_inst)

    arrays = ak.Array({
        column_key: column_func(events)
        for column_key, column_func in columns_to_store.items()
    })

    return arrays
