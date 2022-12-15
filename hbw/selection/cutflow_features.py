# coding: utf-8

"""
Selectors to set ak columns for cutflow features
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, Route, EMPTY_FLOAT
from columnflow.selection import Selector, SelectionResult, selector
from hbw.production.prepare_objects import prepare_objects
ak = maybe_import("awkward")


@selector(
    uses={prepare_objects, "Jet.pt"},
    produces={
        "cutflow.loose_jet1_pt", "cutflow.loose_jet2_pt", "cutflow.loose_jet3_pt", "cutflow.loose_jet4_pt",
        "cutflow.veto_lepton1_pt", "cutflow.veto_lepton2_pt", "cutflow.veto_lepton3_pt",
        "cutflow.n_electron", "cutflow.n_muon", "cutflow.n_lepton",
        "cutflow.n_veto_electron", "cutflow.n_veto_muon", "cutflow.n_veto_lepton",
    },
)
def cutflow_features(self: Selector, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:

    # apply event results to objects and define objects in a convenient way for reconstructing variables
    # but create temporary ak.Array to not override objects in events
    arr = self[prepare_objects](events, results)

    # Nummber of objects
    events = set_ak_column(events, "cutflow.n_electron", ak.num(arr.Electron, axis=1))
    events = set_ak_column(events, "cutflow.n_muon", ak.num(arr.Muon, axis=1))
    events = set_ak_column(events, "cutflow.n_lepton", ak.num(arr.Lepton, axis=1))
    events = set_ak_column(events, "cutflow.n_veto_electron", ak.num(arr.VetoElectron, axis=1))
    events = set_ak_column(events, "cutflow.n_veto_muon", ak.num(arr.VetoMuon, axis=1))
    events = set_ak_column(events, "cutflow.n_veto_lepton", ak.num(arr.VetoLepton, axis=1))

    # loose jets
    for i in range(4):
        events = set_ak_column(
            events, f"cutflow.loose_jet{i+1}_pt",
            Route(f"pt[:, {i}]").apply(arr.LooseJet, EMPTY_FLOAT),
        )
    # veto leptons
    for i in range(3):
        events = set_ak_column(
            events, f"cutflow.veto_lepton{i+1}_pt",
            Route(f"pt[:, {i}]").apply(arr.VetoLepton, EMPTY_FLOAT),
        )
    return events
