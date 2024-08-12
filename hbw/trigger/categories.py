# coding: utf-8
"""
defines categories based on the selection used for the trigger studies
"""

from __future__ import annotations

from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer

np = maybe_import("numpy")
ak = maybe_import("awkward")


# categorizer muon channel
@categorizer(uses={"Muon.pt"}, call_force=True)
def catid_trigger_mu(
    self: Categorizer,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:

    mask = (ak.sum(events.Muon.pt > 15, axis=1) >= 1)
    return events, mask


# categorizer electron channel
@categorizer(uses={"Electron.pt"}, call_force=True)
def catid_trigger_ele(
    self: Categorizer,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:

    mask = (ak.sum(events.Electron.pt > 15, axis=1) >= 1)
    return events, mask


# categorizer for orthogonal measurement (muon channel)
@categorizer(uses={"Jet.pt", "Muon.pt", "Electron.pt", "HLT.*"}, call_force=True)
def catid_trigger_orth_mu(
    self: Categorizer,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:

    mask = (
        (ak.sum(events.Muon.pt > 15, axis=1) >= 1) &
        (ak.sum(events.Electron.pt > 15, axis=1) >= 1) &
        (ak.sum(events.Jet.pt > 25, axis=1) >= 3) &
        (events.HLT[self.config_inst.x.ref_trigger["mu"]])
    )
    return events, mask


# categorizer for orthogonal measurement (electron channel)
@categorizer(uses={"Jet.pt", "Muon.pt", "Electron.pt", "HLT.*"}, call_force=True)
def catid_trigger_orth_ele(
    self: Categorizer,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:

    mask = (
        (ak.sum(events.Muon.pt > 15, axis=1) >= 1) &
        (ak.sum(events.Electron.pt > 15, axis=1) >= 1) &
        (ak.sum(events.Jet.pt > 25, axis=1) >= 3) &
        (events.HLT[self.config_inst.x.ref_trigger["e"]])
    )
    return events, mask
