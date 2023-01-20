# coding: utf-8

"""
Selection methods defining categories based on selection step results.
"""

import functools

from columnflow.util import maybe_import
from columnflow.selection import Selector, SelectionResult, selector

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(uses={"event"})
def catid_selection_incl(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    return ak.ones_like(events.event)


@selector(uses={"event"})
def catid_selection_1e(self: Selector, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:
    return (ak.num(results.objects.Electron.Electron, axis=-1) == 1) & (ak.num(results.objects.Muon.Muon, axis=-1) == 0)


@selector(uses={"event"})
def catid_selection_1mu(self: Selector, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:
    return (ak.num(results.objects.Electron.Electron, axis=-1) == 0) & (ak.num(results.objects.Muon.Muon, axis=-1) == 1)


@selector(uses={"Electron.pt", "Muon.pt"})
def catid_1e(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    return (ak.sum(events.Electron.pt > 0, axis=-1) == 1) & (ak.sum(events.Muon.pt > 0, axis=-1) == 0)


@selector(uses={"Electron.pt", "Muon.pt"})
def catid_1mu(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    return (ak.sum(events.Electron.pt > 0, axis=-1) == 0) & (ak.sum(events.Muon.pt > 0, axis=-1) == 1)


@selector(uses={"Jet.pt", "FatJet.pt"})
def catid_boosted(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    """
    Categorization of events in the boosted category: presence of at least 1 AK8 jet fulfilling
    requirements given by the Selector called in SelectEvents
    """
    return (ak.sum(events.Jet.pt > 0, axis=-1) >= 1) & (ak.sum(events.FatJet.pt > 0, axis=-1) >= 1)


@selector(uses={"Jet.pt", "FatJet.pt"})
def catid_resolved(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    """
    Categorization of events in the resolved category: presence of no AK8 jets fulfilling
    requirements given by the Selector called in SelectEvents
    """
    return (ak.sum(events.Jet.pt > 0, axis=-1) >= 3) & (ak.sum(events.FatJet.pt > 0, axis=-1) == 0)


@selector(uses={catid_resolved, "Jet.btagDeepFlavB"})
def catid_resolved_1b(self: Selector, events: ak.Array, **kwargs) -> ak.Array:

    n_deepjet = ak.sum(events.Jet.btagDeepFlavB >= self.config_inst.x.btag_working_points.deepjet.medium, axis=-1)

    return self[catid_resolved](events, **kwargs) & (n_deepjet == 1)


@selector(uses={catid_resolved, "Jet.btagDeepFlavB"})
def catid_resolved_2b(self: Selector, events: ak.Array, **kwargs) -> ak.Array:

    n_deepjet = ak.sum(events.Jet.btagDeepFlavB >= self.config_inst.x.btag_working_points.deepjet.medium, axis=-1)

    return self[catid_resolved](events, **kwargs) & (n_deepjet >= 2)


for lep_ch in ["1e", "1mu"]:
    for jet_ch in ["resolved_1b", "resolved_2b", "boosted"]:
        for dnn_ch in ["dummy"]:

            funcs = {
                "lep_func": locals()[f"catid_{lep_ch}"],
                "jet_func": locals()[f"catid_{jet_ch}"],
                # "dnn_func": locals()[f"catid_{dnn_ch}"],
            }

            # @selector(name=f"catid_{lep_ch}_{jet_ch}")
            @selector(
                uses=set(funcs.values()),
                produces=set(funcs.values()),
                cls_name=f"catid_{lep_ch}_{jet_ch}",
            )
            def catid_leaf_mask(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
                """
                Selector that call multiple functions and combines their outputs via
                logical `and`.
                """

                masks = [self[func](events, **kwargs) for func in self.uses]
                leaf_mask = functools.reduce((lambda a, b: a & b), masks)

                return leaf_mask
