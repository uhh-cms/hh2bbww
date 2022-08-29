# coding: utf-8

"""
Selection methods for HHtobbWW.
"""

from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production.categories import category_ids
from columnflow.production.processes import process_ids

from hbw.selection.general import increment_stats, jet_energy_shifts
from hbw.selection.cutflow_features import cutflow_features

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(uses={"Jet.pt", "Jet.eta"})
def req_jet(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    mask = (events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4)
    return mask


@selector(uses={"event"})
def sel_incl(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    # select all
    return ak.ones_like(events.event)


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


@selector(uses={req_jet}, produces={"jet_high_multiplicity"}, exposed=True)
def jet_selection(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # example cuts:
    # - require at least 4 jets with pt>30, eta<2.4
    # example columns:
    # - high jet multiplicity region (>=6 selected jets)

    jet_mask = self[req_jet](events)
    jet_sel = ak.num(jet_mask, axis=1) >= 4

    jet_high_multiplicity = ak.num(jet_mask, axis=1) >= 6
    events = set_ak_column(events, "jet_high_multiplicity", jet_high_multiplicity)

    # determine the masked jet indices
    jet_indices = masked_sorted_indices(jet_mask, events.Jet.pt)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={"Jet": jet_sel},
        objects={"Jet": {"Jet": jet_indices}},
    )


@selector(uses={"Electron.pt", "Electron.eta", "Muon.pt", "Muon.eta"})
def lepton_selection(
        self: Selector,
        events: ak.Array,
        stats: defaultdict,
        **kwargs,
) -> Tuple[ak.Array, SelectionResult]:

    e_mask = (events.Electron.pt) > 30 & (abs(events.Electron.eta) < 2.4)
    mu_mask = (events.Muon.pt) > 30 & (abs(events.Muon.eta) < 2.4)

    lep_sel = ak.sum(e_mask, axis=-1) + ak.sum(mu_mask, axis=-1) == 1

    # determine the masked lepton indices
    e_indices = masked_sorted_indices(e_mask, events.Electron.pt)
    mu_indices = masked_sorted_indices(mu_mask, events.Muon.pt)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={"Lepton": lep_sel},
        objects={"Electron": {"Electron": e_indices}, "Muon": {"Muon": mu_indices}},
    )


@selector(
    uses={
        jet_selection, lepton_selection, cutflow_features,
        category_ids, process_ids, increment_stats, "mc_weight",
    },
    produces={
        jet_selection, lepton_selection, cutflow_features,
        category_ids, process_ids, increment_stats, "mc_weight",
    },
    shifts={
        jet_energy_shifts,
    },
    exposed=True,
)
def default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # example cuts:
    # - jet_selection_test
    # example stats:
    # - number of events before and after selection
    # - sum of mc weights before and after selection

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # jet selection
    events, jet_results = self[jet_selection](events, stats, **kwargs)
    results += jet_results

    # lepton selection
    events, lepton_results = self[lepton_selection](events, stats, **kwargs)
    results += lepton_results

    # combined event selection after all steps
    event_sel = (
        jet_results.steps.Jet &
        lepton_results.steps.Lepton
    )
    results.main["event"] = event_sel

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add cutflow features
    events = self[cutflow_features](events, results=results, **kwargs)

    # increment stats
    self[increment_stats](events, event_sel, stats, **kwargs)

    return events, results
