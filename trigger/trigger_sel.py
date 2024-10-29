# coding: utf-8

"""
Selector for triggerstudies
"""

from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import
from columnflow.selection import Selector, SelectionResult, selector

from hbw.selection.common import masked_sorted_indices, configure_selector, pre_selection, post_selection
from hbw.util import four_vec
from hbw.selection.lepton import lepton_definition
from hbw.selection.jet import jet_selection

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses=(four_vec({"Muon", "Electron"})),
    exposed=True,
    version=1,
)
def trigger_sel(
        self: Selector,
        events: ak.Array,
        stats: defaultdict,
        **kwargs,
) -> Tuple[ak.Array, SelectionResult]:

    # get preselection and object definition
    events, results = self[pre_selection](events, stats, **kwargs)

    #  get lepton and jet selection
    events, lepton_results = self[lepton_definition](events, stats, **kwargs)
    results += lepton_results

    events, jet_results = self[jet_selection](events, lepton_results, stats, **kwargs)
    results += jet_results

    # get leptons and check pt
    muons = events.Muon[results.aux["mu_mask_tight"]]
    electrons = events.Electron[results.aux["e_mask_tight"]]

    # muon SFs are only available for muons with pt > 15 GeV
    # electron threshold could be lower, for now set to same value as muon pt
    muon_mask = ak.sum(muons.pt > 15, axis=1) >= 1
    electron_mask = ak.sum(electrons.pt > 15, axis=1) >= 1

    results.steps["TrigMuMask"] = muon_mask
    results.steps["TrigEleMask"] = electron_mask

    # set lepton objects (these will be the objects used after cf.ReduceColumns)
    results.objects["Muon"]["Muon"] = masked_sorted_indices(
        (results.aux["mu_mask_tight"] & (events.Muon.pt > 15)),
        events.Muon.pt,
    )
    results.objects["Electron"]["Electron"] = masked_sorted_indices(
        (results.aux["e_mask_tight"] & (events.Electron.pt > 15)),
        events.Electron.pt,
    )

    # save selection results for different channels
    results.steps["SR_mu"] = (
        results.steps.cleanup &
        results.steps.nJet3 &
        results.steps.nBjet1 &
        results.steps.TrigMuMask
    )
    results.steps["SR_ele"] = (
        results.steps.cleanup &
        results.steps.nJet3 &
        results.steps.nBjet1 &
        results.steps.TrigEleMask
    )
    results.steps["all_but_bjet"] = (
        results.steps.cleanup &
        results.steps.nJet3 &
        (results.steps.TrigMuMask | results.steps.TrigEleMask)
    )

    results.steps["all"] = results.event = (
        results.steps.SR_ele | results.steps.SR_mu
    )

    # some categories not used for the trigger studies (a.t.m.) need additional selection steps
    # at some point it probably makes sense to completely use the object definition and categories of the main analysis
    results.steps["SR"] = results.steps.SR_ele | results.steps.SR_mu
    results.steps["Fake"] = results.steps.SR_ele & results.steps.SR_mu

    # apply postselection
    events, results = self[post_selection](events, results, stats, **kwargs)

    return events, results


@trigger_sel.init
def trigger_sel_init(self: Selector) -> None:

    # init gets called multiple times, so we need to check if the config and dataset instances are already set
    if not getattr(self, "config_inst", None) or not getattr(self, "dataset_inst", None):
        return

    configure_selector(self)

    self.uses.add(pre_selection)
    self.uses.add(post_selection)
    self.uses.add(lepton_definition)
    self.uses.add(jet_selection)

    self.produces = {
        pre_selection,
        post_selection,
        lepton_definition,
        jet_selection,
    }
