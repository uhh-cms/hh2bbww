# coding: utf-8

"""
Selection methods for HHtobbWW.
"""

from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production.categories import category_ids
from columnflow.production.processes import process_ids
from hbw.production.weights import event_weights_to_normalize
from hbw.production.gen_hbw_decay import gen_hbw_decay_products
from hbw.selection.general import jet_energy_shifts  # , increment_stats
from hbw.selection.stats import increment_stats
from hbw.selection.cutflow_features import cutflow_features
from hbw.selection.gen_hbw_features import gen_hbw_decay_features

np = maybe_import("numpy")
ak = maybe_import("awkward")


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.btagDeepFlavB"},
    produces={"jet_high_multiplicity"},
    shifts={jet_energy_shifts},
    exposed=True,
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # HH -> bbWW(qqlnu) jet selection
    # - require at least 3 jets with pt>30, eta<2.4
    # - require at least 1 jet with pt>30, eta<2.4, b-score>0.3040 (Medium WP)

    # jets
    jet_mask = (events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4)
    jet_sel = ak.sum(jet_mask, axis=1) >= 3
    jet_indices = masked_sorted_indices(jet_mask, events.Jet.pt)

    # b-tagged jets, medium working point
    wp_med = self.config_inst.x.btag_working_points.deepjet.medium
    bjet_mask = (jet_mask) & (events.Jet.btagDeepFlavB >= wp_med)
    bjet_sel = ak.sum(bjet_mask, axis=1) >= 1

    # define b-jets as the two b-score leading jets, b-score sorted
    bjet_indices = masked_sorted_indices(jet_mask, events.Jet.btagDeepFlavB)[:, :2]

    # define lightjets as all non b-jets, pt-sorted
    b_idx = ak.fill_none(ak.pad_none(bjet_indices, 2), -1)
    lightjet_indices = jet_indices[(jet_indices != b_idx[:, 0]) & (jet_indices != b_idx[:, 1])]

    # example column: high jet multiplicity region (>=6 jets)
    events = set_ak_column(events, "jet_high_multiplicity", ak.sum(jet_mask, axis=1) >= 6)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={"Jet": jet_sel, "Bjet": bjet_sel},
        # NOTE: we should probably not produce Bjet/Lightjet columns in ReduceEvents;
        #       rather do this in ProduceColumns when needed
        objects={"Jet": {"Jet": jet_indices, "Bjet": bjet_indices, "Lightjet": lightjet_indices}},
        aux={
            "jet_mask": jet_mask,
            "n_central_jets": ak.num(jet_indices),
        },
    )


@selector(uses={
    "Electron.pt", "Electron.eta", "Electron.cutBased",
    "Muon.pt", "Muon.eta", "Muon.tightId", "Muon.looseId", "Muon.pfRelIso04_all",
    "HLT.Ele35_WPTight_Gsf", "HLT.IsoMu27",
})
def lepton_selection(
        self: Selector,
        events: ak.Array,
        stats: defaultdict,
        **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # HH -> bbWW(qqlnu) lepton selection
    # - require exactly 1 lepton (e or mu) with pt_e>28 / pt_mu>25, eta<2.4 and tight ID
    # - veto additional leptons (TODO define exact cuts)
    # - require that events are triggered by SingleMu or SingleEle trigger

    # Veto Lepton masks (TODO define exact cuts)
    e_mask_veto = (events.Electron.pt > 15) & (abs(events.Electron.eta) < 2.4) & (events.Electron.cutBased >= 1)
    mu_mask_veto = (events.Muon.pt > 15) & (abs(events.Muon.eta) < 2.4) & (events.Muon.looseId)

    lep_veto_sel = ak.sum(e_mask_veto, axis=-1) + ak.sum(mu_mask_veto, axis=-1) <= 1

    # 2017 Trigger selection (TODO different triggers based on year of data-taking)
    trigger_sel = (events.HLT.Ele35_WPTight_Gsf) | (events.HLT.IsoMu27)

    # Lepton definition for this analysis
    e_mask = (events.Electron.pt > 28) & (abs(events.Electron.eta) < 2.4) & (events.Electron.cutBased == 4)
    mu_mask = (
        (events.Muon.pt > 25) &
        (abs(events.Muon.eta) < 2.4) &
        (events.Muon.tightId) &
        (events.Muon.pfRelIso04_all < 0.15)
    )

    lep_sel = ak.sum(e_mask, axis=-1) + ak.sum(mu_mask, axis=-1) == 1
    # e_sel = (ak.sum(e_mask, axis=-1) == 1) & (ak.sum(mu_mask, axis=-1) == 0)
    mu_sel = (ak.sum(e_mask, axis=-1) == 0) & (ak.sum(mu_mask, axis=-1) == 1)

    # determine the masked lepton indices
    e_indices = masked_sorted_indices(e_mask, events.Electron.pt)
    mu_indices = masked_sorted_indices(mu_mask, events.Muon.pt)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "Lepton": lep_sel, "VetoLepton": lep_veto_sel, "Trigger": trigger_sel,
            "Muon": mu_sel,  # for comparing results with Msc Analysis
        },
        objects={"Electron": {"Electron": e_indices}, "Muon": {"Muon": mu_indices}},
    )


@selector(
    uses={
        jet_selection, lepton_selection, cutflow_features,
        category_ids, process_ids, event_weights_to_normalize, increment_stats, attach_coffea_behavior,
        "mc_weight",  # not opened per default but always required in Cutflow tasks
    },
    produces={
        jet_selection, lepton_selection, cutflow_features,
        category_ids, process_ids, event_weights_to_normalize, increment_stats, attach_coffea_behavior,
        "mc_weight",  # not opened per default but always required in Cutflow tasks
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
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # jet selection
    events, jet_results = self[jet_selection](events, stats, **kwargs)
    results += jet_results

    # lepton selection
    events, lepton_results = self[lepton_selection](events, stats, **kwargs)
    results += lepton_results

    # combined event selection after all steps except b-jet selection
    results.steps["all_but_bjet"] = (
        results.steps.Jet &
        results.steps.Lepton &
        results.steps.Trigger
    )
    # combined event selection after all steps
    results.main["event"] = (
        results.steps.all_but_bjet &
        results.steps.Bjet
    )

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add cutflow features
    events = self[cutflow_features](events, results=results, **kwargs)

    # produce event weights
    events = self[event_weights_to_normalize](events, results=results, **kwargs)

    # increment stats
    self[increment_stats](events, results, stats, **kwargs)

    return events, results


@selector(
    uses={
        default, "mc_weight",  # mc_weight should be included from default
        gen_hbw_decay_products, gen_hbw_decay_features,
    },
    produces={
        category_ids, process_ids, increment_stats, "mc_weight",
        gen_hbw_decay_products, gen_hbw_decay_features,
    },
    exposed=True,
)
def gen_hbw(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """
    Selector that is used to perform GenLevel studies but also allow categorization and event selection
    using the default reco-level selection.
    Should only be used for HH samples
    """

    if not self.dataset_inst.x("is_hbw", False):
        raise Exception("This selector is only usable for HH samples")

    # run the default Selector
    events, results = self[default](events, stats, **kwargs)

    # extract relevant gen HH decay products
    events = self[gen_hbw_decay_products](events, **kwargs)

    # produce relevant columns
    events = self[gen_hbw_decay_features](events, **kwargs)

    return events, results
