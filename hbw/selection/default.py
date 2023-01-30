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
from hbw.selection.stats import increment_stats
from hbw.selection.cutflow_features import cutflow_features
from hbw.selection.gen_hbw_features import gen_hbw_decay_features, gen_hbw_matching

np = maybe_import("numpy")
ak = maybe_import("awkward")


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB"},
    exposed=True,
)
def vbf_jet_selection(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # TODO: remove the 4 HH jet candidates from fjets

    # assign local index to all Jets
    events = set_ak_column(events, "Jet.local_index", ak.local_index(events.Jet))

    # default requirements for vbf jets (pt, eta and no H->bb jet)
    # NOTE: we might also want to remove the two H->jj jet candidates
    # TODO: how to get the object mask from the object indices in a more convenient way?
    b_indices = ak.fill_none(ak.pad_none(results.objects.Jet.Bjet, 2), -1)
    vbf_jets = events.Jet[(events.Jet.local_index != b_indices[:, 0]) & (events.Jet.local_index != b_indices[:, 1])]
    vbf_jets = vbf_jets[(vbf_jets.pt > 30) & (abs(vbf_jets.eta < 4.7))]

    # build all possible pairs of jets fulfilling the `vbf_jet_mask` requirement
    vbf_pairs = ak.combinations(vbf_jets, 2)
    vbf1, vbf2 = ak.unzip(vbf_pairs)

    # define requirements for vbf pair candidates
    vbf_pairs["deta"] = abs(vbf1.eta - vbf2.eta)
    vbf_pairs["invmass"] = (vbf1 + vbf2).mass
    vbf_mask = (vbf_pairs.deta > 3) & (vbf_pairs.invmass > 500)

    # event selection: at least one vbf pair present (TODO: use it for categorization)
    vbf_selection = ak.sum(vbf_mask >= 1, axis=-1) >= 1

    # apply requirements to vbf pairs
    vbf_pairs = vbf_pairs[vbf_mask]

    # choose the vbf pair based on maximum delta eta
    chosen_vbf_pair = vbf_pairs[ak.singletons(ak.argmax(vbf_pairs.deta, axis=1))]

    # get the local indices (pt sorted)
    vbf1, vbf2 = [chosen_vbf_pair[i] for i in ["0", "1"]]
    vbf_jets = ak.concatenate([vbf1, vbf2], axis=1)
    vbf_jets = vbf_jets[ak.argsort(vbf_jets.pt, ascending=False)]

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={"VBFJetPair": vbf_selection},
        objects={"Jet": {
            "VBFJet": vbf_jets.local_index,
        }},
    )


@selector(
    uses={
        "Jet.pt", "Jet.eta", "Jet.jetId",
        "FatJet.pt", "FatJet.eta",
    },
    produces={"cutflow.n_fatjet"},
    exposed=True,
)
def boosted_jet_selection(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # HH -> bbWW(qqlnu) boosted selection

    # ak4 jets
    ak4_jets = events.Jet[results.objects.Jet.Jet]
    ak4_jet_sel = ak.num(ak4_jets, axis=1) > 0

    # fatjet mask
    fatjet_mask = (events.FatJet.pt > 200) & (abs(events.FatJet.eta) < 2.4)
    events = set_ak_column(events, "cutflow.n_fatjet", ak.sum(fatjet_mask, axis=1))

    fatjet_sel = events.cutflow.n_fatjet >= 1

    boosted_sel = ak4_jet_sel & fatjet_sel

    fatjet_indices = masked_sorted_indices(fatjet_mask, events.FatJet.pt)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "Boosted": boosted_sel,
        },
        objects={
            "FatJet": {
                "FatJet": fatjet_indices,
            },
        },
    )


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.btagDeepFlavB", "Jet.jetId"},
    produces={"cutflow.n_jet", "cutflow.n_deepjet_med"},
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
    jet_mask_loose = (events.Jet.pt > 5) & abs(events.Jet.eta < 2.4)
    jet_mask = (events.Jet.pt > 25) & (abs(events.Jet.eta) < 2.4) & (events.Jet.jetId == 6)
    events = set_ak_column(events, "cutflow.n_jet", ak.sum(jet_mask, axis=1))
    jet_sel = events.cutflow.n_jet >= 3
    jet_indices = masked_sorted_indices(jet_mask, events.Jet.pt)

    # b-tagged jets, medium working point
    wp_med = self.config_inst.x.btag_working_points.deepjet.medium
    btag_mask = (jet_mask) & (events.Jet.btagDeepFlavB >= wp_med)
    events = set_ak_column(events, "cutflow.n_deepjet_med", ak.sum(btag_mask, axis=1))
    btag_sel = events.cutflow.n_deepjet_med >= 1

    # define b-jets as the two b-score leading jets, b-score sorted
    bjet_indices = masked_sorted_indices(jet_mask, events.Jet.btagDeepFlavB)[:, :2]

    # define lightjets as all non b-jets, pt-sorted
    b_idx = ak.fill_none(ak.pad_none(bjet_indices, 2), -1)
    lightjet_indices = jet_indices[(jet_indices != b_idx[:, 0]) & (jet_indices != b_idx[:, 1])]

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={"Jet": jet_sel, "Bjet": btag_sel},
        objects={
            "Jet": {
                "LooseJet": masked_sorted_indices(jet_mask_loose, events.Jet.pt),
                "Jet": jet_indices,
                "Bjet": bjet_indices,
                "Lightjet": lightjet_indices,
            },
        },
        aux={
            "jet_mask": jet_mask,
            "n_central_jets": ak.num(jet_indices),
        },
    )


@selector(uses={
    "Electron.pt", "Electron.eta", "Electron.cutBased", "Electron.mvaFall17V2Iso_WP80",
    "Muon.pt", "Muon.eta", "Muon.tightId", "Muon.looseId", "Muon.pfRelIso04_all",
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
    e_mask_veto = (events.Electron.pt > 1) & (abs(events.Electron.eta) < 2.4) & (events.Electron.cutBased >= 1)
    mu_mask_veto = (events.Muon.pt > 1) & (abs(events.Muon.eta) < 2.4) & (events.Muon.looseId)

    lep_veto_sel = ak.sum(e_mask_veto, axis=-1) + ak.sum(mu_mask_veto, axis=-1) <= 1

    trigger_sel = (events.HLT[self.mu_trigger]) | (events.HLT[self.ele_trigger])

    # Lepton definition for this analysis
    e_mask = (
        (events.Electron.pt > self.ele_pt) &
        (abs(events.Electron.eta) < 2.4) &
        (events.Electron.cutBased == 4) &
        (events.Electron.mvaFall17V2Iso_WP80 == 1)
    )
    mu_mask = (
        (events.Muon.pt > self.mu_pt) &
        (abs(events.Muon.eta) < 2.4) &
        (events.Muon.tightId) &
        (events.Muon.pfRelIso04_all < 0.15)
    )
    lep_sel = ak.sum(e_mask, axis=-1) + ak.sum(mu_mask, axis=-1) == 1
    # e_sel = (ak.sum(e_mask, axis=-1) == 1) & (ak.sum(mu_mask, axis=-1) == 0)
    mu_sel = (ak.sum(e_mask, axis=-1) == 0) & (ak.sum(mu_mask, axis=-1) == 1)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "Lepton": lep_sel, "VetoLepton": lep_veto_sel, "Trigger": trigger_sel,
            "Muon": mu_sel,  # for comparing results with Msc Analysis
        },
        objects={
            "Electron": {
                "VetoElectron": masked_sorted_indices(e_mask_veto, events.Electron.pt),
                "Electron": masked_sorted_indices(e_mask, events.Electron.pt),
            },
            "Muon": {
                "VetoMuon": masked_sorted_indices(mu_mask_veto, events.Muon.pt),
                "Muon": masked_sorted_indices(mu_mask, events.Muon.pt),
            },
        },
    )


@lepton_selection.init
def lepton_selection_init(self: Selector) -> None:
    year = self.config_inst.campaign.x.year

    # Trigger choice based on year of data-taking (for now: only single trigger)
    self.mu_trigger = {
        2016: "IsoMu24",  # or "IsoTkMu27")
        2017: "IsoMu27",
        2018: "IsoMu24",
    }[year]
    self.ele_trigger = {
        2016: "Ele27_WPTight_Gsf",  # or "HLT_Ele115_CaloIdVT_GsfTrkIdT", "HLT_Photon175")
        2017: "Ele35_WPTight_Gsf",  # or "HLT_Ele115_CaloIdVT_GsfTrkIdT", "HLT_Photon200")
        2018: "Ele32_WPTight_Gsf",  # or "HLT_Ele115_CaloIdVT_GsfTrkIdT", "HLT_Photon200")
    }[year]

    self.uses |= {f"HLT.{self.mu_trigger}", f"HLT.{self.ele_trigger}"}

    # Lepton pt thresholds based on year (1 pt above trigger threshold)
    self.ele_pt = {2016: 28, 2017: 36, 2018: 33}[year]
    self.mu_pt = {2016: 25, 2017: 28, 2018: 25}[year]


@selector(
    uses={
        boosted_jet_selection,
        jet_selection, vbf_jet_selection, lepton_selection, cutflow_features,
        category_ids, process_ids, increment_stats, attach_coffea_behavior,
        "mc_weight",  # not opened per default but always required in Cutflow tasks
    },
    produces={
        boosted_jet_selection,
        jet_selection, vbf_jet_selection, lepton_selection, cutflow_features,
        category_ids, process_ids, increment_stats, attach_coffea_behavior,
        "mc_weight",  # not opened per default but always required in Cutflow tasks
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

    # boosted selection
    events, boosted_results = self[boosted_jet_selection](events, results, stats, **kwargs)
    results += boosted_results

    # vbf-jet selection
    events, vbf_jet_results = self[vbf_jet_selection](events, results, stats, **kwargs)
    results += vbf_jet_results

    # lepton selection
    events, lepton_results = self[lepton_selection](events, stats, **kwargs)
    results += lepton_results

    # combined event selection after all steps except b-jet selection
    results.steps["all_but_bjet"] = (
        (results.steps.Jet | results.steps.Boosted) &
        results.steps.Lepton &
        results.steps.Trigger
    )

    # combined event selection after all steps
    # NOTE: we only apply the b-tagging step when no AK8 Jet is present; if some event with AK8 jet
    #       gets categorized into the resolved category, we might need to cut again on the number of b-jets
    results.main["event"] = (
        results.steps.all_but_bjet &
        (results.steps.Bjet | results.steps.Boosted)
    )

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add cutflow features
    if self.config_inst.x("do_cutflow_features", False):
        events = self[cutflow_features](events, results=results, **kwargs)
    """
    # produce event weights
    if self.dataset_inst.is_mc:
        events = self[event_weights_to_normalize](events, results=results, **kwargs)

    # increment stats
    self[increment_stats](events, results, stats, **kwargs)
    """
    return events, results


@default.init
def default_init(self: Selector) -> None:
    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return

    self.uses |= {event_weights_to_normalize}
    self.produces |= {event_weights_to_normalize}


@selector(
    uses={
        default, "mc_weight",  # mc_weight should be included from default
        gen_hbw_decay_products, gen_hbw_decay_features, gen_hbw_matching,
    },
    produces={
        category_ids, process_ids, increment_stats, "mc_weight",
        gen_hbw_decay_products, gen_hbw_decay_features, gen_hbw_matching,
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

    # match genparticles with reco objects
    events = self[gen_hbw_matching](events, results, **kwargs)

    return events, results
