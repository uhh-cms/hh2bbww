# coding: utf-8

"""
Selection modules for HH -> bbWW(qqlnu).
"""

from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.categories import category_ids
from columnflow.production.processes import process_ids

from hbw.selection.common import masked_sorted_indices, boosted_jet_selection, vbf_jet_selection
from hbw.production.weights import event_weights_to_normalize, large_weights_killer
from hbw.selection.stats import hbw_increment_stats
from hbw.selection.cutflow_features import cutflow_features

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB", "Jet.jetId"},
    produces={"cutflow.n_jet", "cutflow.n_deepjet_med"},
    exposed=True,
)
def sl_jet_selection(
    self: Selector,
    events: ak.Array,
    lepton_results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # HH -> bbWW(qqlnu) jet selection
    # - require at least 3 jets with pt>30, eta<2.4
    # - require at least 1 jet with pt>30, eta<2.4, b-score>0.3040 (Medium WP)

    # assign local index to all Jets
    events = set_ak_column(events, "local_index", ak.local_index(events.Jet))

    # jets
    jet_mask_loose = (events.Jet.pt > 5) & abs(events.Jet.eta < 2.4)
    jet_mask = (
        (events.Jet.pt > 25) & (abs(events.Jet.eta) < 2.4) & (events.Jet.jetId == 6) &
        ak.all(events.Jet.metric_table(lepton_results.x.lepton) > 0.4, axis=2)
    )
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


@selector(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Electron.cutBased", "Electron.mvaFall17V2Iso_WP80",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "Muon.tightId", "Muon.looseId", "Muon.pfRelIso04_all",
        "Tau.pt", "Tau.eta", "Tau.idDeepTau2017v2p1VSe",
        "Tau.idDeepTau2017v2p1VSmu", "Tau.idDeepTau2017v2p1VSjet",
    },
    e_pt=None, mu_pt=None, e_trigger=None, mu_trigger=None,
)
def sl_lepton_selection(
        self: Selector,
        events: ak.Array,
        stats: defaultdict,
        **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # HH -> bbWW(qqlnu) lepton selection
    # - require exactly 1 lepton (e or mu) with pt_e>36 / pt_mu>28, eta<2.4 and tight ID
    # - veto additional leptons (TODO define exact cuts)
    # - require that events are triggered by SingleMu or SingleEle trigger

    # Veto Lepton masks (TODO define exact cuts)
    e_mask_veto = (events.Electron.pt > 20) & (abs(events.Electron.eta) < 2.4) & (events.Electron.cutBased >= 1)
    mu_mask_veto = (events.Muon.pt > 20) & (abs(events.Muon.eta) < 2.4) & (events.Muon.looseId)
    tau_mask_veto = (
        (abs(events.Tau.eta) < 2.3) &
        # (abs(events.Tau.dz) < 0.2) &
        (events.Tau.pt > 20.0) &
        (events.Tau.idDeepTau2017v2p1VSe >= 4) &  # 4: VLoose
        (events.Tau.idDeepTau2017v2p1VSmu >= 8) &  # 8: Tight
        (events.Tau.idDeepTau2017v2p1VSjet >= 2)  # 2: VVLoose
    )

    lep_veto_sel = ak.sum(e_mask_veto, axis=-1) + ak.sum(mu_mask_veto, axis=-1) <= 1
    tau_veto_sel = ak.sum(tau_mask_veto, axis=-1) == 0

    # Lepton definition for this analysis
    e_mask = (
        (events.Electron.pt > self.e_pt) &
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

    lep_sel = ak.sum(e_mask, axis=-1) + ak.sum(mu_mask, axis=-1) >= 1
    e_sel = (ak.sum(e_mask, axis=-1) == 1) & (ak.sum(mu_mask, axis=-1) == 0)
    mu_sel = (ak.sum(e_mask, axis=-1) == 0) & (ak.sum(mu_mask, axis=-1) == 1)

    # dummy mask
    ones = ak.ones_like(lep_sel)

    # individual trigger
    mu_trigger_sel = ones if not self.mu_trigger else events.HLT[self.mu_trigger]
    e_trigger_sel = ones if not self.mu_trigger else events.HLT[self.e_trigger]

    # combined trigger
    trigger_sel = mu_trigger_sel | e_trigger_sel

    # combined trigger, removing events where trigger and lepton types do not match
    # TODO: compare trigger object and lepton
    trigger_lep_crosscheck = (
        (e_trigger_sel & e_sel) |
        (mu_trigger_sel & mu_sel)
    )

    e_indices = masked_sorted_indices(e_mask, events.Electron.pt)
    mu_indices = masked_sorted_indices(mu_mask, events.Muon.pt)
    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "Muon": mu_sel, "Electron": e_sel,
            "Lepton": lep_sel, "VetoLepton": lep_veto_sel,
            "VetoTau": tau_veto_sel,
            "MuTrigger": mu_trigger_sel, "EleTrigger": e_trigger_sel,
            "Trigger": trigger_sel, "TriggerAndLep": trigger_lep_crosscheck,
        },
        objects={
            "Electron": {
                "VetoElectron": masked_sorted_indices(e_mask_veto, events.Electron.pt),
                "Electron": e_indices,
            },
            "Muon": {
                "VetoMuon": masked_sorted_indices(mu_mask_veto, events.Muon.pt),
                "Muon": mu_indices,
            },
            "Tau": {"VetoTau": masked_sorted_indices(tau_mask_veto, events.Tau.pt)},
        },
        aux={
            # save the selected lepton for the duration of the selection
            # multiplication of a coffea particle with 1 yields the lorentz vector
            "lepton": ak.concatenate(
                [
                    events.Electron[e_indices] * 1,
                    events.Muon[mu_indices] * 1,
                ],
                axis=1,
            ),
        },
    )


@sl_lepton_selection.init
def sl_lepton_selection_init(self: Selector) -> None:
    year = self.config_inst.campaign.x.year

    # NOTE: the none will not be overwritten later when doing this...
    # self.mu_trigger = self.e_trigger = None

    # Lepton pt thresholds (if not set manually) based on year (1 pt above trigger threshold)
    # When lepton pt thresholds are set manually, don't use any trigger
    if not self.e_pt:
        self.e_pt = {2016: 28, 2017: 36, 2018: 33}[year]

        # Trigger choice based on year of data-taking (for now: only single trigger)
        self.e_trigger = {
            2016: "Ele27_WPTight_Gsf",  # or "HLT_Ele115_CaloIdVT_GsfTrkIdT", "HLT_Photon175")
            2017: "Ele35_WPTight_Gsf",  # or "HLT_Ele115_CaloIdVT_GsfTrkIdT", "HLT_Photon200")
            2018: "Ele32_WPTight_Gsf",  # or "HLT_Ele115_CaloIdVT_GsfTrkIdT", "HLT_Photon200")
        }[year]
        self.uses.add(f"HLT.{self.e_trigger}")
    if not self.mu_pt:
        self.mu_pt = {2016: 25, 2017: 28, 2018: 25}[year]

        # Trigger choice based on year of data-taking (for now: only single trigger)
        self.mu_trigger = {
            2016: "IsoMu24",  # or "IsoTkMu27")
            2017: "IsoMu27",
            2018: "IsoMu24",
        }[year]
        self.uses.add(f"HLT.{self.mu_trigger}")


@selector(
    uses={
        boosted_jet_selection,
        sl_jet_selection, vbf_jet_selection, sl_lepton_selection,
        category_ids, process_ids, hbw_increment_stats, attach_coffea_behavior,
        mc_weight, large_weights_killer,
    },
    produces={
        boosted_jet_selection,
        sl_jet_selection, vbf_jet_selection, sl_lepton_selection,
        category_ids, process_ids, hbw_increment_stats, attach_coffea_behavior,
        mc_weight, large_weights_killer,
    },
    exposed=True,
)
def sl(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # mc weight
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
        events = self[large_weights_killer](events, **kwargs)

    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # lepton selection
    events, lepton_results = self[sl_lepton_selection](events, stats, **kwargs)
    results += lepton_results

    # jet selection
    events, jet_results = self[sl_jet_selection](events, lepton_results, stats, **kwargs)
    results += jet_results

    # boosted selection
    events, boosted_results = self[boosted_jet_selection](events, lepton_results, jet_results, stats, **kwargs)
    results += boosted_results

    # vbf-jet selection
    events, vbf_jet_results = self[vbf_jet_selection](events, results, stats, **kwargs)
    results += vbf_jet_results

    results.steps["ResolvedOrBoosted"] = (
        (results.steps.Jet & results.steps.Bjet) | results.steps.Boosted
    )

    # combined event selection after all steps except b-jet selection
    results.steps["all_but_bjet"] = (
        # NOTE: the boosted selection actually includes a b-jet selection...
        (results.steps.Jet | results.steps.Boosted_no_bjet) &
        results.steps.Lepton &
        results.steps.VetoLepton &
        results.steps.VetoTau &
        results.steps.Trigger &
        results.steps.TriggerAndLep
    )

    # combined event selection after all steps
    # NOTE: we only apply the b-tagging step when no AK8 Jet is present; if some event with AK8 jet
    #       gets categorized into the resolved category, we might need to cut again on the number of b-jets
    results.main["event"] = (
        results.steps.all_but_bjet &
        ((results.steps.Jet & results.steps.Bjet) | results.steps.Boosted)
    )

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add cutflow features
    if self.config_inst.x("do_cutflow_features", False):
        events = self[cutflow_features](events, results=results, **kwargs)

    # produce event weights
    if self.dataset_inst.is_mc:
        events = self[event_weights_to_normalize](events, results=results, **kwargs)

    # increment stats
    self[hbw_increment_stats](events, results, stats, **kwargs)

    print(f"Fraction of negative weights: {(100 * stats['num_negative_weights'] / stats['num_events']):.2f}%")

    return events, results


lep_15 = sl.derive("sl_lep_15", cls_dict={"ele_pt": 15, "mu_pt": 15})
lep_27 = sl.derive("sl_lep_27", cls_dict={"ele_pt": 27, "mu_pt": 27})


@sl.init
def sl_init(self: Selector) -> None:
    if self.config_inst.x("do_cutflow_features", False):
        self.uses.add(cutflow_features)
        self.produces.add(cutflow_features)

    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return

    self.uses.add(event_weights_to_normalize)
    self.produces.add(event_weights_to_normalize)


from hbw.production.gen_hbw_decay import gen_hbw_decay_products
from hbw.selection.gen_hbw_features import gen_hbw_decay_features, gen_hbw_matching


# TODO: implement the gen modules in sl (or one of the common modules)
@selector(
    uses={
        sl, "mc_weight",  # mc_weight should be included from default
        gen_hbw_decay_products, gen_hbw_decay_features, gen_hbw_matching,
    },
    produces={
        category_ids, process_ids, hbw_increment_stats, "mc_weight",
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
    events, results = self[sl](events, stats, **kwargs)

    # extract relevant gen HH decay products
    events = self[gen_hbw_decay_products](events, **kwargs)

    # produce relevant columns
    events = self[gen_hbw_decay_features](events, **kwargs)

    # match genparticles with reco objects
    events = self[gen_hbw_matching](events, results, **kwargs)

    return events, results
