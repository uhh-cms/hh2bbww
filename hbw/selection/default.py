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
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.categories import category_ids
from columnflow.production.processes import process_ids

from hbw.production.weights import event_weights_to_normalize, large_weights_killer
from hbw.production.gen_hbw_decay import gen_hbw_decay_products
from hbw.selection.stats import increment_stats_old, hbw_increment_stats
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
    uses={
        # jet_selection,
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB",
    },
    exposed=True,
)
def vbf_jet_selection(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:

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
        # jet_selection, lepton_selection,
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.jetId",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass",
        "FatJet.msoftdrop", "FatJet.jetId", "FatJet.subJetIdx1", "FatJet.subJetIdx2",
        "FatJet.tau1", "FatJet.tau2",
    },
    produces={"cutflow.n_hbbjet"},
    exposed=True,
)
def boosted_jet_selection(
    self: Selector,
    events: ak.Array,
    lepton_results: SelectionResult,
    jet_results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # HH -> bbWW(qqlnu) boosted selection

    # leptons (TODO: use fakeable leptons here)
    electron = events.Electron[lepton_results.objects.Electron.Electron]
    muon = events.Muon[lepton_results.objects.Muon.Muon]
    ak4_jets = events.Jet[jet_results.objects.Jet.Jet]

    events = set_ak_column(events, "FatJet.local_index", ak.local_index(events.FatJet))

    # baseline fatjet selection
    fatjet_mask = (
        (events.FatJet.pt > 200) &
        (abs(events.FatJet.eta) < 2.4) &
        (events.FatJet.jetId == 6) &
        (ak.all(events.FatJet.metric_table(electron) > 0.8, axis=2)) &
        (ak.all(events.FatJet.metric_table(muon) > 0.8, axis=2))
    )

    # H->bb fatjet definition based on Aachen analysis
    hbbJet_mask = (
        fatjet_mask &
        (events.FatJet.msoftdrop > 30) &
        (events.FatJet.msoftdrop < 210) &
        (events.FatJet.subJetIdx1 >= 0) &
        (events.FatJet.subJetIdx2 >= 0) &
        (events.FatJet.subJetIdx1 < ak.num(events.Jet)) &
        (events.FatJet.subJetIdx2 < ak.num(events.Jet)) &
        (events.FatJet.tau2 / events.FatJet.tau1 < 0.75)
    )

    # create temporary object with fatjet mask applied and get the subjets
    hbbjets = events.FatJet[hbbJet_mask]
    subjet1 = events.Jet[hbbjets.subJetIdx1]
    subjet2 = events.Jet[hbbjets.subJetIdx2]

    # requirements on H->bb subjets (without b-tagging)
    subjets_mask_no_bjet = (
        (abs(subjet1.eta) < 2.4) & (abs(subjet2.eta) < 2.4) &
        (subjet1.pt > 20) & (subjet2.pt > 20) &
        ((subjet1.pt > 30) | (subjet2.pt > 30))
    )
    hbbjets_no_bjet = hbbjets[subjets_mask_no_bjet]

    boosted_sel_no_bjet = (
        (ak.num(hbbjets_no_bjet, axis=1) >= 1) &
        (ak.sum(ak.any(ak4_jets.metric_table(hbbjets_no_bjet) > 1.2, axis=2), axis=1) > 0)
    )

    # requirements on H->bb subjets (with b-tagging)
    wp_med = self.config_inst.x.btag_working_points.deepjet.medium
    subjets_mask = (
        (abs(subjet1.eta) < 2.4) & (abs(subjet2.eta) < 2.4) &
        (subjet1.pt > 20) & (subjet2.pt > 20) &
        (
            ((subjet1.pt > 30) & (subjet1.btagDeepFlavB > wp_med)) |
            ((subjet2.pt > 30) & (subjet2.btagDeepFlavB > wp_med))
        )
    )

    # apply subjets requirements on hbbjets and pt-sort
    hbbjets = hbbjets[subjets_mask]
    hbbjets = hbbjets[ak.argsort(hbbjets.pt, ascending=False)]

    # number of hbbjets fulfilling all criteria
    events = set_ak_column(events, "cutflow.n_hbbjet", ak.num(hbbjets, axis=1))
    hbbjet_sel = events.cutflow.n_hbbjet >= 1

    # require at least one ak4 jet not included in the subjets of one of the hbbjets
    ak4_jets = ak4_jets[ak.any(ak4_jets.metric_table(hbbjets) > 1.2, axis=2)]

    # NOTE: we might want to remove these ak4 jets from our list of jets
    ak4_jet_sel = ak.num(ak4_jets, axis=1) > 0

    boosted_sel = ak4_jet_sel & hbbjet_sel

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "HbbJet": hbbjet_sel,
            "Boosted": boosted_sel,
            "Boosted_no_bjet": boosted_sel_no_bjet,  # TODO check if correct
        },
        objects={
            "FatJet": {
                # NOTE: we might want to relax requirements here and only apply them later
                #       to simplify optimization studies
                "FatJet": masked_sorted_indices(fatjet_mask, events.FatJet.pt),
                "HbbJet": hbbjets.local_index,
            },
        },
    )


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB", "Jet.jetId"},
    produces={"cutflow.n_jet", "cutflow.n_deepjet_med"},
    exposed=True,
)
def jet_selection(
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
def lepton_selection(
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


@lepton_selection.init
def lepton_selection_init(self: Selector) -> None:
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
        jet_selection, vbf_jet_selection, lepton_selection,
        category_ids, process_ids, hbw_increment_stats, attach_coffea_behavior,
        mc_weight, large_weights_killer,
    },
    produces={
        boosted_jet_selection,
        jet_selection, vbf_jet_selection, lepton_selection,
        category_ids, process_ids, hbw_increment_stats, attach_coffea_behavior,
        mc_weight, large_weights_killer,
    },
    exposed=True,
)
def default(
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
    events, lepton_results = self[lepton_selection](events, stats, **kwargs)
    results += lepton_results

    # jet selection
    events, jet_results = self[jet_selection](events, lepton_results, stats, **kwargs)
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

    return events, results


lep_15 = default.derive("lep_15", cls_dict={"ele_pt": 15, "mu_pt": 15})
lep_27 = default.derive("lep_27", cls_dict={"ele_pt": 27, "mu_pt": 27})


@default.init
def default_init(self: Selector) -> None:
    if self.config_inst.x("do_cutflow_features", False):
        self.uses.add(cutflow_features)
        self.produces.add(cutflow_features)

    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return

    self.uses.add(event_weights_to_normalize)
    self.produces.add(event_weights_to_normalize)


@selector(
    uses={
        default, "mc_weight",  # mc_weight should be included from default
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
    events, results = self[default](events, stats, **kwargs)

    # extract relevant gen HH decay products
    events = self[gen_hbw_decay_products](events, **kwargs)

    # produce relevant columns
    events = self[gen_hbw_decay_features](events, **kwargs)

    # match genparticles with reco objects
    events = self[gen_hbw_matching](events, results, **kwargs)

    return events, results
