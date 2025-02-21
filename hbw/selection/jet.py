# coding: utf-8

"""
Selection modules for HH(bbWW) lepton selections.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Tuple

import law
import order as od

from columnflow.util import maybe_import, DotDict
from columnflow.columnar_util import set_ak_column, optional_column as optional
from columnflow.selection import Selector, SelectionResult, selector

from hbw.selection.common import masked_sorted_indices
from hbw.util import call_once_on_config
from hbw.production.jets import jetId

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@selector(
    uses={jetId, "Jet.{pt,eta,phi,mass,jetId}", optional("Jet.puId")},
    exposed=True,
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    lepton_results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """
    Central definition of Jets in HH(bbWW)

    Requires the following attributes in the config aux:
    - jet_pt: minimum pt for jets (defaults to 25 GeV)
    - b_tagger: b-tagger to use
    - btag_column: column name for b-tagging score
    - btag_wp: b-tagging working point (either "loose", "medium", or "tight")
    - btag_wp_score: score corresponding to b-tagging wp and b-score
    """
    steps = DotDict()

    # assign local index to all Jets
    events = set_ak_column(events, "local_index", ak.local_index(events.Jet))

    # get correct jet working points (Jet.TightId and Jet.TightLepVeto)
    events = self[jetId](events, **kwargs)

    # default jet definition
    jet_mask_loose = (
        (events.Jet.pt >= self.jet_pt) &
        (abs(events.Jet.eta) <= 2.4) &
        (events.Jet.TightLepVeto)
    )

    electron = events.Electron[lepton_results.objects.Electron.Electron]
    muon = events.Muon[lepton_results.objects.Muon.Muon]

    jet_mask = (
        (events.Jet.pt >= self.jet_pt) &
        (abs(events.Jet.eta) <= 2.4) &
        (events.Jet.TightLepVeto) &
        # ak.all(events.Jet.metric_table(lepton_results.x.lepton) > 0.4, axis=2)
        ak.all(events.Jet.metric_table(electron) > 0.4, axis=2) &
        ak.all(events.Jet.metric_table(muon) > 0.4, axis=2)
    )

    # apply loose Jet puId to jets with pt below 50 GeV (not in Run3 samples so skip this for now)
    if self.config_inst.x.run == 2:
        jet_pu_mask = (events.Jet.puId >= 4) | (events.Jet.pt > 50)
        jet_mask = jet_mask & jet_pu_mask

    # get the jet indices for pt-sorting of jets
    jet_indices = masked_sorted_indices(jet_mask, events.Jet.pt)

    steps["nJetReco1"] = ak.num(events.Jet) >= 1
    steps["nJetLoose1"] = ak.sum(jet_mask_loose, axis=1) >= 1

    # add jet steps
    events = set_ak_column(events, "cutflow.n_jet", ak.sum(jet_mask, axis=1))
    steps["nJet1"] = events.cutflow.n_jet >= 1
    steps["nJet2"] = events.cutflow.n_jet >= 2
    steps["nJet3"] = events.cutflow.n_jet >= 3
    steps["nJet4"] = events.cutflow.n_jet >= 4
    if self.config_inst.x("n_jet", 0) > 4:
        steps[f"nJet{self.config_inst.x.n_jet}"] = events.cutflow.n_jet >= self.config_inst.x.n_jet

    # define btag mask
    btag_column = self.config_inst.x.btag_column
    b_score = events.Jet[btag_column]
    # sometimes, jet b-score is nan, so fill it with 0
    if ak.any(np.isnan(b_score)):
        b_score = ak.fill_none(ak.nan_to_none(b_score), 0)
    btag_mask = (jet_mask) & (b_score >= self.config_inst.x.btag_wp_score)

    # add btag steps
    events = set_ak_column(events, "cutflow.n_btag", ak.sum(btag_mask, axis=1))
    steps["nBjet1"] = events.cutflow.n_btag >= 1
    steps["nBjet2"] = events.cutflow.n_btag >= 2
    if self.config_inst.x("n_btag", 0) > 2:
        steps[f"nBjet{self.config_inst.x.n_btag}"] = events.cutflow.n_btag >= self.config_inst.x.n_btag

    # define b-jets as the two b-score leading jets, b-score sorted
    bjet_indices = masked_sorted_indices(jet_mask, b_score)[:, :2]

    # define lightjets as all non b-jets, pt-sorted
    b_idx = ak.fill_none(ak.pad_none(bjet_indices, 2), -1)
    lightjet_indices = jet_indices[(jet_indices != b_idx[:, 0]) & (jet_indices != b_idx[:, 1])]

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps=steps,
        objects={
            "Jet": {
                "Jet": jet_indices,
                "Bjet": bjet_indices,
                "Lightjet": lightjet_indices,
            },
        },
        aux={
            "jet_mask": jet_mask,
            "n_central_jets": ak.num(jet_indices),
            "ht": ak.sum(events.Jet.pt[jet_mask], axis=1),
        },
    )


@jet_selection.init
def jet_selection_init(self: Selector) -> None:
    # configuration of defaults
    self.jet_pt = self.config_inst.x("jet_pt", 25)

    # Add shift dependencies
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag(("jec", "jer"))
    }

    # add btag requirement
    self.uses.add(f"Jet.{self.config_inst.x.btag_column}")

    # update selector steps labels
    self.config_inst.x.selector_step_labels = self.config_inst.x("selector_step_labels", {})
    self.config_inst.x.selector_step_labels.update({
        "nJet1": r"$N_{jets}^{AK4} \geq 1$",
        "nJet2": r"$N_{jets}^{AK4} \geq 2$",
        "nJet3": r"$N_{jets}^{AK4} \geq 3$",
        "nJet4": r"$N_{jets}^{AK4} \geq 4$",
        "nBjet1": r"$N_{jets}^{BTag} \geq 1$",
        "nBjet2": r"$N_{jets}^{BTag} \geq 2$",
    })

    if self.config_inst.x("do_cutflow_features", False):
        # add cutflow features to *produces* only when requested
        self.produces.add("cutflow.n_jet")
        self.produces.add("cutflow.n_btag")

        @call_once_on_config
        def add_jet_cutflow_variables(config: od.Config):
            config.add_variable(
                name="cf_n_jet",
                expression="cutflow.n_jet",
                binning=(7, -0.5, 6.5),
                x_title="Number of jets",
                discrete_x=True,
            )
            config.add_variable(
                name="cf_n_btag",
                expression="cutflow.n_btag",
                binning=(7, -0.5, 6.5),
                x_title=f"Number of b-tagged jets ({self.config_inst.x.b_tagger}, {self.config_inst.x.btag_wp} WP)",
                discrete_x=True,
            )

        add_jet_cutflow_variables(self.config_inst)


@selector(
    uses={jet_selection, "Jet.{pt,eta,phi,mass}"},
    exposed=False,
)
def vbf_jet_selection(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:

    # assign local index to all Jets
    events = set_ak_column(events, "Jet.local_index", ak.local_index(events.Jet.pt))

    # default requirements for vbf jets (pt, eta and no H->bb jet)
    # NOTE: we might also want to remove the two H->jj jet candidates
    # TODO: how to get the object mask from the object indices in a more convenient way?
    b_indices = ak.where(
        results.steps.HbbJet,
        ak.fill_none(ak.pad_none(results.objects.Jet.HbbSubJet, 2), -1),
        ak.fill_none(ak.pad_none(results.objects.Jet.Bjet, 2), -1),
    )
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


@vbf_jet_selection.init
def vbf_jet_selection_init(self: Selector) -> None:
    # Add shift dependencies
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag(("jec", "jer"))
    }

    # update selector step labels
    self.config_inst.x.selector_step_labels = self.config_inst.x("selector_step_labels", {})
    self.config_inst.x.selector_step_labels.update({
        "VBFJetPair": r"$N_{VBFJetPair}^{AK4} \geq 1$",
    })


@selector(
    uses={
        jet_selection,
        "{Electron,Muon,Jet,FatJet}.{pt,eta,phi,mass}",
        "Jet.{jetId}",
        "FatJet.{msoftdrop,jetId,subJetIdx1,subJetIdx2,tau1,tau2}",
    },
    exposed=False,
    single_lepton_selector=True,
)
def sl_boosted_jet_selection(
    self: Selector,
    events: ak.Array,
    lepton_results: SelectionResult,
    jet_results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """
    HH -> bbWW(qqlnu) boosted selection
    TODO: separate common parts from SL dependant
    """
    # assign local index to all Jets
    events = set_ak_column(events, "Jet.local_index", ak.local_index(events.Jet, axis=1))
    events = set_ak_column(events, "FatJet.local_index", ak.local_index(events.FatJet, axis=1))

    # get leptons and jets with object masks applied
    electron = events.Electron[lepton_results.objects.Electron.Electron]
    muon = events.Muon[lepton_results.objects.Muon.Muon]
    ak4_jets = events.Jet[jet_results.objects.Jet.Jet]

    # get separation info between FatJets and AK4 Jets
    dr_fatjet_ak4 = events.FatJet.metric_table(ak4_jets)
    events = set_ak_column(events, "cutflow.FatJet.n_subjets", ak.sum(dr_fatjet_ak4 < 0.8, axis=2))
    events = set_ak_column(events, "cutflow.FatJet.n_separated_jets", ak.sum(dr_fatjet_ak4 > 1.2, axis=2))
    events = set_ak_column(events, "cutflow.FatJet.max_dr_ak4", ak.max(dr_fatjet_ak4, axis=2))

    # baseline fatjet selection
    fatjet_mask = (
        (events.FatJet.pt > 170) &
        (abs(events.FatJet.eta) < 2.4) &
        (events.FatJet.jetId >= 6) &
        (ak.all(events.FatJet.metric_table(electron) > 0.8, axis=2)) &
        (ak.all(events.FatJet.metric_table(muon) > 0.8, axis=2))
    )
    events = set_ak_column(events, "cutflow.n_fatjet", ak.sum(fatjet_mask, axis=1))

    # H->bb fatjet definition based on Aachen analysis
    hbbJet_mask = (
        fatjet_mask &
        (events.FatJet.pt > 200) &
        (events.FatJet.msoftdrop > 30) &
        (events.FatJet.msoftdrop < 210) &
        (events.FatJet.subJetIdx1 >= 0) &
        (events.FatJet.subJetIdx2 >= 0) &
        (events.FatJet.subJetIdx1 < ak.num(events.Jet)) &
        (events.FatJet.subJetIdx2 < ak.num(events.Jet)) &
        (events.FatJet.tau2 / events.FatJet.tau1 < 0.75)
    )
    # for the SL analysis, we additionally require one AK4 jet that is separated with dR > 1.2
    # (as part of the HbbJet definition)
    if self.single_lepton_selector:
        hbbJet_mask = (
            hbbJet_mask &
            (events.cutflow.FatJet.n_separated_jets >= 1)
        )

    # create temporary object with fatjet mask applied and get the subjets
    hbbjets = events.FatJet[hbbJet_mask]
    # TODO: subJetIdx1 only gives indices for the SubJet collection...
    subjet1 = events.Jet[hbbjets.subJetIdx1]
    subjet2 = events.Jet[hbbjets.subJetIdx2]

    # requirements on H->bb subjets (without b-tagging)
    subjets_mask_no_bjet = (
        (abs(subjet1.eta) < 2.4) & (abs(subjet2.eta) < 2.4) &
        (subjet1.pt > 20) & (subjet2.pt > 20) &
        ((subjet1.pt > 30) | (subjet2.pt > 30))
    )
    hbbjets_no_bjet = hbbjets[subjets_mask_no_bjet]
    hbbjet_sel_no_bjet = ak.num(hbbjets_no_bjet, axis=1) >= 1

    # get the correct btag WPs and column from the config (as setup by jet_selection)
    btag_wp_score = self.config_inst.x.btag_wp_score
    btag_column = self.config_inst.x.btag_column

    subjets_mask = (
        (abs(subjet1.eta) < 2.4) & (abs(subjet2.eta) < 2.4) &
        (subjet1.pt > 20) & (subjet2.pt > 20) &
        (
            ((subjet1.pt > 30) & (subjet1[btag_column] > btag_wp_score)) |
            ((subjet2.pt > 30) & (subjet2[btag_column] > btag_wp_score))
        )
    )

    # apply subjets requirements on hbbjets and pt-sort
    hbbjets = hbbjets[subjets_mask]
    hbbjets = hbbjets[ak.argsort(hbbjets.pt, ascending=False)]
    events = set_ak_column(events, "HbbJet", hbbjets)

    # number of hbbjets fulfilling all criteria
    events = set_ak_column(events, "cutflow.n_hbbjet", ak.num(hbbjets, axis=1))
    hbbjet_sel = events.cutflow.n_hbbjet >= 1

    # define HbbSubJet collection (TODO: pt-sort or b-score sort)
    # TODO: when we get multiple HbbJets, we also define more than 2 hbbSubJets
    hbbSubJet_indices = ak.concatenate([hbbjets.subJetIdx1, hbbjets.subJetIdx2], axis=1)

    # build and return selection results plus new columns
    boosted_results = SelectionResult(
        steps={
            "HbbJet_no_bjet": hbbjet_sel_no_bjet,
            "HbbJet": hbbjet_sel,
        },
        objects={
            "FatJet": {
                "FatJet": masked_sorted_indices(fatjet_mask, events.FatJet.pt),
                "HbbJet": hbbjets.local_index,
            },
            "Jet": {"HbbSubJet": hbbSubJet_indices},
        },
    )
    return events, boosted_results


@sl_boosted_jet_selection.init
def sl_boosted_jet_selection_init(self: Selector) -> None:
    # Add shift dependencies
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag(("jec", "jer"))
    }

    # update selector step labels
    self.config_inst.x.selector_step_labels = self.config_inst.x("selector_step_labels", {})
    self.config_inst.x.selector_step_labels.update({
        "HbbJet": r"$N_{H \rightarrow bb}^{AK8} \geq 1$",
    })

    # add produced variables to *produces* only when requested
    if self.config_inst.x("do_cutflow_features", False):
        self.produces |= {"cutflow.{n_fatjet,n_hbbjet}", "cutflow.FatJet.{n_subjets,n_separated_jets,max_dr_ak4}"}


# boosted selection for the DL channel (only one parameter needs to be changed)
dl_boosted_jet_selection = sl_boosted_jet_selection.derive(
    "dl_boosted_jet_selection",
    cls_dict={"single_lepton_selector": False},
)
