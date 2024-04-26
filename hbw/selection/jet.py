# coding: utf-8

"""
Selection modules for HH(bbWW) lepton selections.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Tuple

import law
import order as od

from cmsdb.constants import m_z
from columnflow.util import maybe_import, DotDict
from columnflow.columnar_util import set_ak_column, optional_column as optional
from columnflow.selection import Selector, SelectionResult, selector

from hbw.selection.common import masked_sorted_indices
from hbw.util import four_vec, call_once_on_config

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@selector(
    uses=four_vec("Jet", {"jetId"}) | {optional("Jet.puId")},
    exposed=True,
    b_tagger="particlenet",
    btag_wp="medium",
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
    """

    steps = DotDict()

    # assign local index to all Jets
    events = set_ak_column(events, "local_index", ak.local_index(events.Jet))


    # default jet definition
    jet_mask_loose = (
        (events.Jet.pt >= 25) &
        (abs(events.Jet.eta) <= 2.4) &
        (events.Jet.jetId >= 2)  # 1: loose, 2: tight, 4: isolated, 6: tight+isolated
    )

    electron = events.Electron[lepton_results.objects.Electron.FakeableElectron]
    muon = events.Muon[lepton_results.objects.Muon.FakeableMuon]

    jet_mask = (
        (events.Jet.pt >= 25) &
        (abs(events.Jet.eta) <= 2.4) &
        (events.Jet.jetId >= 2) &  # 1: loose, 2: tight, 4: isolated, 6: tight+isolated
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
        },
    )


@jet_selection.init
def jet_selection_init(self: Selector) -> None:
    # Add shift dependencies
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag(("jec", "jer"))
    }

    # set the main b_tagger + working point as defined from the selector
    self.config_inst.x.b_tagger = self.b_tagger
    self.config_inst.x.btag_wp = self.btag_wp
    self.config_inst.x.btag_wp_score = (
        self.config_inst.x.btag_working_points[self.config_inst.x.b_tagger][self.config_inst.x.btag_wp]
    )
    if self.config_inst.x.b_tagger == "deepjet":
        self.config_inst.x.btag_sf = ("deepJet_shape", self.config_inst.x.btag_sf_jec_sources, "btagDeepFlavB")
    elif self.config_inst.x.b_tagger == "particlenet":
        self.config_inst.x.btag_sf = ("particleNet_shape", self.config_inst.x.btag_sf_jec_sources, "btagPNetB")

    self.btag_column = self.config_inst.x.btag_column = {
        "deepjet": "btagDeepFlavB",
        "particlenet": "btagPNetB",
    }.get(self.config_inst.x.b_tagger, self.config_inst.x.b_tagger)

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
                x_title=f"Number of b-tagged jets ({self.b_tagger}, {self.btag_wp} WP)",
                discrete_x=True,
            )

        add_jet_cutflow_variables(self.config_inst)


@selector(
    uses=(
        four_vec("Electron", {
            "dxy", "dz", "miniPFRelIso_all", "sip3d", "cutBased", "lostHits",  # Electron Preselection
            "mvaIso_WP90",  # used as replacement for "mvaNoIso_WPL" in Preselection
            "mvaTTH", "jetRelIso",  # cone-pt
            "deltaEtaSC", "sieie", "hoe", "eInvMinusPInv", "convVeto", "jetIdx",  # Fakeable Electron
        }) |
        four_vec("Muon", {
            "dxy", "dz", "miniPFRelIso_all", "sip3d", "looseId",  # Muon Preselection
            "mediumId", "mvaTTH", "jetRelIso",  # cone-pt
            "jetIdx",  # Fakeable Muon
        }) |
        four_vec("Tau", {
            "dz", "idDeepTau2017v2p1VSe", "idDeepTau2017v2p1VSmu", "idDeepTau2017v2p1VSjet",
        }) | {
            jet_selection,  # the jet_selection init needs to be run to set the correct b_tagger
        }
    ),
    produces={
        "Muon.cone_pt", "Muon.is_tight",
        "Electron.cone_pt", "Muon.is_tight",
    },
)
def lepton_definition(
        self: Selector,
        events: ak.Array,
        stats: defaultdict,
        synchronization: bool = True,
        **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """
    Central definition of Leptons in HH(bbWW)
    """
    # initialize dicts for the selection steps
    steps = DotDict()

    # reconstruct relevant variables
    events = set_ak_column(events, "Electron.cone_pt", ak.where(
        events.Electron.mvaTTH >= 0.30,
        events.Electron.pt,
        0.9 * events.Electron.pt * (1.0 + events.Electron.jetRelIso),
    ))
    events = set_ak_column(events, "Muon.cone_pt", ak.where(
        (events.Muon.mediumId & (events.Muon.mvaTTH >= 0.50)),
        events.Muon.pt,
        0.9 * events.Muon.pt * (1.0 + events.Muon.jetRelIso),
    ))

    electron = events.Electron
    muon = events.Muon

    # preselection masks
    e_mask_loose = (
        # (electron.cone_pt >= 7) &
        (electron.pt >= 7) &
        (abs(electron.eta) <= 2.5) &
        (abs(electron.dxy) <= 0.05) &
        (abs(electron.dz) <= 0.1) &
        (electron.miniPFRelIso_all <= 0.4) &
        (electron.sip3d <= 8) &
        (electron.mvaIso_WP90) &  # TODO: replace when possible
        # (electron.mvaNoIso_WPL) &  # missing
        (electron.lostHits <= 1)
    )
    mu_mask_loose = (
        # (muon.cone_pt >= 5) &
        (muon.pt >= 5) &
        (abs(muon.eta) <= 2.4) &
        (abs(muon.dxy) <= 0.05) &
        (abs(muon.dz) <= 0.1) &
        (muon.miniPFRelIso_all <= 0.4) &
        (muon.sip3d <= 8) &
        (muon.looseId)
    )

    # lepton invariant mass cuts
    loose_leptons = ak.concatenate([
        events.Electron[e_mask_loose] * 1,
        events.Muon[mu_mask_loose] * 1,
    ], axis=1)

    lepton_pairs = ak.combinations(loose_leptons, 2)
    l1, l2 = ak.unzip(lepton_pairs)
    lepton_pairs["m_inv"] = (l1 + l2).mass

    steps["ll_lowmass_veto"] = ~ak.any((lepton_pairs.m_inv < 12), axis=1)
    steps["ll_zmass_veto"] = ~ak.any((abs(lepton_pairs.m_inv - m_z.nominal) <= 10), axis=1)

    # get the correct btag WPs and column from the config (as setup by jet_selection)
    btag_wp_score = self.config_inst.x.btag_wp_score
    btag_tight_score = self.config_inst.x.btag_working_points[self.config_inst.x.b_tagger]["tight"]
    btag_column = self.config_inst.x.btag_column

    # TODO: I am not sure if the lepton.matched_jet is working as intended
    # TODO: fakeable masks seem to be too tight

    # fakeable masks
    e_mask_fakeable = (
        e_mask_loose &
        (
            (abs(electron.eta + electron.deltaEtaSC) > 1.479) & (electron.sieie <= 0.030) |
            (abs(electron.eta + electron.deltaEtaSC) <= 1.479) & (electron.sieie <= 0.011)
        ) &
        (electron.hoe <= 0.10) &
        (electron.eInvMinusPInv >= -0.04) &
        (electron.convVeto) &
        (electron.lostHits == 0) &
        ((electron.mvaTTH >= 0.30) | (electron.mvaIso_WP90)) &
        (
            ((electron.mvaTTH < 0.30) & (electron.matched_jet[btag_column] <= btag_tight_score)) |
            ((electron.mvaTTH >= 0.30) & (electron.matched_jet[btag_column] <= btag_wp_score))
        ) &
        (electron.matched_jet[btag_column] <= btag_wp_score) &
        ((electron.mvaTTH >= 0.30) | (electron.jetRelIso < 0.70))
    )

    mu_mask_fakeable = (
        mu_mask_loose &
        (muon.cone_pt >= 10) &
        (
            ((muon.mvaTTH < 0.50) & (muon.matched_jet[btag_column] <= btag_tight_score)) |
            ((muon.mvaTTH >= 0.50) & (muon.matched_jet[btag_column] <= btag_wp_score))
        ) &
        # missing: DeepJet of nearby jet
        ((muon.mvaTTH >= 0.50) | (muon.jetRelIso < 0.80))
    )

    # tight masks
    e_mask_tight = (
        e_mask_fakeable &
        (electron.mvaTTH >= 0.30)
    )
    mu_mask_tight = (
        mu_mask_fakeable &
        (muon.mvaTTH >= 0.50) &
        (muon.mediumId)
    )

    # tau veto mask (only needed in SL?)
    # TODO: update criteria
    tau_mask_veto = (
        (abs(events.Tau.eta) < 2.3) &
        # (abs(events.Tau.dz) < 0.2) &
        (events.Tau.pt > 20.0) &
        (events.Tau.idDeepTau2017v2p1VSe >= 4) &  # 4: VLoose
        (events.Tau.idDeepTau2017v2p1VSmu >= 8) &  # 8: Tight
        (events.Tau.idDeepTau2017v2p1VSjet >= 2)  # 2: VVLoose
    )

    # store number of Loose/Fakeable/Tight electrons/muons/taus as cutflow variables
    events = set_ak_column(events, "cutflow.n_loose_electron", ak.sum(e_mask_loose, axis=1))
    events = set_ak_column(events, "cutflow.n_loose_muon", ak.sum(mu_mask_loose, axis=1))
    events = set_ak_column(events, "cutflow.n_fakeable_electron", ak.sum(e_mask_fakeable, axis=1))
    events = set_ak_column(events, "cutflow.n_fakeable_muon", ak.sum(mu_mask_fakeable, axis=1))
    events = set_ak_column(events, "cutflow.n_tight_electron", ak.sum(e_mask_tight, axis=1))
    events = set_ak_column(events, "cutflow.n_tight_muon", ak.sum(mu_mask_tight, axis=1))
    events = set_ak_column(events, "cutflow.n_veto_tau", ak.sum(tau_mask_veto, axis=1))

    # store info whether lepton is tight or not
    events = set_ak_column(events, "Muon.is_tight", mu_mask_tight)
    events = set_ak_column(events, "Electron.is_tight", e_mask_tight)

    # create the SelectionResult
    lepton_results = SelectionResult(
        steps=steps,
        objects={
            "Electron": {
                "LooseElectron": masked_sorted_indices(e_mask_loose, electron.pt),
                "FakeableElectron": masked_sorted_indices(e_mask_fakeable, electron.pt),
                "TightElectron": masked_sorted_indices(e_mask_tight, electron.pt),
            },
            "Muon": {
                "LooseMuon": masked_sorted_indices(mu_mask_loose, muon.pt),
                "FakeableMuon": masked_sorted_indices(mu_mask_fakeable, muon.pt),
                "TightMuon": masked_sorted_indices(mu_mask_tight, muon.pt),
            },
            "Tau": {"VetoTau": masked_sorted_indices(tau_mask_veto, events.Tau.pt)},
        },
        aux={
            "mu_mask_fakeable": mu_mask_fakeable,
            "e_mask_fakeable": e_mask_fakeable,
            "mu_mask_tight": mu_mask_tight,
            "e_mask_tight": e_mask_tight,
        },
    )

    return events, lepton_results


@lepton_definition.init
def lepton_definition_init(self: Selector) -> None:
    # update selector steps labels
    self.config_inst.x.selector_step_labels = self.config_inst.x("selector_step_labels", {})
    self.config_inst.x.selector_step_labels.update({
        "ll_lowmass_veto": r"$m_{ll} > 12 GeV$",
        "ll_zmass_veto": r"$|m_{ll} - m_{Z}| > 10 GeV$",
    })

    if self.config_inst.x("do_cutflow_features", False):
        # add cutflow features to *produces* only when requested
        self.produces.add("cutflow.n_loose_electron")
        self.produces.add("cutflow.n_loose_muon")
        self.produces.add("cutflow.n_fakeable_electron")
        self.produces.add("cutflow.n_fakeable_muon")
        self.produces.add("cutflow.n_tight_electron")
        self.produces.add("cutflow.n_tight_muon")
        self.produces.add("cutflow.n_veto_tau")

        # TODO: add cutflow variables aswell


@selector(
    uses={jet_selection} | four_vec("Jet"),
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
    uses=(
        {
            jet_selection,
        } |
        four_vec(
            {"Jet", "Electron", "Muon"},
        ) | {"Jet.jetId"} |
        four_vec(
            "FatJet", {"msoftdrop", "jetId", "subJetIdx1", "subJetIdx2", "tau1", "tau2"},
        )
    ),
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
    events = set_ak_column(events, "FatJet.n_subjets", ak.sum(dr_fatjet_ak4 < 0.8, axis=2))
    events = set_ak_column(events, "FatJet.n_separated_jets", ak.sum(dr_fatjet_ak4 > 1.2, axis=2))
    events = set_ak_column(events, "FatJet.max_dr_ak4", ak.max(dr_fatjet_ak4, axis=2))

    # baseline fatjet selection
    fatjet_mask = (
        (events.FatJet.pt > 200) &
        (abs(events.FatJet.eta) < 2.4) &
        (events.FatJet.jetId == 6) &
        (ak.all(events.FatJet.metric_table(electron) > 0.8, axis=2)) &
        (ak.all(events.FatJet.metric_table(muon) > 0.8, axis=2))
    )
    events = set_ak_column(events, "cutflow.n_fatjet", ak.sum(fatjet_mask, axis=1))

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
    # for the SL analysis, we additionally require one AK4 jet that is separated with dR > 1.2
    # (as part of the HbbJet definition)
    if self.single_lepton_selector:
        hbbJet_mask = (
            hbbJet_mask &
            (events.FatJet.n_separated_jets >= 1)
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
        self.produces |= {"cutflow.n_fatjet", "cutflow.n_hbbjet"} | four_vec(
            {"FatJet", "HbbJet"},
            {"n_subjets", "n_separated_jets", "max_dr_ak4"},
            skip_defaults=True,
        )


# boosted selection for the DL channel (only one parameter needs to be changed)
dl_boosted_jet_selection = sl_boosted_jet_selection.derive(
    "dl_boosted_jet_selection",
    cls_dict={"single_lepton_selector": False},
)
