# coding: utf-8

"""
Selection modules for HH(bbWW) lepton selections.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Tuple

import law

from cmsdb.constants import m_z
from columnflow.util import maybe_import, DotDict
from columnflow.columnar_util import set_ak_column
from columnflow.selection import Selector, SelectionResult, selector

from hbw.selection.common import masked_sorted_indices
from hbw.selection.jet import jet_selection
from hbw.util import four_vec


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


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
