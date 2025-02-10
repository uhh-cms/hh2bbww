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
from columnflow.columnar_util import set_ak_column
from columnflow.selection import Selector, SelectionResult, selector

# from columnflow.production.cms.electron import electron_weights
# from columnflow.production.cms.muon import muon_weights

from hbw.selection.common import masked_sorted_indices
from hbw.selection.jet import jet_selection
from hbw.util import call_once_on_config


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@selector(
    uses={
        # muon_weights, electron_weights  # we could load muon and electron weights producer for checks
        "{Electron,Muon,Tau}.{pt,eta,phi,mass}",
        "Electron.{dxy,dz,cutBased}",
        "Muon.{dxy,dz,looseId,pfIsoId}",
        "Tau.{dz,idDeepTau2017v2p1VSe,idDeepTau2017v2p1VSmu,idDeepTau2017v2p1VSjet,decayMode,idDeepTau2018v2p5VSe,idDeepTau2018v2p5VSmu,idDeepTau2018v2p5VSjet}",
        jet_selection,
    },
    produces={
        "Muon.is_tight", "Electron.is_tight",
    },
    # TODO: we would need to move these attributes to the main Selector init to make this configurable
    muon_id="TightId",  # options: MediumId, MediumPromptId, TightId
    muon_iso="TightPFIso",  # options: LoosePFIso, TightPFIso (for MediumId only: Loose/Medium/TightMiniIso)
    electron_id="TightId",  # options: LooseId, MediumId, TightId, wp80iso, wp90iso, wp80noiso, wp90noiso
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

    electron = events.Electron
    muon = events.Muon

    #
    # loose masks
    # TODO: the loose id + iso reqs might depend on the requested (tight) id + iso
    #
    e_mask_loose = (
        (electron.pt >= 7) &
        (abs(electron.eta) <= 2.5) &
        (abs(electron.dxy) <= 0.05) &
        (abs(electron.dz) <= 0.1) &
        (electron.cutBased >= 1)  # veto Id
    )
    mu_mask_loose = (
        (muon.pt >= 5) &
        (abs(muon.eta) <= 2.4) &
        (abs(muon.dxy) <= 0.05) &  # muon efficiencies are computed with dxy < 0.2; loosen?
        (abs(muon.dz) <= 0.1) &  # muon efficiencies are computed with dz < 0.5; loosen?
        (muon.looseId) &  # loose Id
        (muon.pfIsoId >= 2)  # loose Isolation
    )

    #
    # fakeable masks
    #

    e_mask_fakeable = (
        e_mask_loose &
        (electron.pt >= 10)
    )

    mu_mask_fakeable = (
        mu_mask_loose &
        (muon.pt >= 10) &
        self.muon_id_req(muon)
    )

    #
    # tight masks
    #

    e_mask_tight = (
        e_mask_fakeable &
        self.electron_id_req(electron)
    )
    mu_mask_tight = (
        mu_mask_fakeable &
        self.muon_iso_req(muon)
    )

    # tau veto mask (only needed in SL?)
    # TODO: update criteria
    tau_mask_veto = (
        (abs(events.Tau.eta) < 2.3) &
        # (abs(events.Tau.dz) < 0.2) &
        (events.Tau.pt > 5.0) &
        (events.Tau.idDeepTau2018v2p5VSe >= 2) &  # 2: VvLoose
        (events.Tau.idDeepTau2018v2p5VSmu >= 1) &  # 1: VLoose
        (events.Tau.idDeepTau2018v2p5VSjet >= 50) &  # 5: Medium
        (
            (events.Tau.decayMode == 0) |
            (events.Tau.decayMode == 1) |
            (events.Tau.decayMode == 2) |
            (events.Tau.decayMode == 10) |
            (events.Tau.decayMode == 11)
        )
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
    steps["ll_mass_81"] = ~ak.any((lepton_pairs.m_inv - m_z.nominal) >= 10, axis=1)

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


@lepton_definition.setup
def lepton_definition_setup(
    self: Selector,
    reqs: dict,
    inputs: dict,
    reader_targets: dict,
) -> None:
    # collection of id and isolation requirements
    self.muon_id_req = {
        "LooseId": lambda muon: muon.looseId,
        "MediumId": lambda muon: muon.mediumId,
        "TightId": lambda muon: muon.tightId,
        "MediumPromptId": lambda muon: muon.mediumPromptId,
    }[self.muon_id]

    self.muon_iso_req = {
        "LooseMiniIso": lambda muon: (muon.miniIsoId >= 1),
        "MediumMiniIso": lambda muon: (muon.miniIsoId >= 2),
        "TightMiniIso": lambda muon: (muon.miniIsoId >= 3),
        "LoosePFIso": lambda muon: (muon.pfIsoId >= 2),
        "MediumPFIso": lambda muon: (muon.pfIsoId >= 3),
        "TightPFIso": lambda muon: (muon.pfIsoId >= 4),
    }[self.muon_iso]

    self.electron_id_req = {
        "LooseId": lambda electron: (electron.cutBased >= 2),
        "MediumId": lambda electron: (electron.cutBased >= 3),
        "TightId": lambda electron: (electron.cutBased >= 4),
        "MediumPromptId": lambda electron: electron.mediumPromptId,
        "wp80iso": lambda electron: electron[self.e_mva_iso_wp80],
        "wp90iso": lambda electron: electron[self.e_mva_iso_wp90],
    }[self.electron_id]


@call_once_on_config()
def add_nlep_variables(config: od.Config) -> None:
    """
    Function to add cutflow variables to the config. Variable names of objects follow the format.
    """

    # n object variables
    for var in (
        "n_loose_electron",
        "n_loose_muon",
        "n_fakeable_electron",
        "n_fakeable_muon",
        "n_tight_electron",
        "n_tight_muon",
        "n_veto_tau",
    ):
        config.add_variable(
            name=f"cf.{var}",
            expression=f"cutflow.{var}",
            binning=(11, -0.5, 10.5),
            x_title=f"Number of {var.split('_')[1]} {var.split('_')[-1]}s",
        )

    for wp in ("loose", "fakeable", "tight"):
        config.add_variable(
            name=f"cf.n_{wp}_lepton",
            expression=lambda events, wp=wp: events.cutflow[f"n_{wp}_electron"] + events.cutflow[f"n_{wp}_muon"],
            binning=(11, -0.5, 10.5),
            x_title=f"Number of {wp} leptons",
            aux={"inputs": [f"cutflow.n_{wp}_electron", f"cutflow.n_{wp}_muon"]},
        )


@lepton_definition.init
def lepton_definition_init(self: Selector) -> None:
    # store used muon and electron id and isolation in the config
    self.config_inst.x.muon_id = self.muon_id
    self.config_inst.x.muon_iso = self.muon_iso
    self.config_inst.x.electron_id = self.electron_id

    # add required electron and muon id and iso columns
    muon_id_column = {
        "LooseId": "looseId",
        "MediumId": "mediumId",
        "TightId": "tightId",
        "MediumPromptId": "mediumPromptId",
    }[self.muon_id]
    self.uses.add(f"Muon.{muon_id_column}")

    muon_iso_column = {
        "LooseMiniIso": "miniIsoId",
        "MediumMiniIso": "miniIsoId",
        "TightMiniIso": "miniIsoId",
        "MediumPFIso": "pfIsoId",
        "TightPFIso": "pfIsoId",
    }[self.muon_iso]
    self.uses.add(f"Muon.{muon_iso_column}")

    # mva isolation columns: depend on NanoAOD version
    self.e_mva_iso_wp80 = "mvaIso_WP90" if self.config_inst.x.run == 3 else "mvaFall17V2Iso_WP80"
    self.e_mva_iso_wp90 = "mvaIso_WP90" if self.config_inst.x.run == 3 else "mvaFall17V2Iso_WP90"

    electron_id_column = {
        "LooseId": "cutBased",
        "MediumId": "cutBased",
        "TightId": "cutBased",
        "wp80iso": self.e_mva_iso_wp80,
        "wp90iso": self.e_mva_iso_wp90,
    }[self.electron_id]
    self.uses.add(f"Electron.{electron_id_column}")

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

        # add cutflow variables to config
        add_nlep_variables(self.config_inst)
