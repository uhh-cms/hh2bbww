# coding: utf-8

"""
Selection methods defining categories based on selection step results.
"""

from __future__ import annotations

from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer
from hbw.util import BTAG_COLUMN

np = maybe_import("numpy")
ak = maybe_import("awkward")


# Categories for HHH Analysis region / signal region to handle HHH sepcific cuts after selection
# This is the default HHH sognal region
@categorizer(
    uses={"mll", "Jet.pt", BTAG_COLUMN(
        "Jet"), "{Electron,Muon}.{pt,eta,phi,mass}"},
    n_jet=2,
)
def mask_fn_sr(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    btag_column = self.config_inst.x.btag_column
    btag_wp_score = self.config_inst.x.btag_wp_score
    n_deepjet = ak.sum(events.Jet[btag_column] >= btag_wp_score, axis=-1)
    mask = (events.mll >= 12) & (events.mll < 80)
    mask = mask & (ak.num(events.Jet["pt"], axis=-1) >= self.n_jet)
    mask = mask & (n_deepjet >= 2)
    mask = mask & (events.Lepton[:, 1].pt > 15)
    return events, mask


# Masks for JME and BTV checks,

# ---------------------Check on Jet selection -------------------------------------
@categorizer(
    uses={"mll", "Jet.pt", BTAG_COLUMN(
        "Jet"), "{Electron,Muon}.{pt,eta,phi,mass}"},
    n_jet=0,
)
def mask_fn_check_JME(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.mll >= 12) & (events.mll < 80)
    mask = mask & (ak.num(events.Jet["pt"], axis=-1) >= self.n_jet)
    mask = mask & (events.Lepton[:, 1].pt > 15)
    return events, mask


mask_fn_j1 = mask_fn_check_JME.derive("mask_fn_j1", cls_dict={"n_jet": 1})
mask_fn_j2 = mask_fn_check_JME.derive("mask_fn_j2", cls_dict={"n_jet": 2})
mask_fn_j3 = mask_fn_check_JME.derive("mask_fn_j3", cls_dict={"n_jet": 3})


@categorizer(
    uses={"mll", "Jet.pt", BTAG_COLUMN(
        "Jet"), "{Electron,Muon}.{pt,eta,phi,mass}"},
    n_jet=3,
)
def mask_fn_geqj3(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.mll >= 12) & (events.mll < 80)
    mask = mask & (ak.num(events.Jet["pt"], axis=-1) > self.n_jet)
    mask = mask & (events.Lepton[:, 1].pt > 15)
    return events, mask


# ---------- Check on Bejt selection -----------------------------------------------
@categorizer(
    uses={"mll", "Jet.pt", BTAG_COLUMN(
        "Jet"), "{Electron,Muon}.{pt,eta,phi,mass}"},
    b_jet=0,
)
def mask_fn_check_BTV(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    btag_column = self.config_inst.x.btag_column
    btag_wp_score = self.config_inst.x.btag_wp_score
    n_deepjet = ak.sum(events.Jet[btag_column] >= btag_wp_score, axis=-1)
    mask = (events.mll >= 12) & (events.mll < 80)
    mask = mask & (n_deepjet == self.b_jet)
    mask = mask & (events.Lepton[:, 1].pt > 15)
    return events, mask


mask_fn_bj1 = mask_fn_check_BTV.derive("mask_fn_bj1", cls_dict={"b_jet": 1})
mask_fn_bj2 = mask_fn_check_BTV.derive("mask_fn_bj2", cls_dict={"b_jet": 2})
mask_fn_bj3 = mask_fn_check_BTV.derive("mask_fn_bj3", cls_dict={"b_jet": 3})


@categorizer(
    uses={"mll", "Jet.pt", BTAG_COLUMN(
        "Jet"), "{Electron,Muon}.{pt,eta,phi,mass}"},
    b_jet=3,
)
def mask_fn_geqbj3(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    btag_column = self.config_inst.x.btag_column
    btag_wp_score = self.config_inst.x.btag_wp_score
    n_deepjet = ak.sum(events.Jet[btag_column] >= btag_wp_score, axis=-1)
    mask = (events.mll >= 12) & (events.mll < 80)
    mask = mask & (n_deepjet > self.b_jet)
    mask = mask & (events.Lepton[:, 1].pt > 15)
    return events, mask


@categorizer(
    uses={"mll", "Jet.pt", BTAG_COLUMN(
        "Jet"), "{Electron,Muon}.{pt,eta,phi,mass}"},
    b_jet=2,
)
def mask_fn_geqbj2(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    btag_column = self.config_inst.x.btag_column
    btag_wp_score = self.config_inst.x.btag_wp_score
    n_deepjet = ak.sum(events.Jet[btag_column] >= btag_wp_score, axis=-1)
    mask = (events.mll >= 12) & (events.mll < 80)
    mask = mask & (n_deepjet > self.b_jet)
    mask = mask & (events.Lepton[:, 1].pt > 15)
    return events, mask


@categorizer(
    uses={"mll", "Jet.pt", BTAG_COLUMN(
        "Jet"), "{Electron,Muon}.{pt,eta,phi,mass}"},
    n_jet=1,
)
def mask_fn_nj1(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # btag_column = self.config_inst.x.btag_column
    # btag_wp_score = self.config_inst.x.btag_wp_score
    # n_deepjet = ak.sum(events.Jet[btag_column] >= btag_wp_score, axis=-1)
    mask = (events.mll >= 12) & (events.mll < 80)
    mask = mask & (ak.num(events.Jet["pt"], axis=-1) >= self.n_jet)
    mask = mask & (events.Lepton[:, 1].pt > 15)
    return events, mask


# This is the HHH phase space, without the mll cut
@categorizer(
    uses={"Jet.pt", BTAG_COLUMN("Jet"), "{Electron,Muon}.{pt,eta,phi,mass}"},
    n_jet=2,
)
def mask_fn_ar(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    btag_column = self.config_inst.x.btag_column
    btag_wp_score = self.config_inst.x.btag_wp_score
    n_deepjet = ak.sum(events.Jet[btag_column] >= btag_wp_score, axis=-1)
    mask = (ak.num(events.Jet["pt"], axis=-1) >= self.n_jet)
    mask = mask & (n_deepjet >= 3)
    mask = mask & (events.Lepton[:, 1].pt > 15)
    return events, mask


# Require Lepton criteria for SR and at least 3b jets
# (eventually handled bby ML model / inference but for Plots/ yield table)
# Creates issued with couple of datasets that do not have 3 bjets --> should be handled by dataset collection
@categorizer(
    uses={"Jet.pt", BTAG_COLUMN("Jet"), "{Electron,Muon}.{pt,eta,phi,mass}"},
    n_jet=2,
)
def _yield(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    btag_column = self.config_inst.x.btag_column
    btag_wp_score = self.config_inst.x.btag_wp_score
    n_deepjet = ak.sum(events.Jet[btag_column] >= btag_wp_score, axis=-1)
    mask = (n_deepjet >= 3)
    mask = mask & (events.Lepton[:, 1].pt > 15)
    return events, mask


@categorizer(
    uses={"gatja_output_0"},
)
def mask_gatja_h1(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.gatja_output_0 >= 0.6)
    return events, mask


@categorizer(
    uses={"gatja_output_0"},
)
def mask_non_gatja_h1(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.gatja_output_0 < 0.6)
    return events, mask


@categorizer(
    uses={"gatja_output_0"},
)
def mask_gatja_h1_conservative(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.gatja_output_0 >= 0.85)
    return events, mask


@categorizer(
    uses={"gatja_output_0"},
)
def mask_non_gatja_h1_conservative(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.gatja_output_0 < 0.85)
    return events, mask


# ----------------------- Helper masks ----------------------------------------------------------
# Cut subleadiong lepton pt at 15 GeV (to match the correciton and SF)
@categorizer(
    uses={
        "{Electron,Muon}.{pt,eta,phi,mass}",
    },
)
def mask_fn_lep2_pt15(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = events.Lepton[:, 1].pt > 15
    return events, mask
