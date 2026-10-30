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
    uses={"mll", "Jet.pt", BTAG_COLUMN("Jet"), "{Electron,Muon}.{pt,eta,phi,mass}"},
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
    mask = mask & (n_deepjet >= 2)
    mask = mask & (events.Lepton[:, 1].pt > 15)
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
