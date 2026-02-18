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


@categorizer(
    uses={"mll", "Jet.pt", BTAG_COLUMN("Jet")},
    n_jet=2,
)
def mask_fn_hhh_sr_bcut(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    btag_column = self.config_inst.x.btag_column
    btag_wp_score = self.config_inst.x.btag_wp_score
    n_deepjet = ak.sum(events.Jet[btag_column] >= btag_wp_score, axis=-1)
    mask = (events.mll >= 12) & (events.mll < 80)
    mask = mask & (ak.num(events.Jet["pt"], axis=-1) >= self.n_jet)
    mask = mask & (n_deepjet >= 2)
    return events, mask


@categorizer(
    uses={"Jet.pt", BTAG_COLUMN("Jet")},
    n_jet=2,
)
def mask_fn_hhh_bcut(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    btag_column = self.config_inst.x.btag_column
    btag_wp_score = self.config_inst.x.btag_wp_score
    n_deepjet = ak.sum(events.Jet[btag_column] >= btag_wp_score, axis=-1)
    mask = (ak.num(events.Jet["pt"], axis=-1) >= self.n_jet)
    mask = mask & (n_deepjet >= 2)
    return events, mask


@categorizer(
    uses={"mll", "Jet.pt"},
    n_jet=2,
)
def mask_fn_hhh_sr(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.mll >= 12) & (events.mll < 80)
    mask = mask & (ak.num(events.Jet["pt"], axis=-1) >= self.n_jet)
    return events, mask


@categorizer(uses={"mll"}, mll=20)
def mask_fn_mll20(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, (events.mll > self.mll)
