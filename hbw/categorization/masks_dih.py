# coding: utf-8

"""
Selection methods defining categories based on selection step results.
"""

from __future__ import annotations

from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer
from hbw.util import MET_COLUMN, IF_DY

np = maybe_import("numpy")
ak = maybe_import("awkward")


@categorizer(uses={"{Electron,Muon}.{pt,eta,phi,mass}", "mll"})
def mask_fn_highpt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Categorizer that selects events in the phase space that we understand.
    Needs to be used in combination with a Producer that defines the leptons.
    """
    mask = (events.Lepton[:, 0].pt > 70) & (events.Lepton[:, 1].pt > 50) & (events.mll > 20)
    return events, mask


@categorizer(uses={"gen_hbw_decay.*.*"})
def mask_fn_gen_barrel(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Categorizer that selects events generated only in the barrel region
    """
    mask = (abs(events.gen_hbw_decay["sec1"]["eta"]) < 2.4) & (abs(events.gen_hbw_decay["sec2"]["eta"]) < 2.4)
    return events, mask


@categorizer(uses={"mll"}, mll=20)
def mask_fn_mll20(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, (events.mll > self.mll)


@categorizer(uses={"mli_mbb"}, mbb=80)
def mask_fn_mbb80(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, (events.mli_mbb > self.mbb)


mask_fn_mll15 = mask_fn_mll20.derive("mask_fn_mll15", cls_dict={"mll": 15})


@categorizer(uses={MET_COLUMN("pt"), MET_COLUMN("phi"), IF_DY("RecoilCorrMET.{pt,phi}")}, met_req=70)
def mask_fn_met70(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    if self.dataset_inst.has_tag("is_dy"):
        mask = events.RecoilCorrMET.pt < self.met_req
    else:
        mask = events[self.config_inst.x.met_name]["pt"] < self.met_req
    return events, mask


@categorizer(uses={MET_COLUMN("pt"), MET_COLUMN("phi"), IF_DY("RecoilCorrMET.{pt,phi}"), "Muon.pt"}, met_req=70)
def mask_fn_dyvr(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    if self.dataset_inst.has_tag("is_dy"):
        mask = events.RecoilCorrMET.pt < self.met_req
        mask = mask & (ak.sum(events.Muon["pt"] > 0, axis=-1) == 2)
    else:
        mask = events[self.config_inst.x.met_name]["pt"] < self.met_req
        mask = mask & (ak.sum(events.Muon["pt"] > 0, axis=-1) == 2)
    return events, mask


@categorizer(
    uses={
        MET_COLUMN("pt"), MET_COLUMN("phi"), IF_DY("RecoilCorrMET.{pt,phi}"),
    },
    met_req=40,
)
def mask_fn_met_geq40(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    if self.dataset_inst.has_tag("is_dy"):
        mask = events.RecoilCorrMET.pt >= self.met_req
    else:
        mask = events[self.config_inst.x.met_name]["pt"] >= self.met_req
    return events, mask


@categorizer(
    uses={
        "{Electron,Muon}.{pt,eta,phi,mass}",
    },
)
def mask_fn_lep2_pt15(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.Lepton[:, 1].pt > 15)
    return events, mask


@categorizer(
    uses={
        "{Electron,Muon}.{pt,eta,phi,mass}",
    },
)
def mask_fn_lep2_pt10(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.Lepton[:, 1].pt > 10)
    return events, mask


@categorizer(
    uses={
        MET_COLUMN("pt"), MET_COLUMN("phi"), IF_DY("RecoilCorrMET.{pt,phi}"),
        "{Electron,Muon}.{pt,eta,phi,mass}",
    },
    met_req=40,
)
def mask_fn_met_geq40_lep2_pt15(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    if self.dataset_inst.has_tag("is_dy"):
        mask = events.RecoilCorrMET.pt >= self.met_req
    else:
        mask = events[self.config_inst.x.met_name]["pt"] >= self.met_req
    mask = mask & (events.Lepton[:, 1].pt > 15)
    return events, mask


@categorizer(
    uses={
        "trigger_ids",
    },
)
def mask_fn_single_lep_triggers(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ak.any(events.trigger_ids == 101, axis=-1) | ak.any(events.trigger_ids == 201, axis=-1)
    return events, mask
