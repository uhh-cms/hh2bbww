# coding: utf-8

"""
Categorizers to apply additional cuts when building histograms, e.g. applying an orthogonal trigger
"""

from __future__ import annotations

from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer
# from columnflow.selection import SelectionResult
# from columnflow.columnar_util import has_ak_column, optional_column
#
# from hbw.util import MET_COLUMN, BTAG_COLUMN

np = maybe_import("numpy")
ak = maybe_import("awkward")


def check_l1_seeds(self: Categorizer, events: ak.Array, trigger) -> ak.Array:
    """
    Check if the unprescaled L1 seeds of a given trigger have fired
    """
    l1_seeds_fired = ak.Array([False] * len(events))

    for l1_seed in self.config_inst.x.hlt_L1_seeds[trigger]:
        l1_seeds_fired = l1_seeds_fired | events.L1[l1_seed]

    return l1_seeds_fired


########################################################################################################################
# Mask functions for dilepton channel
########################################################################################################################
# TODO: Using these deferred functions like MET_COlUMN this could probably done more flexible for all channels
@categorizer()
def mask_fn_dl_orth_with_l1_seeds(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Applies the orthogonal trigger for the dilepton channel, checking if the L1 seeds have fired
    """

    trigger_fired = events.HLT[self.config_inst.x.dl_orthogonal_trigger]
    l1_seeds_fired = check_l1_seeds(self, events, self.config_inst.x.dl_orthogonal_trigger)

    mask = trigger_fired & l1_seeds_fired

    return events, mask


@mask_fn_dl_orth_with_l1_seeds.init
def mask_fn_dl_orth_with_l1_seeds_init(self: Categorizer) -> None:
    """
    Add HLT and L1 columns for the orthogonal trigger
    """
    if not self.config_inst.x.dl_orthogonal_trigger or not self.config_inst.x.hlt_L1_seeds:
        raise ValueError("No orthogonal trigger or associated L1 seeds set for dilepton channel")

    self.uses.add(f"HLT.{self.config_inst.x.dl_orthogonal_trigger}")
    for l1_seed in self.config_inst.x.hlt_L1_seeds[self.config_inst.x.dl_orthogonal_trigger]:
        self.uses.add(f"L1.{l1_seed}")


# orthogonal trigger for the dilepton channel without L1 seeds
@categorizer()
def mask_fn_dl_orth(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Applies the orthogonal trigger for the dilepton channel without checking L1 seeds
    """

    trigger_fired = events.HLT[self.config_inst.x.dl_orthogonal_trigger]
    mask = trigger_fired

    return events, mask


@mask_fn_dl_orth.init
def mask_fn_dl_orth_init(self: Categorizer) -> None:
    """
    Add HLT column for the orthogonal trigger without L1 seeds
    """
    if not self.config_inst.x.dl_orthogonal_trigger:
        raise ValueError("No orthogonal trigger set for dilepton channel")

    self.uses.add(f"HLT.{self.config_inst.x.dl_orthogonal_trigger}")


# categorizer that applies the lepton pt cuts
@categorizer(
    uses={"Lepton.pt"},
)
def mask_fn_lep_pt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Applies the lepton pt cuts for the dilepton channel
    """

    mask = events.Lepton.pt[:, 0] > 25

    return events, mask


# categorizer with lepton pt cut and orthogonal trigger
@categorizer()
def mask_fn_lep_pt_orth(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Applies the lepton pt cuts and the orthogonal trigger for the dilepton channel
    """

    trigger_fired = events.HLT[self.config_inst.x.dl_orthogonal_trigger]
    l1_seeds_fired = check_l1_seeds(self, events, self.config_inst.x.dl_orthogonal_trigger)

    orth_mask = trigger_fired & l1_seeds_fired

    mask = (events.Lepton.pt[:, 0] > 25) & orth_mask

    return events, mask


@mask_fn_lep_pt_orth.init
def mask_fn_lep_pt_orth_init(self: Categorizer) -> None:
    """
    Add HLT and L1 columns for the orthogonal trigger with lepton pt cut
    """
    if not self.config_inst.x.dl_orthogonal_trigger or not self.config_inst.x.hlt_L1_seeds:
        raise ValueError("No orthogonal trigger or associated L1 seeds set for dilepton channel")

    self.uses.add(f"HLT.{self.config_inst.x.dl_orthogonal_trigger}")
    for l1_seed in self.config_inst.x.hlt_L1_seeds[self.config_inst.x.dl_orthogonal_trigger]:
        self.uses.add(f"L1.{l1_seed}")

    self.uses.add("Lepton.pt")


# cats with additional mll cut
@categorizer(
    uses={"Lepton.pt", "mll"},
)
def mask_fn_mll_lep(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Applies the mll cut for the dilepton channel
    """

    mask = (events.Lepton.pt[:, 0] > 25) & (events.mll > 20)

    return events, mask


@categorizer(
    uses={"Lepton.pt", "mll"},
)
def mask_fn_mll_lep_orth(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Applies the mll cut and the orthogonal trigger for the dilepton channel
    """

    trigger_fired = events.HLT[self.config_inst.x.dl_orthogonal_trigger]
    l1_seeds_fired = check_l1_seeds(self, events, self.config_inst.x.dl_orthogonal_trigger)

    orth_mask = trigger_fired & l1_seeds_fired

    cut_mask = (events.Lepton.pt[:, 0] > 25) & (events.mll > 20)

    mask = cut_mask & orth_mask

    return events, mask


@mask_fn_mll_lep_orth.init
def mask_fn_mll_lep_orth_init(self: Categorizer) -> None:
    """
    Add HLT and L1 columns for the orthogonal trigger with mll cut
    """
    if not self.config_inst.x.dl_orthogonal_trigger or not self.config_inst.x.hlt_L1_seeds:
        raise ValueError("No orthogonal trigger or associated L1 seeds set for dilepton channel")

    self.uses.add(f"HLT.{self.config_inst.x.dl_orthogonal_trigger}")
    for l1_seed in self.config_inst.x.hlt_L1_seeds[self.config_inst.x.dl_orthogonal_trigger]:
        self.uses.add(f"L1.{l1_seed}")
