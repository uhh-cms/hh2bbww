# coding: utf-8

"""
Selection methods defining categories based on selection step results.
"""

from __future__ import annotations

import law

from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer
from columnflow.selection import SelectionResult
from columnflow.columnar_util import has_ak_column, optional_column

np = maybe_import("numpy")
ak = maybe_import("awkward")


@categorizer(uses={"event"}, call_force=True)
def catid_selection_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ak.ones_like(events.event) > 0
    return events, mask

#
# Categorizers based on gen info
#


@categorizer(
    uses=optional_column("HardGenPart.pdgId", "GenPart.pdgId", "GenPart.statusFlags"),
    gp_dict={},  # dict with (tuple of) pdgId + number of required prompt particles with this pdgId
    ignore_charge=True,
    call_force=True,
    _operator="eq",
)
def catid_n_gen_particles(
    self: Categorizer, events: ak.Array, results: SelectionResult | None = None, **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """ Categorizer to select events with a certain number of prompt gen particles """

    # possible options to compare number of particles with required number of particles: ==, >=, >, <=, <
    assert self._operator in ("eq", "ge", "gt", "le", "lt")

    # start with true mask
    mask = np.ones(len(events), dtype=bool)
    if self.dataset_inst.is_data:
        # for data, always return true mask
        return events, mask

    if has_ak_column(events, "HardGenPart.pdgId"):
        gp_id = events.HardGenPart.pdgId
    else:
        try:
            # try to get gp_id column via SelectionResult
            gp_id = events.GenPart.pdgId[results.objects.GenPart.HardGenPart]
        except AttributeError:
            # try to select hard gen particles via status flags
            gp_id = events.GenPart.pdgId[events.GenPart.hasFlags("isHardProcess")]

    if self.ignore_charge:
        gp_id = abs(gp_id)

    for pdgIds, required_n_particles in self.gp_dict.items():
        # make sure that 'pdgIds' is a tuple
        pdgIds = law.util.make_tuple(pdgIds)

        # get number of gen particles with requested pdgIds for each event
        n_particles = sum([ak.sum(gp_id == pdgId, axis=1) for pdgId in pdgIds])

        # compare number of gen particles with required number of particles with requested operator
        this_mask = getattr(n_particles, f"__{self._operator}__")(required_n_particles)
        mask = mask & this_mask

    return events, mask


catid_gen_0lep = catid_n_gen_particles.derive("catid_gen_0lep", cls_dict={"gp_dict": {11: 0, 13: 0, 15: 0}})
catid_gen_1e = catid_n_gen_particles.derive("catid_gen_1e", cls_dict={"gp_dict": {11: 1, 13: 0, 15: 0}})
catid_gen_1mu = catid_n_gen_particles.derive("catid_gen_1mu", cls_dict={"gp_dict": {11: 0, 13: 1, 15: 0}})
catid_gen_1tau = catid_n_gen_particles.derive("catid_gen_1tau", cls_dict={"gp_dict": {11: 0, 13: 0, 15: 1}})
# catid_gen_2e = catid_n_gen_particles.derive("catid_gen_2e", cls_dict={"gp_dict": {11: 2, 13: 0, 15: 0}})
# catid_gen_2mu = catid_n_gen_particles.derive("catid_gen_2mu", cls_dict={"gp_dict": {11: 0, 13: 2, 15: 0}})
# catid_gen_2tau = catid_n_gen_particles.derive("catid_gen_2tau", cls_dict={"gp_dict": {11: 0, 13: 0, 15: 2}})
# catid_gen_emu = catid_n_gen_particles.derive("catid_gen_emu", cls_dict={"gp_dict": {11: 1, 13: 1, 15: 0}})
# catid_gen_etau = catid_n_gen_particles.derive("catid_gen_etau", cls_dict={"gp_dict": {11: 1, 13: 0, 15: 1}})
# catid_gen_mutau = catid_n_gen_particles.derive("catid_gen_mutau", cls_dict={"gp_dict": {11: 0, 13: 1, 15: 1}})
catid_geq_2_gen_leptons = catid_n_gen_particles.derive(
    "catid_geq_2_gen_leptons", cls_dict={"gp_dict": {(11, 13, 15): 2}, "_operator": "ge"},
)


#
# Categorizer called as part of cf.SelectEvents
#


# SL
@categorizer(uses={"event"}, call_force=True)
def catid_selection_1e(
    self: Categorizer, events: ak.Array, results: SelectionResult, **kwargs,
) -> tuple[ak.Array, ak.Array]:
    mask = (ak.num(results.objects.Electron.Electron, axis=-1) == 1) & (ak.num(results.objects.Muon.Muon, axis=-1) == 0)
    return events, mask


@categorizer(uses={"event"}, call_force=True)
def catid_selection_1mu(
    self: Categorizer, events: ak.Array, results: SelectionResult, **kwargs,
) -> tuple[ak.Array, ak.Array]:
    mask = (ak.num(results.objects.Electron.Electron, axis=-1) == 0) & (ak.num(results.objects.Muon.Muon, axis=-1) == 1)
    return events, mask


# DL
@categorizer(uses={"event"}, call_force=True)
def catid_selection_2e(
    self: Categorizer, events: ak.Array, results: SelectionResult, **kwargs,
) -> tuple[ak.Array, ak.Array]:
    mask = (ak.num(results.objects.Electron.Electron, axis=-1) == 2) & (ak.num(results.objects.Muon.Muon, axis=-1) == 0)
    return events, mask


@categorizer(uses={"event"}, call_force=True)
def catid_selection_2mu(
    self: Categorizer, events: ak.Array, results: SelectionResult, **kwargs,
) -> tuple[ak.Array, ak.Array]:
    mask = (ak.num(results.objects.Electron.Electron, axis=-1) == 0) & (ak.num(results.objects.Muon.Muon, axis=-1) == 2)
    return events, mask


@categorizer(uses={"event"}, call_force=True)
def catid_selection_emu(
    self: Categorizer, events: ak.Array, results: SelectionResult, **kwargs,
) -> tuple[ak.Array, ak.Array]:
    mask = (ak.num(results.objects.Electron.Electron, axis=-1) == 1) & (ak.num(results.objects.Muon.Muon, axis=-1) == 1)
    return events, mask


#
# Categorizer called as part of cf.ProduceColumns
#


# SL
@categorizer(uses={"Electron.pt", "Muon.pt"}, call_force=True)
def catid_1e(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ((ak.sum(events.Electron.pt > 0, axis=-1) == 1) & (ak.sum(events.Muon.pt > 0, axis=-1) == 0))
    return events, mask


@categorizer(uses={"Electron.pt", "Muon.pt"}, call_force=True)
def catid_1mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ((ak.sum(events.Electron.pt > 0, axis=-1) == 0) & (ak.sum(events.Muon.pt > 0, axis=-1) == 1))
    return events, mask


# DL
@categorizer(uses={"Electron.pt", "Muon.pt"}, call_force=True)
def catid_2e(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ((ak.sum(events.Electron.pt > 0, axis=-1) == 2) & (ak.sum(events.Muon.pt > 0, axis=-1) == 0))
    return events, mask


@categorizer(uses={"Electron.pt", "Muon.pt"}, call_force=True)
def catid_2mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ((ak.sum(events.Electron.pt > 0, axis=-1) == 0) & (ak.sum(events.Muon.pt > 0, axis=-1) == 2))
    return events, mask


@categorizer(uses={"Electron.pt", "Muon.pt"}, call_force=True)
def catid_emu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ((ak.sum(events.Electron.pt > 0, axis=-1) == 1) & (ak.sum(events.Muon.pt > 0, axis=-1) == 1))
    return events, mask


@categorizer(uses={"Jet.pt", "HbbJet.pt"}, call_force=True)
def catid_boosted(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Categorization of events in the boosted category: presence of at least 1 AK8 jet candidate
    for the H->bb decay
    """
    mask = (ak.sum(events.HbbJet.pt > 0, axis=-1) >= 1)
    return events, mask


@categorizer(uses={"Jet.pt", "HbbJet.pt"}, call_force=True)
def catid_resolved(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Categorization of events in the resolved category: presence of no AK8 jet candidate
    for the H->bb decay
    """
    mask = (ak.sum(events.HbbJet.pt > 0, axis=-1) == 0)
    return events, mask


@categorizer(uses={"Jet.btagDeepFlavB"}, call_force=True)
def catid_1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    n_deepjet = ak.sum(events.Jet.btagDeepFlavB >= self.config_inst.x.btag_working_points.deepjet.medium, axis=-1)
    mask = (n_deepjet == 1)
    return events, mask


@categorizer(uses={"Jet.btagDeepFlavB"}, call_force=True)
def catid_2b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    n_deepjet = ak.sum(events.Jet.btagDeepFlavB >= self.config_inst.x.btag_working_points.deepjet.medium, axis=-1)
    mask = (n_deepjet >= 2)
    return events, mask


# TODO: not hard-coded -> use config?
ml_processes = [
    "ggHH_kl_1_kt_1_sl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww",
    "tt", "st", "w_lnu", "dy_lep", "v_lep",
    "ggHH_kl_1_kt_1_dl_hbbhww", "t_bkg", "sig",
    "graviton_hh_ggf_bbww_m250", "graviton_hh_ggf_bbww_m350", "graviton_hh_ggf_bbww_m450",
    "graviton_hh_ggf_bbww_m600", "graviton_hh_ggf_bbww_m750", "graviton_hh_ggf_bbww_m1000",
]
for proc in ml_processes:
    @categorizer(
        uses=set(f"mlscore.{proc1}" for proc1 in ml_processes),
        cls_name=f"catid_ml_{proc}",
        proc_col_name=f"{proc}",
        # skip check because we don't know which ML processes were used from the MLModel
        check_used_columns=False,
        call_force=True,
    )
    def dnn_mask(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
        """
        dynamically built Categorizer that categorizes events based on dnn scores
        """
        # start with true mask
        outp_mask = np.ones(len(events), dtype=bool)
        for col_name in events.mlscore.fields:
            # check for each mlscore if *this* score is larger and combine all masks
            mask = events.mlscore[self.proc_col_name] >= events.mlscore[col_name]
            outp_mask = outp_mask & mask

        return events, outp_mask
