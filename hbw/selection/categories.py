# coding: utf-8

"""
Selection methods defining categories based on selection step results.
"""

from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer
from columnflow.selection import SelectionResult

np = maybe_import("numpy")
ak = maybe_import("awkward")


@categorizer(uses={"event"}, call_force=True)
def catid_selection_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ak.ones_like(events.event) > 0
    return events, mask

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
    Categorization of events in the boosted category: presence of at least 1 AK8 jet fulfilling
    requirements given by the Selector called in SelectEvents
    """
    mask = (ak.sum(events.Jet.pt > 0, axis=-1) >= 1) & (ak.sum(events.HbbJet.pt > 0, axis=-1) >= 1)
    return events, mask


@categorizer(uses={"Jet.pt", "FatJet.pt"}, call_force=True)
def catid_resolved(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Categorization of events in the resolved category: presence of no AK8 jets fulfilling
    requirements given by the Selector called in SelectEvents
    """
    mask = (ak.sum(events.Jet.pt > 0, axis=-1) >= 3) & (ak.sum(events.FatJet.pt > 0, axis=-1) == 0)
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
