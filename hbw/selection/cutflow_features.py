# coding: utf-8

"""
Selectors to set ak columns for cutflow features
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_INT, optional_column as optional
from columnflow.selection import SelectionResult
from columnflow.production import Producer, producer

from hbw.production.prepare_objects import prepare_objects
from hbw.config.cutflow_variables import add_cutflow_variables

ak = maybe_import("awkward")


@producer(
    # used and produced columns per object are defined in init function
    uses={prepare_objects, "PV.npvs", optional("Pileup.nTrueInt"), optional("Pileup.nPU")},
    produces={
        "cutflow.n_electron", "cutflow.n_muon", "cutflow.n_lepton",
        "cutflow.n_veto_electron", "cutflow.n_veto_muon", "cutflow.n_veto_lepton",
        "cutflow.n_veto_tau",
        "cutflow.npvs", "cutflow.npu_true", "cutflow.npu",
    },
    # skip the checking existence of used/produced columns for now because some columns are not there
    check_used_columns=False,
    check_produced_columns=False,

)
def cutflow_features(self: Producer, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:

    # apply event results to objects and define objects in a convenient way for reconstructing variables
    # but create temporary ak.Array to not override objects in events
    arr = self[prepare_objects](events, results)

    # Nummber of objects
    events = set_ak_column(events, "cutflow.n_electron", ak.num(arr.Electron, axis=1))
    events = set_ak_column(events, "cutflow.n_muon", ak.num(arr.Muon, axis=1))
    events = set_ak_column(events, "cutflow.n_lepton", ak.num(arr.Lepton, axis=1))
    events = set_ak_column(events, "cutflow.n_veto_electron", ak.num(arr.VetoElectron, axis=1))
    events = set_ak_column(events, "cutflow.n_veto_muon", ak.num(arr.VetoMuon, axis=1))
    events = set_ak_column(events, "cutflow.n_veto_lepton", ak.num(arr.VetoLepton, axis=1))
    events = set_ak_column(events, "cutflow.n_veto_tau", ak.num(arr.VetoTau, axis=1))

    # Primary vertices
    events = set_ak_column(events, "cutflow.npvs", events.PV.npvs)
    if "Pileup" in events.fields:
        events = set_ak_column(events, "cutflow.npu_true", events.Pileup.nTrueInt)
        events = set_ak_column(events, "cutflow.npu", events.Pileup.nPU)
    else:
        events = set_ak_column(events, "cutflow.npu_true", EMPTY_INT)
        events = set_ak_column(events, "cutflow.npu", EMPTY_INT)

    # save up to 4 loose jets
    events = set_ak_column(events, "cutflow.LooseJet", arr.LooseJet[:, :3])

    # save up to 3 veto leptons
    events = set_ak_column(events, "cutflow.VetoLepton", arr.VetoLepton[:, :2])
    events = set_ak_column(events, "cutflow.VetoElectron", arr.VetoElectron[:, :2])
    events = set_ak_column(events, "cutflow.VetoMuon", arr.VetoMuon[:, :2])

    return events


@cutflow_features.init
def cutflow_features_init(self: Producer) -> None:
    # add cutflow variable instances to config
    add_cutflow_variables(self.config_inst)

    # define used and produced columns
    self.lepton_columns = {
        "pt", "eta",
        "dxy", "dz", "pfRelIso03_all", "pfRelIso04_all", "miniPFRelIso_all", "mvaTTH",
    }
    self.electron_columns = {"pfRelIso03_all", "mvaFall17V2Iso", "mvaFall17V2noIso"}
    self.muon_columns = {"pfRelIso04_all", "mvaLowPt"}
    self.jet_columns = {"pt"}

    self.uses |= (
        set(
            f"{obj}.{var}" for obj in ("Electron", "Muon") for var in self.lepton_columns
        ) |
        set(
            f"Electron.{var}" for var in self.electron_columns
        ) |
        set(
            f"Muon.{var}" for var in self.muon_columns
        )
    )

    self.produces |= (
        set(
            f"cutflow.VetoLepton.{var}" for var in self.lepton_columns
        ) |
        set(
            f"cutflow.VetoMuon.{var}" for var in self.muon_columns
        ) |
        set(
            f"cutflow.VetoElectron.{var}" for var in self.electron_columns
        ) |
        set(
            f"cutflow.LooseJet.{var}" for var in self.jet_columns
        )
    )
