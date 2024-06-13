# coding: utf-8

"""
Production methods with gen-level object information.
"""

from __future__ import annotations

__all__ = ["n_gen_particles", "hbw_process_ids", "hh_bbvv_process_producer"]

import law
import order as od

from columnflow.util import maybe_import
from columnflow.production import producer, Producer
from columnflow.production.processes import process_ids
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)

pdgId_map = {
    1: "down",
    2: "up",
    3: "strange",
    4: "charm",
    5: "bottom",
    6: "top",
    11: "electron",
    12: "e_neutrino",
    13: "muon",
    14: "mu_neutrino",
    15: "tau",
    16: "tau_neutrino",
    21: "gluon",
    22: "photon",
    23: "z",
    24: "w",
    25: "higgs",
}


@producer(produces={"process_id"})
def hbw_process_ids(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Assigns each event a single process id, based on the first process that is registered for the
    internal py:attr:`dataset_inst`. This is rather a dummy method and should be further implemented
    depending on future needs (e.g. for sample stitching).
    """
    logger.info(f"running process Producer {self.process_producer.cls_name} for dataset {self.dataset_inst.name}")
    events = self[self.process_producer](events, **kwargs)

    return events


@hbw_process_ids.init
def hbw_process_ids_init(self: Producer) -> None:
    if not hasattr(self, "dataset_inst"):
        return

    if self.dataset_inst.has_tag("is_hbv"):
        self.process_producer = hh_bbvv_process_producer
    elif self.dataset_inst.has_tag("is_z_jets"):
        self.process_producer = dy_nlo_process_producer
    elif len(self.dataset_inst.processes) == 1:
        self.process_producer = process_ids
    else:
        raise NotImplementedError(
            f"TODO: implement process Producer for dataset {self.dataset_inst.name} "
            f"with Processes {self.dataset_inst.processes.names()}"
        )

    if self.process_producer:
        self.uses.add(self.process_producer)
        self.produces.add(self.process_producer)

    return


@producer(
    uses={"GenPart.pdgId", "GenPart.statusFlags", "GenPart.genPartIdxMother"},
    mc_only=True,
)
def n_gen_particles(
    self: Producer,
    events: ak.Array,
    flags: list[str] = ["isHardProcess"],
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """ Categorizer to select events with a certain number of prompt gen particles """
    if self.dataset_inst.is_data:
        return events

    gp = events.GenPart

    # only consider genparticles that have a mother (no initial-state partons)
    gp = gp[gp.genPartIdxMother >= 0]

    if flags:
        for flag in law.util.make_list(flags):
            if "!" in flag:
                flag = flag.replace("!", "")
                gp = gp[~gp.hasFlags(flag)]
            else:
                gp = gp[gp.hasFlags(flag)]
            gp = gp[~ak.is_none(gp, axis=1)]

    gp_id = abs(gp.pdgId)

    for pdgId in self.pdgIds:
        events = set_ak_column(events, f"n_gen.{pdgId_map[pdgId]}", ak.sum(gp_id == pdgId, axis=1))

    events = set_ak_column(events, "n_gen.clep", events.n_gen.electron + events.n_gen.muon + events.n_gen.tau)
    events = set_ak_column(events, "n_gen.neutrino", (
        events.n_gen.e_neutrino + events.n_gen.mu_neutrino + events.n_gen.tau_neutrino
    ))
    events = set_ak_column(events, "n_gen.quark", (
        events.n_gen.down + events.n_gen.up + events.n_gen.strange +
        events.n_gen.charm + events.n_gen.bottom + events.n_gen.top
    ))
    events = set_ak_column(events, "n_gen.hadronic", events.n_gen.quark + events.n_gen.gluon)

    return events


@n_gen_particles.init
def n_gen_particles_init(self: Producer) -> None:
    self.pdgIds = tuple(pdgId_map.keys())
    self.produces.update({f"n_gen.{particle}" for particle in pdgId_map.values()})
    self.produces.update({"n_gen.clep", "n_gen.neutrino", "n_gen.quark", "n_gen.hadronic"})


def get_process_id_from_masks(
    events: ak.Array,
    process_masks: dict[str, ak.Array],
    dataset_inst: od.Dataset,
) -> ak.Array:
    """
    Assigns a process ID to each event based on the masks in *process_masks*.
    """
    leaf_procs = dataset_inst.get_leaf_processes()

    process_id = ak.Array(np.zeros(len(events)).astype(np.int32))
    for proc_name, mask in process_masks.items():
        if ak.any(mask):
            if not dataset_inst.has_process(proc_name):
                raise Exception(
                    f"Events from dataset {dataset_inst.name} are assigned process {proc_name} "
                    f"but dataset has only {leaf_procs} registered as leaf processes"
                )
            proc_id = dataset_inst.get_process(proc_name).id

            if not ak.all(process_id[mask] == 0):
                raise Exception(f"Events from dataset {dataset_inst.name} have overlapping processes")

            process_id = ak.where(mask, proc_id, process_id)

    if ak.any(process_id == 0):
        raise Exception(f"Events from dataset {dataset_inst.name} have not been assigned any process")

    return process_id


@producer(
    uses={n_gen_particles},
    produces={n_gen_particles, "process_id"},
)
def hh_bbvv_process_producer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[n_gen_particles](events, **kwargs)
    if ak.any(events.n_gen.higgs != 2):
        raise Exception("HH samples should always have exactly two Higgs bosons")

    n_clep = events.n_gen.clep
    n_neutrino = events.n_gen.neutrino
    n_z = events.n_gen.z
    n_w = events.n_gen.w
    base_proc_name = self.dataset_inst.name.split("_hbb_h")[0]
    process_masks = {
        f"{base_proc_name}_hbb_hww2l2nu": (n_clep == 2) & (n_w == 2),
        f"{base_proc_name}_hbb_hwwqqlnu": (n_clep == 1) & (n_w == 2),
        f"{base_proc_name}_hbb_hzz2l2nu": (n_clep == 2) & (n_neutrino == 2) & (n_z == 2),
        f"{base_proc_name}_hbb_hww4q": (n_clep == 0) & (n_w == 2),
        f"{base_proc_name}_hbb_hzz4l": (n_clep == 4) & (n_z == 2),
        f"{base_proc_name}_hbb_hzz2l2q": (n_clep == 2) & (n_neutrino == 0) & (n_z == 2),
        f"{base_proc_name}_hbb_hzz2q2nu": (n_clep == 0) & (n_neutrino == 2) & (n_z == 2),
        f"{base_proc_name}_hbb_hzz4nu": (n_clep == 0) & (n_neutrino == 4) & (n_z == 2),
        f"{base_proc_name}_hbb_hzz4q": (n_clep == 0) & (n_neutrino == 0) & (n_z == 2),
    }

    process_id = get_process_id_from_masks(events, process_masks, self.dataset_inst)
    events = set_ak_column(events, "process_id", process_id, value_type=np.int32)

    return events


@producer(
    uses={"LHE.NpNLO"},
    produces={"process_id", "LHE.NpNLO"},
)
def dy_nlo_process_producer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    n_partons = events.LHE.NpNLO

    base_proc_name = "dy_lep_m50"
    process_masks = {
        f"{base_proc_name}_0j": n_partons == 0,
        f"{base_proc_name}_1j": n_partons == 1,
        f"{base_proc_name}_2j": n_partons == 2,
        f"{base_proc_name}_3j": n_partons == 3,  # should not be assigned
    }

    process_id = get_process_id_from_masks(events, process_masks, self.dataset_inst)
    events = set_ak_column(events, "process_id", process_id, value_type=np.int32)

    return events
