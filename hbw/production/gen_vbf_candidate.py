# coding: utf-8

"""
Producer for generator-level VBF candidates in HH->bbWW decays.
"""

from __future__ import annotations


import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.columnar_util import attach_behavior

from hbw.config.cutflow_variables import add_gen_variables


np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={"GenPart.{pt,eta,phi,mass,pdgId,statusFlags,genPartIdxMother}"},
    produces={"gen_hbw_decay.{h1,h2,b1,b2,v1,v2,v1d1,v1d2,v2d1,v2d2}.{pt,eta,phi,mass,pdgId}"},
)
def gen_vbf_candidate(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produce gen-level Z or W bosons from `GenParticle` collection.
    """
    def all_or_raise(arr, msg):
        if not ak.all(arr):
            raise Exception(f"{msg} in {100 * ak.mean(~arr):.3f}% of cases")

    # TODO: for now, this only works for the qq, but could maybe be generalized to all HH->bbWW decays

    # only consider hard process genparticles
    gp = events.GenPart
    gp["index"] = ak.local_index(gp, axis=1)
    gp = gp[events.GenPart.hasFlags("isHardProcess")]
    gp = gp[~ak.is_none(gp, axis=1)]
    abs_id = abs(gp.pdgId)

    # find initial-state particles
    isp_mask = (gp.distinctParentIdxG == -1) & (gp.pt == 0)
    isp = gp[isp_mask]

    # find all non-Higgs daughter particles from inital state
    sec = ak.flatten(isp.children, axis=2)
    sec = sec[abs(sec.pdgId) != 25]
    sec = ak.pad_none(sec, 2)
    gp_ghost = ak.zip({f: EMPTY_FLOAT for f in sec.fields}, with_name="GenParticle")  # TODO: avoid union type
    sec = ak.fill_none(sec, gp_ghost, axis=1)  # axis=1 necessary

    # find hard Higgs bosons
    h = gp[abs_id == 25]
    nh = ak.num(h, axis=1)
    all_or_raise(nh == 2, "number of Higgs != 2")

    # bottoms from H decay
    b = gp[abs_id == 5]
    b = b[(abs(b.distinctParent.pdgId) == 25)]
    b = b[~ak.is_none(b, axis=1)]
    nb = ak.num(b, axis=1)
    all_or_raise(nb == 2, "number of bottom quarks from Higgs decay != 2")

    # Ws or Zs from H decay
    v = gp[(abs_id == 24) | (abs_id == 23)]
    v = v[(abs(v.distinctParent.pdgId) == 25)]
    v = v[~ak.is_none(v, axis=1)]
    nv = ak.num(v, axis=1)
    all_or_raise(nv == 2, "number of Vector bosons from Higgs decay != 2")

    # leptons from W decays
    is_lepton = (abs_id >= 11) & (abs_id <= 16)
    is_quark = (abs_id >= 1) & (abs_id <= 5)
    vdecays = gp[(is_lepton | is_quark)]
    vdecays = vdecays[(abs(vdecays.distinctParent.pdgId) == 24) | (abs(vdecays.distinctParent.pdgId) == 23)]
    vdecays = vdecays[~ak.is_none(vdecays, axis=1)]
    nvdecays = ak.num(vdecays, axis=1)
    all_or_raise((nvdecays % 2) == 0, "number of leptons or quarks from V decays is not dividable by 2")
    all_or_raise(nvdecays == 4, "number of leptons or quarks from V decays != 4")

    # check if decay product charges are valid
    sign = lambda part: (part.pdgId > 0) * 2 - 1
    all_or_raise(ak.sum(sign(b), axis=1) == 0, "two ss bottoms")

    b1 = b[:, 0]
    b2 = b[:, 1]
    v1 = v[:, 0]
    v2 = v[:, 1]

    all_or_raise(sign(b1) == 1, "b1 should have positive charge")
    all_or_raise(sign(b2) == -1, "b2 should have negative charge")
    all_or_raise(sign(v1) == 1, "v1 should have positive charge")
    all_or_raise((sign(v2) == -1) | (v2.pdgId == 23), "v2 should have negative charge or be Z boson")

    # assign decay products to v1 and v2, assuming that the first two decay products are from v1
    # and the last two from v2
    v1decays = vdecays[:, :2]
    v2decays = vdecays[:, 2:]

    v1_valid = ak.sum(sign(v1decays.distinctParent), axis=1) == 2
    all_or_raise(v1_valid, "Both parents of v1decays should have positive charge")
    v2_valid = (
        (ak.sum(sign(v2decays.distinctParent), axis=1) == -2) |
        (ak.sum(v2decays.distinctParent.pdgId == 23, axis=1))
    )
    all_or_raise(v2_valid, "Both parents of v2decays should have negative charge or be Z bosons")

    hhgen = {
        "h1": h[:, 0],
        "h2": h[:, 1],
        "b1": b1,
        "b2": b2,
        "v1": v1,
        "v2": v2,
        "v1d1": v1decays[:, 0],
        "v1d2": v1decays[:, 1],
        "v2d1": v2decays[:, 0],
        "v2d2": v2decays[:, 1],
        "sec1": sec[:, 0],
        "sec2": sec[:, 1],
    }

    gen_hbw_decay = ak.Array({
        gp: {f: np.float32(hhgen[gp][f]) for f in ["pt", "eta", "phi", "mass", "pdgId"]} for gp in hhgen.keys()
    })
    events = set_ak_column(events, "gen_hbw_decay", gen_hbw_decay)

    return events


@gen_vbf_candidate.skip
def gen_vbf_candidate_skip(self: Producer) -> ak.Array:
    # skip Producer if the dataset is not a HH->bbWW dataset
    return not self.dataset_inst.has_tag("is_hbv")


@gen_vbf_candidate.init
def gen_vbf_candidate_init(self: Producer) -> None:
    add_gen_variables(self.config_inst)


@producer(
    uses={
        "gen_hbw_decay.*.*",
    },
    produces={"vbfpair.{dr,deta,mass}"},
)
def gen_hbw_decay_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    sec1 = attach_behavior(events.gen_hbw_decay.sec1, "PtEtaPhiMLorentzVector")
    sec2 = attach_behavior(events.gen_hbw_decay.sec2, "PtEtaPhiMLorentzVector")

    events = set_ak_column(events, "vbfpair.dr", sec1.delta_r(sec2))
    events = set_ak_column(events, "vbfpair.deta", abs(sec1.eta - sec2.eta))
    events = set_ak_column(events, "vbfpair.mass", (sec1 + sec2).mass)

    for route in self.produced_columns:
        events = set_ak_column_f32(events, route, ak.fill_none(ak.nan_to_none(route.apply(events)), -10))

    return events
