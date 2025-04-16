# coding: utf-8

"""
Producers for L1 prefiring weights.
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
    uses={
        "GenPart.pdgId", "GenPart.statusFlags",
        "GenPart.pt", "GenPart.eta", "GenPart.phi", "GenPart.mass",
    },
    produces={"gen_hbw_decay.*.*"},
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

    # Ws from H decay
    w = gp[abs_id == 24]
    w = w[(abs(w.distinctParent.pdgId) == 25)]
    w = w[~ak.is_none(w, axis=1)]
    # nw = ak.num(w, axis=1)

    # non-top quarks from W decays
    qs = gp[(abs_id >= 1) & (abs_id <= 5)]
    qs = qs[(abs(qs.distinctParent.pdgId) == 24)]
    qs = qs[~ak.is_none(qs, axis=1)]
    # nqs = ak.num(qs, axis=1)

    # check if decay product charges are valid
    sign = lambda part: (part.pdgId > 0) * 2 - 1
    all_or_raise(ak.sum(sign(b), axis=1) == 0, "two ss bottoms")

    # identify b1 as particle, b2 as antiparticle
    b1 = b[sign(b) == 1][:, 0]
    b2 = b[sign(b) == -1][:, 0]

    # TODO: identify H->bb and H->WW and switch from h1/h2 to hbb/hww
    # TODO: most fields have type='len(events) * ?genParticle' -> get rid of the '?'

    hhgen = {
        "h1": h[:, 0],
        "h2": h[:, 1],
        "b1": b1,
        "b2": b2,
        "sec1": sec[:, 0],
        "sec2": sec[:, 1],
    }

    gen_hbw_decay = ak.Array({
        gp: {f: np.float32(hhgen[gp][f]) for f in ["pt", "eta", "phi", "mass", "pdgId"]} for gp in hhgen.keys()
    })
    events = set_ak_column(events, "gen_hbw_decay", gen_hbw_decay)

    return events


@gen_vbf_candidate.skip
def gen_vbf_candidate_skip(self: Producer) -> None:
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
