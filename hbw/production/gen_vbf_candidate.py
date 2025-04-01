# coding: utf-8

"""
Producers for L1 prefiring weights.
"""

from __future__ import annotations


import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, InsertableDict, DotDict
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.columnar_util import attach_behavior
from hbw.production.prepare_objects import prepare_objects

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
    #produces={"gen_hbw_decay"},
    # requested GenVBoson columns, passed to the *uses* and *produces*
    # produced_v_columns={"pt"},
    # mc_only=True,
)
def gen_vbf_candidate(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produce gen-level Z or W bosons from `GenParticle` collection.
    """
    def all_or_raise(arr, msg):
        if not ak.all(arr):
            raise Exception(f"{msg} in {100 * ak.mean(~arr):.3f}% of cases")

    # TODO: for now, this only works for HH->bbWW(qqlnu), but could maybe be generalized to all HH->bbWW decays

    # only consider hard process genparticles
    gp = events.GenPart
    gp["index"] = ak.local_index(gp, axis=1)
    gp = gp[events.GenPart.hasFlags("isHardProcess")]
    gp = gp[~ak.is_none(gp, axis=1)]
    abs_id = abs(gp.pdgId)

    # find initial-state particles
    isp_mask = (gp.distinctParentIdxG == -1) &  (gp.pt == 0)
    isp = gp[isp_mask]
    # isp = gp[ak.is_none(gp.parent.pdgId, axis=1)]

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
    nw = ak.num(w, axis=1)
    # all_or_raise(nw == 2, "number of Ws != 2") # TODO: FOund more then one W in 4% -> check 

    # non-top quarks from W decays
    qs = gp[(abs_id >= 1) & (abs_id <= 5)]
    qs = qs[(abs(qs.distinctParent.pdgId) == 24)]
    qs = qs[~ak.is_none(qs, axis=1)]
    nqs = ak.num(qs, axis=1)

    # leptons from W decays
    # ls = gp[(abs_id >= 11) & (abs_id <= 16)]
    # ls = ls[(abs(ls.distinctParent.pdgId) == 24)]
    # ls = ls[~ak.is_none(ls, axis=1)]
    # nls = ak.num(ls, axis=1)
    # all_or_raise((nls % 2) == 0, "number of leptons from W decays is not dividable by 2")
    # all_or_raise(nls >= 3, "number of leptons from W decays != 2")

    # all_or_raise(nqs + nls == 2 * nw, "number of W decay products invalid")

    # check if decay product charges are valid
    sign = lambda part: (part.pdgId > 0) * 2 - 1
    all_or_raise(ak.sum(sign(b), axis=1) == 0, "two ss bottoms")
    # all_or_raise(ak.sum(sign(w), axis=1) == 0, "two ss Ws")
    # all_or_raise(ak.sum(sign(qs), axis=1) == 0, "sign-imbalance for quarks")
    # all_or_raise(ak.sum(sign(ls), axis=1) == 0, "sign-imbalance for leptons")

    # identify decay products of W's
    # lepton = ls[abs(ls.pdgId) % 2 == 1][:, 0]
    # neutrino = ls[abs(ls.pdgId) % 2 == 0][:, 0]
    # q_dtype = qs[abs(qs.pdgId) % 2 == 1][:, 0]
    # q_utype = qs[abs(qs.pdgId) % 2 == 0][:, 0]

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
        # "wlep": wlep,
        # "whad": whad,
        # "l": lepton,
        # "nu": neutrino,
        # "q1": q_dtype,
        # "q2": q_utype,
        "sec1": sec[:, 0],
        "sec2": sec[:, 1],
    }

    gen_hbw_decay = ak.Array({
        gp: {f: np.float32(hhgen[gp][f]) for f in ["pt", "eta", "phi", "mass", "pdgId"]} for gp in hhgen.keys()
    })
    events = set_ak_column(events, "gen_hbw_decay", gen_hbw_decay)
    # __import__("IPython").embed()
    # for var in ["pt", "eta", "phi", "mass"]:
    #     for gp in ["h1", "h2", "b1", "b2", "l", "nu", "q1", "q2", "sec1", "sec2"]:
    #         events = set_ak_column(events, f"gen.{gp}_{var}", gen_hbw_decay[gp][var])

    return events


@gen_vbf_candidate.skip
def gen_vbf_candidate_skip(self: Producer) -> None:
    add_gen_variables(self.config_inst)


@producer(
    uses={
        #"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB",
        "gen_hbw_decay.*.*",
    },
    produces={"vbfpair.{dr,deta,mass}"},
)
def gen_hbw_decay_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # __import__("IPython").embed()
    sec1 = attach_behavior(events.gen_hbw_decay.sec1, "PtEtaPhiMLorentzVector")
    sec2 = attach_behavior(events.gen_hbw_decay.sec2, "PtEtaPhiMLorentzVector")

    # set_zeros = ak.zeros_like(events.gen_hbw_decay)
    # set_ones = ak.ones_like(events.gen_hbw_decay)
    # set_neg = ak.full_like(events.gen_hbw_decay, -999)
    # mask1 = abs(events.gen_hbw_decay.sec1.eta) < 2.4
    # mask2 = abs(events.gen_hbw_decay.sec2.eta) < 2.4
    # barrel = ak.where(mask1, set_ones, set_neg)
    # barrel = ak.where(mask2, barrel, set_neg)
    # n_events = ak.concatenate(
    #     [set_zeros[..., None], barrel[..., None]],
    #     axis=-1,
    # )
    # __import__("IPython").embed()
    # events = set_ak_column(events, "vbfpair.nevents", n_events.sec1.eta)
    
    events = set_ak_column(events, "vbfpair.dr", sec1.delta_r(sec2))
    events = set_ak_column(events, "vbfpair.deta", abs(sec1.eta - sec2.eta))
    events = set_ak_column(events, "vbfpair.mass", (sec1 + sec2).mass)

    for route in self.produced_columns:
        events = set_ak_column_f32(events, route, ak.fill_none(ak.nan_to_none(route.apply(events)), -10))
    
    return events

# @gen_hbw_decay_features.init
# def gen_hbw_decay_features_init(self: Producer) -> None:
    # if self.config_inst.x("call_add_gen_variables", True):
    # add gen variables but only on first call
    # add_gen_variables(self.config_inst)
        # self.config_inst.x.call_add_gp_variables = False

