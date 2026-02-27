# coding: utf-8

"""
Producer for generator-level VBF candidates in HH->bbWW decays.
"""

from __future__ import annotations

import law

import functools
import math

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import has_ak_column, set_ak_column, EMPTY_FLOAT
from columnflow.columnar_util import attach_behavior, Route

from hbw.config.cutflow_variables import add_gen_variables

from hbw.util import call_once_on_config


np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_f64 = functools.partial(set_ak_column, value_type=np.float64)


logger = law.logger.get_logger(__name__)


@producer(
    uses={
        "GenPart.{pt,eta,phi,mass,pdgId,statusFlags,genPartIdxMother}",
    },
    produces={
        "gen_hbw_decay.{h1,h2,b1,b2,v1,v2,v1d1,v1d2,v2d1,v2d2}.{pt,eta,phi,mass,pdgId}",
    },
)
def gen_hbv_decay(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
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
        gp: {f: np.float64(hhgen[gp][f]) for f in ["pt", "eta", "phi", "mass", "pdgId"]} for gp in hhgen.keys()
    })
    events = set_ak_column(events, "gen_hbw_decay", gen_hbw_decay)

    return events


@gen_hbv_decay.skip
def gen_hbv_decay_skip(self: Producer) -> ak.Array:
    # skip Producer if the dataset is not a HH->bbWW dataset
    return not self.dataset_inst.has_tag("is_hbv")


@gen_hbv_decay.init
def gen_hbv_decay_init(self: Producer) -> None:
    add_gen_variables(self.config_inst)


@producer(
    uses={
        "gen_hbw_decay.*.*",
    },
    produces={
        # "gen_hbw.lep0.{pt,eta,phi,mass,pdgId}",
        "gen_hbw.{b1,b2,lep1,lep2,w1,w2,higgs,h1,h2,sec1,sec2}.{pt,eta,phi,mass,abseta,pdgId}",
        "gen_hbw.{hh,hbb,hww,wlep1,wlep2,bbll,dilep,invis,llnn,vbfpair}.{pt,eta,phi,mass,dr,deta,dphi,ptdiff,abseta,summass,pt_asym}",  # noqa: E501
    },
    version=12,
)
def gen_hbw_decay_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for field in events.gen_hbw_decay.fields:
        events = set_ak_column(events, f"gen_hbw_decay.{field}", attach_behavior(
            events.gen_hbw_decay[field], "PtEtaPhiMLorentzVector",
        ))

    gp = events.gen_hbw_decay

    leading_w = ak.where(
        gp.v1.mass > gp.v2.mass, gp.v1, gp.v2,
    )
    subleading_w = ak.where(
        gp.v1.mass <= gp.v2.mass, gp.v1, gp.v2,
    )

    is_charged_lepton = lambda abs_id: (abs_id == 11) | (abs_id == 13) | (abs_id == 15)
    # is_neutrino = lambda abs_id: (abs_id == 12) | (abs_id == 14) | (abs_id == 16)

    l1 = ak.where(
        is_charged_lepton(abs(gp.v1d1.pdgId)), gp.v1d1, gp.v1d2,
    )
    nu1 = ak.where(
        is_charged_lepton(abs(gp.v1d1.pdgId)), gp.v1d2, gp.v1d1,
    )
    l2 = ak.where(
        is_charged_lepton(abs(gp.v2d1.pdgId)), gp.v2d1, gp.v2d2,
    )
    nu2 = ak.where(
        is_charged_lepton(abs(gp.v2d1.pdgId)), gp.v2d2, gp.v2d1,
    )

    leading_lep = ak.where(
        l1.pt > l2.pt, l1, l2,
    )
    nu_leading_lep = ak.where(
        l1.pt > l2.pt, nu1, nu2,
    )
    subleading_lep = ak.where(
        l1.pt <= l2.pt, l1, l2,
    )
    nu_subleading_lep = ak.where(
        l1.pt <= l2.pt, nu1, nu2,
    )

    leading_b = ak.where(
        gp.b1.pt > gp.b2.pt, gp.b1, gp.b2,
    )
    subleading_b = ak.where(
        gp.b1.pt <= gp.b2.pt, gp.b1, gp.b2,
    )

    leading_isp = ak.where(
        gp.sec1.pt > gp.sec2.pt, gp.sec1, gp.sec2,
    )
    subleading_isp = ak.where(
        gp.sec1.pt <= gp.sec2.pt, gp.sec1, gp.sec2,
    )

    leading_h = ak.where(
        gp.h1.pt > gp.h2.pt, gp.h1, gp.h2,
    )
    subleading_h = ak.where(
        gp.h1.pt <= gp.h2.pt, gp.h1, gp.h2,
    )

    events = set_ak_column(events, "gen_hbw.lep1", leading_lep)
    events = set_ak_column(events, "gen_hbw.lep2", subleading_lep)
    events = set_ak_column(events, "gen_hbw.higgs", gp.h1)  # does not matter which one
    events = set_ak_column(events, "gen_hbw.h1", leading_h)
    events = set_ak_column(events, "gen_hbw.h2", subleading_h)
    events = set_ak_column(events, "gen_hbw.w1", leading_w)
    events = set_ak_column(events, "gen_hbw.w2", subleading_w)
    events = set_ak_column(events, "gen_hbw.b1", leading_b)
    events = set_ak_column(events, "gen_hbw.b2", subleading_b)
    events = set_ak_column(events, "gen_hbw.sec1", leading_isp)
    events = set_ak_column(events, "gen_hbw.sec2", subleading_isp)

    for col in ("lep1", "lep2", "higgs", "h1", "h2", "w1", "w2", "b1", "b2", "sec1", "sec2"):
        events = set_ak_column(events, f"gen_hbw.{col}.abseta", abs(events.gen_hbw[col].eta))
        events = set_ak_column(events, f"gen_hbw.{col}.pt", abs(events.gen_hbw[col].pt))
        events = set_ak_column(events, f"gen_hbw.{col}.mass", abs(events.gen_hbw[col].mass))
        # events = set_ak_column(events, f"gen_hbw.{col}.p", abs(events.gen_hbw[col].p))

    gen_hbw = events.gen_hbw

    def make_pair_features(
            base_collection, col_name: str, part1, part2,
            additional_features: list[str] = ["ptdiff", "dr", "deta", "dphi", "abseta", "summass", "pt_asym"],
    ):
        pair = part1 + part2
        base_collection = set_ak_column(base_collection, f"{col_name}", pair)
        if "ptdiff" in additional_features:
            base_collection = set_ak_column(base_collection, f"{col_name}.ptdiff", abs(part1.pt - part2.pt))
        if "dr" in additional_features:
            base_collection = set_ak_column(base_collection, f"{col_name}.dr", part1.delta_r(part2))
        if "deta" in additional_features:
            base_collection = set_ak_column(base_collection, f"{col_name}.deta", abs(part1.eta - part2.eta))
        if "dphi" in additional_features:
            base_collection = set_ak_column(base_collection, f"{col_name}.dphi", part1.delta_phi(part2))
        if "abseta" in additional_features:
            base_collection = set_ak_column(base_collection, f"{col_name}.abseta", abs(pair.eta))
        if "summass" in additional_features:
            base_collection = set_ak_column(base_collection, f"{col_name}.summass", part1.mass + part2.mass)
        if "pt_asym" in additional_features:
            base_collection = set_ak_column(
                base_collection,
                f"{col_name}.pt_asym",
                abs(part1.pt - part2.pt) / (part1.pt + part2.pt),
            )

        # hotfix: pt and mass is not automatically stored as additional field (only accessible via behavior),
        # but needs to be explicitely set as column to be stored on disc
        base_collection = set_ak_column(base_collection, f"{col_name}.pt", base_collection[col_name].pt)
        base_collection = set_ak_column(base_collection, f"{col_name}.mass", base_collection[col_name].mass)

        return base_collection

    gen_hbw = make_pair_features(gen_hbw, "hh", gp.h1, gp.h2)
    gen_hbw = make_pair_features(gen_hbw, "hbb", gp.b1, gp.b2)
    gen_hbw = make_pair_features(gen_hbw, "hww", leading_w, subleading_w)
    gen_hbw = make_pair_features(gen_hbw, "wlep1", leading_lep, nu_leading_lep)
    gen_hbw = make_pair_features(gen_hbw, "wlep2", subleading_lep, nu_subleading_lep)

    gen_hbw = make_pair_features(gen_hbw, "bbll", gp.b1 + gp.b2, l1 + l2)
    gen_hbw = make_pair_features(gen_hbw, "dilep", l1, l2)
    gen_hbw = make_pair_features(gen_hbw, "invis", nu1, nu2)
    gen_hbw = make_pair_features(gen_hbw, "llnn", l1 + l2, nu1 + nu2)

    gen_hbw = make_pair_features(gen_hbw, "vbfpair", gp.sec1, gp.sec2)

    vbfpair_nonphysical = np.isinf(gen_hbw.vbfpair.eta)
    if ak.any(vbfpair_nonphysical):
        logger.debug(
            f"Found {ak.sum(vbfpair_nonphysical)} events with non-physical VBF pair (inf eta), "
            "setting all VBF pair variables to EMPTY_FLOAT",
        )
        for col in gen_hbw.vbfpair.fields:
            gen_hbw = set_ak_column(
                gen_hbw,
                f"vbfpair.{col}",
                ak.where(
                    vbfpair_nonphysical,
                    EMPTY_FLOAT,
                    gen_hbw.vbfpair[col],
                ),
            )

    events = set_ak_column(events, "gen_hbw", gen_hbw)
    for route in self.produced_columns:
        if not has_ak_column(events, route):
            logger.warning(f"Produced column {route} is missing")
            continue
        events = set_ak_column_f32(events, route, ak.where(
            np.isinf(route.apply(events)),
            -10,
            ak.nan_to_num(route.apply(events), -10),
        ))
        # events = set_ak_column_f64(events, route, ak.fill_none(ak.nan_to_none(route.apply(events)), -10))

    return events


from hbw.config.styling import default_var_unit
bins_dict = {
    "pt": {
        "default": (120, 0., 480.),
        "gen_hbw.vbfpair": (120, 0., 600.),
        "gen_hbw.lep1": (120, 0., 240.),
        "gen_hbw.lep2": (120, 0., 120.),
        "gen_hbw.b2": (120, 0., 240.),
    },
    "p": {
        "default": (120, 0., 1200.),
        "gen_hbw.vbfpair": (120, 0., 2400.),
        # "gen_hbw.lep1": (120, 0., 800.),
        # "gen_hbw.lep2": (120, 0., 800.),
    },
    "E": {
        "default": (120, 0., 2400.),
        "gen_hbw.vbfpair": (120, 0., 4800.),
        # "gen_hbw.lep1": (120, 0., 800.),
        # "gen_hbw.lep2": (120, 0., 800.),
    },
    "eta": {
        "default": (120, -5., 5.),
    },
    "phi": {
        "default": (96, -3.2, 3.2),
    },
    "mass": {
        "default": (120, 0., 480.),
        "gen_hbw.vbfpair": (120, 0., 4800.),
        "gen_hbw.hh": (120, 0., 1200.),
        "gen_hbw.bbll": (120, 0., 1200.),
        "gen_hbw.dilep": (120, 0., 120.),
        "gen_hbw.invis": (120, 0., 120.),
        "gen_hbw.w1": (120, 0., 120.),
        "gen_hbw.w2": (120, 0., 120.),
        "gen_hbw.wlep1": (120, 0., 120.),
        "gen_hbw.wlep2": (120, 0., 120.),
    },
    "abseta": {
        "default": (120, 0., 5.),
    },
    "dr": {
        "default": (120, 0., 10.),
        "gen_hbw.vbfpair": (120, 0., 14.),
    },
    "deta": {
        "default": (120, 0., 10.),
        "gen_hbw.vbfpair": (120, 0., 12.),
    },
    "dphi": {
        "default": (96, 0., 3.2),
    },
    "ptdiff": {
        "default": (120, 0., 240.),
        "gen_hbw.vbfpair": (120, 0., 600.),
    },
    "pt_asym": {
        "default": (120, 0., 1.),
    },
}
var_title_dict = {
    "pt": lambda col_repr_comb: rf"$p_{{T}}^{{{col_repr_comb}}}$",
    "p": lambda col_repr_comb: rf"$p^{{{col_repr_comb}}}$",
    "E": lambda col_repr_comb: rf"$E^{{{col_repr_comb}}}$",
    "eta": lambda col_repr_comb: rf"$\eta_{{{col_repr_comb}}}$",
    "phi": lambda col_repr_comb: rf"$\phi_{{{col_repr_comb}}}$",
    "mass": lambda col_repr_comb: rf"$m_{{{col_repr_comb}}}$",
    "abseta": lambda col_repr_comb: rf"$|\eta_{{{col_repr_comb}}}|$",
    "dr": lambda col_repr_delta: rf"$\Delta R ({{{col_repr_delta}}})$",
    "deta": lambda col_repr_delta: rf"$\Delta \eta ({{{col_repr_delta}}})$",
    "dphi": lambda col_repr_delta: rf"$\Delta \phi ({{{col_repr_delta}}})$",
    "ptdiff": lambda col_repr_delta: rf"$\Delta p_{{T}} ({{{col_repr_delta}}})$",
    "pt_asym": lambda col_repr_delta: rf"$\Delta p_{{T}} ({{{col_repr_delta}}}) / \Sigma p_{{T}}$",
}


def make_pair_variables(
    config,
    col_name: str,
    obj_reprs: str | tuple[str, str] = ("obj1", "obj2"),
    x_title_prefix: str = "",
    base_features: list[str] = ["pt", "eta", "phi", "mass", "abseta", "p", "E"],
    additional_features: list[str] = ["ptdiff", "dr", "deta", "dphi", "pt_asym"],
    var_basename: str | None = None,
):
    if isinstance(obj_reprs, str):
        col_repr_delta = col_repr_comb = obj_reprs
    else:
        col_repr_delta = f"{obj_reprs[0]},{obj_reprs[1]}"
        col_repr_comb = f"{obj_reprs[0]}{obj_reprs[1]}"

    if var_basename is None:
        var_basename = f"GENHBW_{col_name.split('.')[-1]}"

    for feature in base_features:
        # if feature in ("p", "E"):
        #     continue
        var_title = var_title_dict.get(feature, lambda col_repr_comb: feature)(col_repr_comb)
        binning = bins_dict.get(feature).get(col_name, bins_dict.get(feature).get("default"))
        var = config.add_variable(  # noqa: F841
            name=f"{var_basename}_{feature}",
            # expression to use behaviour instead of direct column access
            expression=lambda events, col_name=col_name, feature=feature: (
                getattr(Route(col_name).apply(events), feature)
            ),
            binning=binning,
            unit=default_var_unit.get(feature, ""),
            x_title=rf"{x_title_prefix} {var_title}",
            aux={
                "inputs": {f"{col_name}.{{{feature},pt,eta,phi,mass}}"},
                "overflow": False if feature in ["eta", "abseta", "phi"] else True,
                "rebin": math.ceil(binning[0] / 40) or 1,
            },
        )
        # if feature in ("E",):
        #     var.x.inputs = {f"{col_name}.{{pt,eta,phi,mass}}"}

    for feature in additional_features:
        var_title = var_title_dict.get(feature, lambda col_repr_comb: feature)(col_repr_delta)
        binning = bins_dict.get(feature).get(col_name, bins_dict.get(feature).get("default"))
        config.add_variable(
            name=f"{var_basename}_{feature}",
            # for pair variables, direct column access is fine
            expression=f"{col_name}.{feature}",
            binning=binning,
            unit=default_var_unit.get(feature, ""),
            x_title=rf"{x_title_prefix} {var_title}",
            aux={
                # "inputs": {f"{col_name}.{{{feature},pt,eta,phi,mass}}"},
                "overflow": True,
                "rebin": math.ceil(binning[0] / 40) or 1,
            },
        )


@gen_hbw_decay_features.init
def gen_hbw_decay_features_init(self: Producer) -> None:
    @call_once_on_config
    def add_gen_hbw_decay_variables(config):
        make_pair_variables(config, "gen_hbw.hh", ["h", "h"], "Gen")
        make_pair_variables(config, "gen_hbw.hbb", ["b", "b"], "Gen")
        make_pair_variables(config, "gen_hbw.hww", ["W", "W"], "Gen")
        make_pair_variables(config, "gen_hbw.wlep1", [r"\ell 1", r"\nu 1"], "Gen")
        make_pair_variables(config, "gen_hbw.wlep2", [r"\ell 2", r"\nu 2"], "Gen")

        make_pair_variables(config, "gen_hbw.bbll", ["bb", r"\ell\ell"], "Gen")
        make_pair_variables(config, "gen_hbw.dilep", [r"\ell", r"\ell"], "Gen")
        make_pair_variables(config, "gen_hbw.invis", [r"\nu", r"\nu"], "Gen")
        make_pair_variables(config, "gen_hbw.llnn", [r"\ell\ell", r"\nu\nu"], "Gen")

        make_pair_variables(config, "gen_hbw.vbfpair", ["j", "j"], "Gen")

        make_pair_variables(config, "gen_hbw.higgs", "Higgs", "Gen", additional_features=[])
        make_pair_variables(config, "gen_hbw.h1", "H1", "Gen", additional_features=[])
        make_pair_variables(config, "gen_hbw.h2", "H2", "Gen", additional_features=[])
        make_pair_variables(config, "gen_hbw.w1", "W1", "Gen", additional_features=[])
        make_pair_variables(config, "gen_hbw.w2", "W2", "Gen", additional_features=[])
        make_pair_variables(config, "gen_hbw.b1", "b1", "Gen", base_features=["p", "pt", "eta", "phi", "abseta"], additional_features=[])  # noqa: E501
        make_pair_variables(config, "gen_hbw.b2", "b2", "Gen", base_features=["p", "pt", "eta", "phi", "abseta"], additional_features=[])  # noqa: E501
        make_pair_variables(config, "gen_hbw.lep1", r"\ell 1", "Gen", base_features=["p", "pt", "eta", "phi", "abseta"], additional_features=[])  # noqa: E501
        make_pair_variables(config, "gen_hbw.lep2", r"\ell 2", "Gen", base_features=["p", "pt", "eta", "phi", "abseta"], additional_features=[])  # noqa: E501
        make_pair_variables(config, "gen_hbw.sec1", "isp 1", "Gen", base_features=["p", "pt", "eta", "phi"], additional_features=[])  # noqa: E501
        make_pair_variables(config, "gen_hbw.sec2", "isp 2", "Gen", base_features=["p", "pt", "eta", "phi"], additional_features=[])  # noqa: E501

        config.add_variable(
            name="GENHBW_mw_sum",
            expression=lambda events: (
                events.gen_hbw.w1.mass + events.gen_hbw.w2.mass
            ),
            binning=(80, 0., 160.),
            unit="GeV",
            x_title=r"Gen $m_{W1} + m_{W2}$",
            aux={
                "inputs": {"gen_hbw.w1.mass", "gen_hbw.w2.mass"},
                "overflow": True,
                "rebin": 2,
            },
        )

        config.add_variable(
            name="GENHBW_mll_mnn_sum",
            expression=lambda events: (
                events.gen_hbw.dilep.mass + events.gen_hbw.invis.mass
            ),
            binning=(80, 0., 160.),
            unit="GeV",
            x_title=r"Gen $m_{\ell\ell} + m_{\nu\nu}$",
            aux={
                "inputs": {"gen_hbw.dilep.mass", "gen_hbw.invis.mass"},
                "overflow": True,
                "rebin": 2,
            },
        )

        var = config.get_variable("GENHBW_dilep_mass").copy()
        var.name = "GENHBW_dilep_mass_for_eff"
        var.binning = (120, 0., 120.)
        var.x.rebin = 1
        var.x.overflow = True
        var.x.cumsum_reverse = False
        var.id += 1000  # prevent duplicate variable ID
        config.add_variable(var)
        var = config.get_variable("GENHBW_dilep_mass_for_eff").copy()
        var.name = "GENHBW_dilep_mass_reverse_for_eff"
        var.x.cumsum_reverse = True
        var.id += 1000  # prevent duplicate variable ID
        config.add_variable(var)

        for obj in ("lep1", "lep2", "b1", "b2"):
            # create copies of pt variables with finer binning for efficiency curves
            var = config.get_variable(f"GENHBW_{obj}_pt").copy()
            var.name = f"GENHBW_{obj}_pt_for_eff"
            var.binning = (200, 0., 50.)
            var.x.rebin = 1
            var.x.cumsum_reverse = True
            var.x.overflow = True
            var.id += 1000  # prevent duplicate variable ID
            config.add_variable(var)

            # NOTE: needs to be rerun with cumsum_reverse=False
            var = config.get_variable(f"GENHBW_{obj}_abseta").copy()
            var.name = f"GENHBW_{obj}_abseta_for_eff"
            var.binning = (200, 0., 5.)
            var.x.rebin = 1
            var.x.overflow = True
            var.x.cumsum_reverse = False
            var.id += 1000  # prevent duplicate variable ID
            config.add_variable(var)

    add_gen_hbw_decay_variables(self.config_inst)
