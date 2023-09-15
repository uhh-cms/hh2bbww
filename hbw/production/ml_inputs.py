# coding: utf-8

"""
Column production methods related to higher-level features.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.production.categories import category_ids

from hbw.production.weights import event_weights
from hbw.production.prepare_objects import prepare_objects
from hbw.config.ml_variables import add_ml_variables
from hbw.config.categories import add_categories_production
from hbw.util import four_vec
ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        category_ids, event_weights,
        prepare_objects,
        "Electron.charge", "Muon.charge",
        "HbbJet.msoftdrop", "HbbJet.deepTagMD_HbbvsQCD",
        "Jet.btagDeepFlavB", "Bjet.btagDeepFlavB", "Lightjet.btagDeepFlavB",
    } | four_vec(
        {"Electron", "Muon", "MET", "Jet", "Bjet", "Lightjet", "HbbJet"},
    ),
    produces={
        category_ids, event_weights,
        # other produced columns set in the init function
    },
)
def ml_inputs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # add event weights
    if self.dataset_inst.is_mc:
        events = self[event_weights](events, **kwargs)

    # add behavior and define new collections (e.g. Lepton)
    events = self[prepare_objects](events, **kwargs)

    # produce (new) category ids
    events = self[category_ids](events, **kwargs)

    # object padding
    events = set_ak_column(events, "Lightjet", ak.pad_none(events.Lightjet, 2))
    events = set_ak_column(events, "Bjet", ak.pad_none(events.Bjet, 2))
    # events = set_ak_column(events, "FatJet", ak.pad_none(events.FatJet, 1))
    events = set_ak_column(events, "HbbJet", ak.pad_none(events.HbbJet, 1))

    # low-level features
    # TODO: this could be more generalized
    for var in ["pt", "eta"]:
        events = set_ak_column_f32(events, f"mli_b1_{var}", events.Bjet[:, 0][var])
        events = set_ak_column_f32(events, f"mli_b2_{var}", events.Bjet[:, 1][var])
        events = set_ak_column_f32(events, f"mli_j1_{var}", events.Lightjet[:, 0][var])
        events = set_ak_column_f32(events, f"mli_j2_{var}", events.Lightjet[:, 1][var])
        events = set_ak_column_f32(events, f"mli_lep_{var}", events.Lepton[:, 0][var])
        events = set_ak_column_f32(events, f"mli_met_{var}", events.MET[var])

    # H->bb FatJet
    for var in ["pt", "eta", "phi", "mass", "msoftdrop", "deepTagMD_HbbvsQCD"]:
        events = set_ak_column_f32(events, f"mli_fj_{var}", events.HbbJet[:, 0][var])

    # jets in general
    events = set_ak_column_f32(events, "mli_ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column_f32(events, "mli_n_jet", ak.num(events.Jet.pt, axis=1))

    # all possible jet pairs
    jet_pairs = ak.combinations(events.Jet, 2)
    dr = jet_pairs[:, :, "0"].delta_r(jet_pairs[:, :, "1"])
    events = set_ak_column_f32(events, "mindr_jj", ak.min(dr, axis=1))

    # bjets in general
    wp_med = self.config_inst.x.btag_working_points.deepjet.medium
    events = set_ak_column_f32(events, "mli_n_deepjet", ak.num(events.Jet[events.Jet.btagDeepFlavB > wp_med], axis=1))
    events = set_ak_column_f32(events, "mli_deepjetsum", ak.sum(events.Jet.btagDeepFlavB, axis=1))
    events = set_ak_column_f32(events, "mli_b_deepjetsum", ak.sum(events.Bjet.btagDeepFlavB, axis=1))
    events = set_ak_column_f32(events, "mli_l_deepjetsum", ak.sum(events.Lightjet.btagDeepFlavB, axis=1))

    # hbb features
    events = set_ak_column_f32(events, "mli_dr_bb", events.Bjet[:, 0].delta_r(events.Bjet[:, 1]))
    events = set_ak_column_f32(events, "mli_dphi_bb", abs(events.Bjet[:, 0].delta_phi(events.Bjet[:, 1])))

    hbb = events.Bjet[:, 0] + events.Bjet[:, 1]
    events = set_ak_column_f32(events, "mli_mbb", hbb.mass)

    mindr_lb = ak.min(events.Bjet.delta_r(events.Lepton[:, 0]), axis=-1)
    events = set_ak_column_f32(events, "mli_mindr_lb", mindr_lb)

    # wjj features
    events = set_ak_column_f32(events, "mli_dr_jj", events.Lightjet[:, 0].delta_r(events.Lightjet[:, 1]))
    events = set_ak_column_f32(events, "mli_dphi_jj", abs(events.Lightjet[:, 0].delta_phi(events.Lightjet[:, 1])))

    wjj = events.Lightjet[:, 0] + events.Lightjet[:, 1]
    events = set_ak_column_f32(events, "mli_mjj", wjj.mass)

    mindr_lj = ak.min(events.Lightjet.delta_r(events.Lepton[:, 0]), axis=-1)
    events = set_ak_column_f32(events, "mli_mindr_lj", mindr_lj)

    # wlnu features
    wlnu = events.MET + events.Lepton[:, 0]
    events = set_ak_column_f32(events, "mli_dphi_lnu", abs(events.Lepton[:, 0].delta_phi(events.MET)))
    # NOTE: this column can be set to nan value
    events = set_ak_column_f32(events, "mli_mlnu", wlnu.mass)

    # hww features
    hww = wlnu + wjj
    hww_vis = events.Lepton[:, 0] + wjj

    events = set_ak_column_f32(events, "mli_mjjlnu", hww.mass)
    events = set_ak_column_f32(events, "mli_mjjl", hww_vis.mass)

    # angles
    events = set_ak_column_f32(events, "mli_dphi_bb_jjlnu", abs(hbb.delta_phi(hww)))
    events = set_ak_column_f32(events, "mli_dr_bb_jjlnu", hbb.delta_r(hww))

    events = set_ak_column_f32(events, "mli_dphi_bb_jjl", abs(hbb.delta_phi(hww_vis)))
    events = set_ak_column_f32(events, "mli_dr_bb_jjl", hbb.delta_r(hww_vis))

    events = set_ak_column_f32(events, "mli_dphi_bb_nu", abs(hbb.delta_phi(events.MET)))
    events = set_ak_column_f32(events, "mli_dphi_jj_nu", abs(wjj.delta_phi(events.MET)))
    events = set_ak_column_f32(events, "mli_dr_bb_l", hbb.delta_r(events.MET))
    events = set_ak_column_f32(events, "mli_dr_jj_l", hbb.delta_r(events.MET))

    # hh features
    hh = hbb + hww
    hh_vis = hbb + hww_vis

    events = set_ak_column_f32(events, "mli_mbbjjlnu", hh.mass)
    events = set_ak_column_f32(events, "mli_mbbjjl", hh_vis.mass)

    s_min = (
        2 * events.MET.pt * ((hh_vis.mass ** 2 + hh_vis.energy ** 2) ** 0.5 -
        hh_vis.pt * np.cos(hh_vis.delta_phi(events.MET)) + hh_vis.mass ** 2)
    ) ** 0.5
    events = set_ak_column_f32(events, "mli_s_min", s_min)

    # create ll object and ll variables
    ll = (events.Lepton[:, 0] + events.Lepton[:, 1])
    deltaR_ll = events.Lepton[:, 0].delta_r(events.Lepton[:, 1])
    # events = set_ak_column_f32(events, "mli_ll_pt", ll.pt)
    events = set_ak_column_f32(events, "mli_mll", ll.mass)
    events = set_ak_column_f32(events, "mli_dr_ll", deltaR_ll)

    # minimum deltaR between lep and jet
    lljj_pairs = ak.cartesian([events.Lepton, events.Bjet], axis=1)
    lep, jet = ak.unzip(lljj_pairs)
    min_dr_lljj = (ak.min(lep.delta_r(jet), axis=-1))
    events = set_ak_column_f32(events, "mli_min_dr_llbb", min_dr_lljj)

    # bb pt 
    events = set_ak_column_f32(events, "mli_bb_pt", hbb.pt)
    
    # fill nan/none values of all produced columns
    for col in self.ml_columns:
        events = set_ak_column(events, col, ak.fill_none(ak.nan_to_none(events[col]), EMPTY_FLOAT))

    return events


@ml_inputs.init
def ml_inputs_init(self: Producer) -> None:
    # define ML input separately to self.produces
    self.ml_columns = {
        "mli_mll", "mli_dr_ll", "mli_min_dr_llbb", "mli_bb_pt",
        "mli_ht", "mli_n_jet", "mli_n_deepjet",
        "mli_deepjetsum", "mli_b_deepjetsum", "mli_l_deepjetsum",
        "mli_dr_bb", "mli_dphi_bb", "mli_mbb", "mli_mindr_lb", "mli_dr_jj", "mli_dphi_jj", "mli_mjj", "mli_mindr_lj",
        "mli_dphi_lnu", "mli_mlnu", "mli_mjjlnu", "mli_mjjl", "mli_dphi_bb_jjlnu", "mli_dr_bb_jjlnu",
        "mli_dphi_bb_jjl", "mli_dr_bb_jjl", "mli_dphi_bb_nu", "mli_dphi_jj_nu", "mli_dr_bb_l", "mli_dr_jj_l",
        "mli_mbbjjlnu", "mli_mbbjjl", "mli_s_min",
    } | set(
        f"mli_{obj}_{var}"
        for obj in ["b1", "b2", "j1", "j2", "lep", "met"]
        for var in ["pt", "eta"]
    ) | set(
        f"mli_{obj}_{var}"
        for obj in ["fj"]
        for var in ["pt", "eta", "phi", "mass", "msoftdrop", "deepTagMD_HbbvsQCD"]
    )
    self.produces |= self.ml_columns

    # add categories to config
    add_categories_production(self.config_inst)

    # add variable instances to config
    add_ml_variables(self.config_inst)
