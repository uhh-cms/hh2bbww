# coding: utf-8

"""
Column production methods related to higher-level features.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT

from hbw.production.prepare_objects import prepare_objects
from hbw.config.variables import add_feature_variables
from hbw.config.dl.variables import add_dl_variables
from hbw.util import four_vec

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")
# from coffea.nanoevents.methods.nanoaod import behavior

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses=four_vec("Jet"),
    produces={"m_jj", "jj_pt", "deltaR_jj"},
)
def jj_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # create jj features
    jj = (events.Jet[:, 0] + events.Jet[:, 1])
    deltaR_jj = events.Jet[:, 0].delta_r(events.Jet[:, 1])
    events = set_ak_column_f32(events, "m_jj", jj.mass)
    events = set_ak_column_f32(events, "jj_pt", jj.pt)
    events = set_ak_column_f32(events, "deltaR_jj", deltaR_jj)

    # fill none values
    for col in self.produces:
        events = set_ak_column_f32(events, col, ak.fill_none(events[col], EMPTY_FLOAT))
    return events


@producer(
    uses={
        "HbbJet.msoftdrop",
    } | four_vec("Bjet"),
    produces={"m_bb", "bb_pt", "deltaR_bb", "m_bb_combined"},
)
def bb_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # create bb features
    bb = (events.Bjet[:, 0] + events.Bjet[:, 1])
    events = set_ak_column_f32(events, "m_bb", bb.mass)
    events = set_ak_column_f32(events, "bb_pt", bb.pt)

    deltaR_bb = events.Bjet[:, 0].delta_r(events.Bjet[:, 1])
    events = set_ak_column_f32(events, "deltaR_bb", deltaR_bb)

    # combination of resolved and boosted bb mass
    m_bb_combined = ak.where(ak.num(events.HbbJet) > 0, events.HbbJet[:, 0].msoftdrop, bb.mass)
    events = set_ak_column_f32(events, "m_bb_combined", m_bb_combined)

    # fill none values
    for col in self.produces:
        events = set_ak_column_f32(events, col, ak.fill_none(events[col], EMPTY_FLOAT))

    return events


@producer(
    uses={
        prepare_objects,
        bb_features, jj_features,
        "Electron.pt", "Electron.eta", "Muon.pt", "Muon.eta",
        "Muon.charge", "Electron.charge",
        "Jet.pt", "Jet.eta", "Jet.btagDeepFlavB", "Jet.btagPNetB",
        "Bjet.pt",
        "HbbJet.pt",
        "FatJet.pt", "FatJet.tau1", "FatJet.tau2",
    },
    produces={
        bb_features, jj_features,
        "ht", "n_jet", "n_electron", "n_muon", "n_deepjet", "n_fatjet", "n_hbbjet",
        "FatJet.tau21", "n_bjet",
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # add behavior and define new collections (e.g. Lepton)
    events = self[prepare_objects](events, **kwargs)

    # object padding
    events = set_ak_column(events, "Jet", ak.pad_none(events.Jet, 2))
    events = set_ak_column(events, "Bjet", ak.pad_none(events.Bjet, 2))
    events = set_ak_column(events, "FatJet", ak.pad_none(events.FatJet, 1))
    events = set_ak_column(events, "HbbJet", ak.pad_none(events.HbbJet, 1))

    # ht and number of objects (safe for None entries)
    events = set_ak_column_f32(events, "ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_jet", ak.sum(events.Jet.pt > 0, axis=1))
    events = set_ak_column(events, "n_bjet", ak.sum(events.Bjet.pt > 0, axis=1))
    events = set_ak_column(events, "n_electron", ak.sum(events.Electron.pt > 0, axis=1))
    events = set_ak_column(events, "n_muon", ak.sum(events.Muon.pt > 0, axis=1))
    wp_med_deepjet = self.config_inst.x.btag_working_points.deepjet.medium
    events = set_ak_column(events, "n_deepjet", ak.sum(events.Jet.btagDeepFlavB > wp_med_deepjet, axis=1))
    wp_med_particlenet = self.config_inst.x.btag_working_points.particlenet.medium
    events = set_ak_column(events, "n_particlenet", ak.sum(events.Jet.btagPNetB > wp_med_particlenet, axis=1))
    events = set_ak_column(events, "n_fatjet", ak.sum(events.FatJet.pt > 0, axis=1))
    events = set_ak_column(events, "n_hbbjet", ak.sum(events.HbbJet.pt > 0, axis=1))

    # Subjettiness
    events = set_ak_column_f32(events, "FatJet.tau21", events.FatJet.tau2 / events.FatJet.tau1)

    # bb and jj features
    events = self[bb_features](events, **kwargs)
    events = self[jj_features](events, **kwargs)

    # undo object padding (remove None entries)
    for obj in ["Jet", "Bjet", "FatJet"]:
        events = set_ak_column(events, obj, events[obj][~ak.is_none(events[obj], axis=1)])

    return events


@features.init
def features_init(self: Producer) -> None:
    # add variable instances to config
    add_feature_variables(self.config_inst)


@producer(
    uses=four_vec({"Electron", "Muon", "Bjet", "MET"}) | {
        features,
        "Electron.charge", "Muon.charge",
    },
    produces={
        features,
        "deltaR_ll", "ll_pt", "m_bb", "deltaR_bb", "bb_pt",
        "MT", "min_dr_lljj", "delta_Phi", "m_lljjMET",
        "m_ll_check", "E_miss", "charge", "wp_score",
    },
)
def dl_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # Inherit common features and prepares Object Lepton. Bjet, etc.
    events = self[features](events, **kwargs)

    # create ll object and ll variables
    ll = (events.Lepton[:, 0] + events.Lepton[:, 1])
    deltaR_ll = events.Lepton[:, 0].delta_r(events.Lepton[:, 1])
    events = set_ak_column_f32(events, "ll_pt", ll.pt)
    events = set_ak_column_f32(events, "m_ll_check", ll.mass)
    events = set_ak_column_f32(events, "deltaR_ll", deltaR_ll)

    # minimum deltaR between lep and jet
    lljj_pairs = ak.cartesian([events.Lepton, events.Bjet], axis=1)
    lep, jet = ak.unzip(lljj_pairs)
    min_dr_lljj = (ak.min(lep.delta_r(jet), axis=-1))
    events = set_ak_column_f32(events, "min_dr_lljj", min_dr_lljj)

    # Transverse mass
    MT = (2 * events.MET.pt * ll.pt * (1 - np.cos(ll.delta_phi(events.MET)))) ** 0.5
    events = set_ak_column_f32(events, "MT", MT)

    # delta Phi between ll and bb object
    bb = (events.Bjet[:, 0] + events.Bjet[:, 1])
    events = set_ak_column_f32(events, "delta_Phi", abs(ll.delta_phi(bb)))

    # invariant mass of all decay products
    m_lljjMET = (events.Bjet[:, 0] + events.Bjet[:, 1] + events.Lepton[:, 0] + events.Lepton[:, 1] + events.MET[:]).mass
    events = set_ak_column(events, "m_lljjMET", m_lljjMET)

    # Lepton charge
    events = set_ak_column(events, "charge", (events.Lepton.charge))

    # fill none values for dl variables
    dl_variable_list = [
        "m_bb", "bb_pt", "deltaR_bb", "ll_pt", "m_ll_check", "deltaR_ll", "min_dr_lljj",
        "charge", "MT", "delta_Phi", "E_miss", "m_lljjMET",
    ]
    for var in dl_variable_list:
        events = set_ak_column_f32(events, var, ak.fill_none(events[var], EMPTY_FLOAT))

    return events


@dl_features.init
def dl_features_init(self: Producer) -> None:
    # add variable instances to config
    add_dl_variables(self.config_inst)


from hbw.production.resonant_features import resonant_features


@producer(
    uses={
        features, resonant_features,
    },
    produces={
        features, resonant_features,
    },
)
def sl_res_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # Inherit common features and prepares Object Lepton. Bjet, etc.
    events = self[features](events, **kwargs)
    events = self[resonant_features](events, **kwargs)

    return events
