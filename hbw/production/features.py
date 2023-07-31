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
from hbw.config.categories import add_categories_production
from hbw.config.variables import add_feature_variables

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")
# from coffea.nanoevents.methods.nanoaod import behavior

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass"},
    produces={"m_jj", "deltaR_jj"},
)
def jj_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    m_jj = (events.Jet[:, 0] + events.Jet[:, 1]).mass
    events = set_ak_column_f32(events, "m_jj", m_jj)

    deltaR_jj = events.Jet[:, 0].delta_r(events.Jet[:, 1])
    events = set_ak_column_f32(events, "deltaR_jj", deltaR_jj)

    # fill none values
    for col in self.produces:
        events = set_ak_column_f32(events, col, ak.fill_none(events[col], EMPTY_FLOAT))
    return events


@producer(
    uses={"Bjet.pt", "Bjet.eta", "Bjet.phi", "Bjet.mass"},
    produces={"m_bb", "deltaR_bb"},
)
def bb_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    m_bb = (events.Bjet[:, 0] + events.Bjet[:, 1]).mass
    events = set_ak_column_f32(events, "m_bb", m_bb)

    deltaR_bb = events.Bjet[:, 0].delta_r(events.Bjet[:, 1])
    events = set_ak_column_f32(events, "deltaR_bb", deltaR_bb)

    # fill none values
    for col in self.produces:
        events = set_ak_column_f32(events, col, ak.fill_none(events[col], EMPTY_FLOAT))

    return events
"""
@producer(
    uses={
        "Lepton.pt", "Lepton.eta", "Lepton.phi", "Lepton.mass",
        "Bjet.pt", "Bjet.eta", "Bjet.phi", "Bjet.mass", "Bjet.btagDeepFlavB",
        "MET.pt", "MET.eta", "MET.phi", "MET.mass",
        "channel_id", "m_ll",
        },
    produces={
        "deltaR_ll", "ll_pt", "m_bb", "deltaR_bb", "bb_pt",
        "MT", "min_dr_lljj", "delta_Phi", "m_lljjMET",
        "lep1_pt", "lep2_pt", "m_ll_check","m_ll",
        "m_ll", "n_bjet", "wp_score", "E_miss", "channel_id",
        },
)
def dl_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    events = set_ak_column(events, "lep1_pt", events.Lepton[:,0].pt)
    events = set_ak_column(events, "lep2_pt", events.Lepton[:,1].pt)

    bb = (events.Bjet[:, 0] + events.Bjet[:, 1])
    deltaR_bb = events.Bjet[:, 0].delta_r(events.Bjet[:, 1])
    events = set_ak_column_f32(events, "m_bb", bb.mass)
    events = set_ak_column_f32(events, "bb_pt", bb.pt)
    events = set_ak_column_f32(events, "deltaR_bb", deltaR_bb)

    ll = (events.Lepton[:, 0] + events.Lepton[:, 1])
    deltaR_ll = events.Lepton[:, 0].delta_r(events.Lepton[:, 1])
    events = set_ak_column_f32(events, "ll_pt", ll.pt)
    events = set_ak_column_f32(events, "m_ll_check", ll.mass)
    events = set_ak_column_f32(events, "deltaR_ll", deltaR_ll)

    lljj_pairs = ak.cartesian([events.Lepton, events.Bjet], axis=1)
    lep,jet = ak.unzip(lljj_pairs)
    min_dr_lljj = (ak.min(lep.delta_r(jet), axis=-1))
    #min_dr_lljj = ak.min(events.Bjet.delta_r(events.Lepton), axis=-1)
    events = set_ak_column_f32(events, "min_dr_lljj", min_dr_lljj)
    MT = (2 * events.MET.pt * ll.pt * (1 - np.cos(ll.delta_phi(events.MET)))) ** 0.5
    events = set_ak_column_f32(events, "MT", MT)
    events = set_ak_column_f32(events, "delta_Phi", abs(ll.delta_phi(bb)))
    events = set_ak_column_f32(events, "m_lljjMET", (ll + bb + 1*events.MET).mass)

    events = set_ak_column(events, "n_bjet", ak.sum(events.Bjet.pt > 0, axis=1))
    events = set_ak_column(events, "wp_score", events.Bjet.btagDeepFlavB)
    events = set_ak_column(events, "m_ll", events.m_ll)
    events = set_ak_column(events, "E_miss", events.MET[:].pt)
    events = set_ak_column(events, "m_lljjMET",(events.Bjet[:,0] + events.Bjet[:,1] + events.Lepton[:,0] + events.Lepton[:,1] + events.MET[:]).mass)
    events = set_ak_column(events, "channel_id", events.channel_id)

    # fill none values
    for col in self.produces:
        events = set_ak_column_f32(events, col, ak.fill_none(events[col], EMPTY_FLOAT))


@producer(
    uses={
        prepare_objects, category_ids, event_weights,
        dl_features, #bb_features, jj_features, dl_features,
        "Electron.pt", "Electron.eta", "Muon.pt", "Muon.eta",
        "Jet.pt", "Jet.eta", "Jet.btagDeepFlavB",
        "Lightjet.pt",
        "Bjet.btagDeepFlavB",
        "FatJet.pt", "FatJet.tau1", "FatJet.tau2",
        "Electron.charge", "Muon.charge", #"HbbJet.pt",
    },
    produces={
        category_ids, event_weights,
        dl_features, #bb_features, jj_features, dl_features,
        "ht", "n_jet", "n_electron", "n_muon", "n_deepjet", "n_fatjet", "charge", #"n_hbbjet", "FatJet.tau21",
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # add event weights
    if self.dataset_inst.is_mc:
        events = self[event_weights](events, **kwargs)

    # add behavior and define new collections (e.g. Lepton)
    events = self[prepare_objects](events, **kwargs)

    # produce (new) category ids
    events = self[category_ids](events, **kwargs)

    # object padding
    events = set_ak_column(events, "Jet", ak.pad_none(events.Jet, 2))
    events = set_ak_column(events, "Lightjet", ak.pad_none(events.Lightjet, 2))
    events = set_ak_column(events, "Bjet", ak.pad_none(events.Bjet, 2))
    events = set_ak_column(events, "FatJet", ak.pad_none(events.FatJet, 1))
    # events = set_ak_column(events, "HbbJet", ak.pad_none(events.HbbJet, 1))

    # ht and number of objects (safe for None entries)
    events = set_ak_column_f32(events, "ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_jet", ak.sum(events.Jet.pt > 0, axis=1))
    events = set_ak_column(events, "n_electron", ak.sum(events.Electron.pt > 0, axis=1))
    events = set_ak_column(events, "n_muon", ak.sum(events.Muon.pt > 0, axis=1))
    wp_med = self.config_inst.x.btag_working_points.deepjet.medium
    events = set_ak_column(events, "n_deepjet", ak.sum(events.Jet.btagDeepFlavB > wp_med, axis=1))
    events = set_ak_column(events, "n_fatjet", ak.sum(events.FatJet.pt > 0, axis=1))
    events = set_ak_column(events, "charge", (events.Lepton.charge))
    # events = set_ak_column(events, "n_hbbjet", ak.sum(events.HbbJet.pt > 0, axis=1))

    # Subjettiness
    events = set_ak_column_f32(events, "FatJet.tau21", events.FatJet.tau2 / events.FatJet.tau1)

    #events = self[bb_features](events, **kwargs)
    #events = self[jj_features](events, **kwargs)
    events = self[dl_features](events, **kwargs)
    
    # undo object padding (remove None entries)
    for obj in ["Jet", "Lightjet", "Bjet", "FatJet"]:
        events = set_ak_column(events, obj, events[obj][~ak.is_none(events[obj], axis=1)])

    return events
"""

@producer(
    uses={ 
        "Lepton.pt", "Lepton.eta", "Lepton.phi", "Lepton.mass",
        "Bjet.pt", "Bjet.eta", "Bjet.phi", "Bjet.mass",
        "MET.pt", "MET.eta", "MET.phi", "MET.mass",
        },
    produces={
        "deltaR_ll", "ll_pt", "m_bb", "deltaR_bb", "bb_pt",
        "MT", "min_dr_lljj", "delta_Phi", "m_lljjMET",
        "lep1_pt", "lep2_pt", "m_ll_check",
        },
)
def dl_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    events = set_ak_column(events, "lep1_pt", events.Lepton[:,0].pt)
    events = set_ak_column(events, "lep2_pt", events.Lepton[:,1].pt)

    bb = (events.Bjet[:, 0] + events.Bjet[:, 1])
    #m_bb = (events.Bjet[:, 0] + events.Bjet[:, 1]).mass
    deltaR_bb = events.Bjet[:, 0].delta_r(events.Bjet[:, 1])
    events = set_ak_column_f32(events, "m_bb", bb.mass)
    events = set_ak_column_f32(events, "bb_pt", bb.pt)
    events = set_ak_column_f32(events, "deltaR_bb", deltaR_bb)

    ll = (events.Lepton[:, 0] + events.Lepton[:, 1])
    deltaR_ll = events.Lepton[:, 0].delta_r(events.Lepton[:, 1])
    #events = set_ak_column_f32(events, "m_ll", ll.mass)
    events = set_ak_column_f32(events, "ll_pt", ll.pt)
    events = set_ak_column_f32(events, "m_ll_check", ll.mass)
    events = set_ak_column_f32(events, "deltaR_ll", deltaR_ll)

    lljj_pairs = ak.cartesian([events.Lepton, events.Bjet], axis=1)
    lep,jet = ak.unzip(lljj_pairs) 
    min_dr_lljj = (ak.min(lep.delta_r(jet), axis=-1))  
    #min_dr_lljj = ak.min(events.Bjet.delta_r(events.Lepton), axis=-1)
    events = set_ak_column_f32(events, "min_dr_lljj", min_dr_lljj) 
    MT = (2 * events.MET.pt * ll.pt * (1 - np.cos(ll.delta_phi(events.MET)))) ** 0.5
    events = set_ak_column_f32(events, "MT", MT)
    events = set_ak_column_f32(events, "delta_Phi", abs(ll.delta_phi(bb)))
    #events = set_ak_column_f32(events, "m_lljjMET", (ll + bb + 1*events.MET).mass)

    # fill none values
    for col in self.produces:
        events = set_ak_column_f32(events, col, ak.fill_none(events[col], EMPTY_FLOAT))

    return events


@producer(
    
    uses={
        prepare_objects, category_ids, event_weights,
        bb_features, jj_features, dl_features,
        "Electron.pt", "Electron.eta", "Muon.pt", "Muon.eta",
        "Muon.charge", "Electron.charge",
        "Jet.pt", "Jet.eta", "Jet.btagDeepFlavB",
        "Lightjet.pt",
        "Bjet.btagDeepFlavB", "Bjet.pt",
        "FatJet.pt", "FatJet.tau1", "FatJet.tau2",
        "m_ll", "MET.pt", "channel_id", #"Electron.charge",
    },
    produces={
        category_ids, event_weights,
        dl_features,
        "ht", "n_jet", "n_electron", "n_muon", "n_deepjet", "n_fatjet", 
        "FatJet.tau21", "channel_id", "m_ll",
        "lep1_pt", "lep2_pt", "E_miss", "m_lljjMET", "charge", "n_bjet", "wp_score",
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    
    # add event weights
    if self.dataset_inst.is_mc:
        events = self[event_weights](events, **kwargs)

    # add behavior and define new collections (e.g. Lepton)
    events = self[prepare_objects](events, **kwargs)

    # produce (new) category ids
    events = self[category_ids](events, **kwargs)

    # object padding
    events = set_ak_column(events, "Jet", ak.pad_none(events.Jet, 2))
    events = set_ak_column(events, "Lightjet", ak.pad_none(events.Lightjet, 2))
    events = set_ak_column(events, "Bjet", ak.pad_none(events.Bjet, 2))
    events = set_ak_column(events, "n_bjet", ak.sum(events.Bjet.pt > 0, axis=1))
    events = set_ak_column(events, "wp_score", events.Bjet.btagDeepFlavB)
    events = set_ak_column(events, "FatJet", ak.pad_none(events.FatJet, 1))
    #events = set_ak_column(events, "HbbJet", ak.pad_none(events.HbbJet, 1))

    # ht and number of objects (safe for None entries)
    events = set_ak_column_f32(events, "ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_jet", ak.sum(events.Jet.pt > 0, axis=1))
    events = set_ak_column(events, "n_electron", ak.sum(events.Electron.pt > 0, axis=1))
    events = set_ak_column(events, "n_muon", ak.sum(events.Muon.pt > 0, axis=1))
    wp_med = self.config_inst.x.btag_working_points.deepjet.medium
    events = set_ak_column(events, "n_deepjet", ak.sum(events.Jet.btagDeepFlavB > wp_med, axis=1))
    events = set_ak_column(events, "n_fatjet", ak.sum(events.FatJet.pt > 0, axis=1))
    events = set_ak_column(events, "channel_id", events.channel_id)
    events = set_ak_column(events, "m_ll", events.m_ll)
    events = set_ak_column(events, "charge", (events.Lepton.charge))
    events = set_ak_column(events, "lep1_pt", events.Lepton[:,0].pt)
    events = set_ak_column(events, "lep2_pt", events.Lepton[:,1].pt)
    events = set_ak_column(events, "E_miss", events.MET[:].pt)
    events = set_ak_column(events, "m_lljjMET",(events.Bjet[:,0] + events.Bjet[:,1] + events.Lepton[:,0] + events.Lepton[:,1] + events.MET[:]).mass)
    # events = set_ak_column(events, "n_hbbjet", ak.sum(events.HbbJet.pt > 0, axis=1))

    # Subjettiness
    events = set_ak_column_f32(events, "FatJet.tau21", events.FatJet.tau2 / events.FatJet.tau1)

    events = self[dl_features](events, **kwargs)

    # undo object padding (remove None entries)
    for obj in ["Jet", "Lightjet", "Bjet", "FatJet"]:
        events = set_ak_column(events, obj, events[obj][~ak.is_none(events[obj], axis=1)])

    return events

@features.init
def features_init(self: Producer) -> None:
    if self.config_inst.x("add_categories_production", True):
        # add categories but only on first call
        add_categories_production(self.config_inst)
        self.config_inst.x.add_categories_production = False

    if self.config_inst.x("add_feature_variables", True):
        # add variable instances but only on first call
        add_feature_variables(self.config_inst)
        self.config_inst.x.add_feature_variables = False
