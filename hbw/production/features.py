# coding: utf-8

"""
Column production methods related to higher-level features.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.production.util import attach_coffea_behavior

# from columnflow.production.categories import category_ids
from hbw.production.tmp_categories import category_ids
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


@producer(
    uses={
        attach_coffea_behavior, prepare_objects, category_ids,
        bb_features, jj_features,
        "Electron.pt", "Electron.eta", "Muon.pt", "Muon.eta",
        "Jet.pt", "Jet.eta", "Jet.btagDeepFlavB",
        "Bjet.btagDeepFlavB",
        "FatJet.pt", "FatJet.tau1", "FatJet.tau2",
    },
    produces={
        attach_coffea_behavior, category_ids,
        bb_features, jj_features,
        "ht", "n_jet", "n_electron", "n_muon", "n_deepjet", "n_fatjet", "FatJet.tau21",
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # TODO: make work
    """
    # ensure coffea behavior
    custom_collections = {"Bjet": {
        "type_name": "Jet",
        "check_attr": "metric_table",
        "skip_fields": "*Idx*G",
    }}
    events = self[attach_coffea_behavior](
        events, collections=custom_collections, behavior=coffea.nanoevents.methods.nanoaod.behavior, **kwargs
    )
    """

    # add event weights
    if self.dataset_inst.is_mc:
        events = self[event_weights](events, **kwargs)

    # produce (new) category ids
    events = self[category_ids](events, **kwargs)

    # use Jets, Electrons and Muons to define Bjets, Lightjets and Lepton
    events = self[prepare_objects](events, **kwargs)

    # object padding
    events = set_ak_column(events, "Jet", ak.pad_none(events.Jet, 2))
    events = set_ak_column(events, "Lightjet", ak.pad_none(events.Lightjet, 2))
    events = set_ak_column(events, "Bjet", ak.pad_none(events.Bjet, 2))
    events = set_ak_column(events, "FatJet", ak.pad_none(events.FatJet, 1))

    # ht and number of objects (safe for None entries)
    events = set_ak_column_f32(events, "ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_jet", ak.sum(events.Jet.pt > 0, axis=1))
    events = set_ak_column(events, "n_electron", ak.sum(events.Electron.pt > 0, axis=1))
    events = set_ak_column(events, "n_muon", ak.sum(events.Muon.pt > 0, axis=1))
    wp_med = self.config_inst.x.btag_working_points.deepjet.medium
    events = set_ak_column(events, "n_deepjet", ak.sum(events.Jet.btagDeepFlavB > wp_med, axis=1))
    events = set_ak_column(events, "n_fatjet", ak.sum(events.FatJet.pt > 0, axis=1))

    # Subjettiness
    events = set_ak_column_f32(events, "FatJet.tau21", events.FatJet.tau2 / events.FatJet.tau1)

    events = self[bb_features](events, **kwargs)
    events = self[jj_features](events, **kwargs)

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

    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return

    self.uses |= {event_weights}
    self.produces |= {event_weights}
