# coding: utf-8

"""
Column production method
"""

import functools
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from hbw.util import four_vec
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
np = maybe_import("numpy")
maybe_import("coffea.nanoevents.methods.nanoaod")


set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses=four_vec({"Electron", "Muon", "Bjet", "MET", "Lightjet"}),
    produces={
        "eta_Whadron", "eta_Wlepton", "eta_Higgs_WW", "eta_Higgs_bb", "eta_Higgs_bb",
        "phi_Whadron", "phi_Wlepton", "phi_Higgs_WW", "phi_Higgs_bb", "phi_Heavy_Higgs",
        "pt_Whadron", "pt_Wlepton", "pt_Higgs_WW", "pt_Higgs_bb", "pt_Heavy_Higgs",
        "m_Whadron", "m_Wlepton", "m_Higgs_WW", "m_Higgs_bb", "m_Heavy_Higgs",
        # TODO: only store objects instead of individual columns
        # "Whadron.*", "Wlepton.*", "Higgs_WW.*", "Higgs_bb.*", "Heavy_Higgs.*",
    },
)
def resonant_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    #         object of resonans 2Higgs analysis
    #
    #                           _    b
    #                    Higgs_bb  /
    #                  - - - - - -
    #                /             \ _
    #               /                b
    #  Heavy_Higgs /                          l
    # - - - - - - -                 Wlepton  /
    #              \                ---------
    #               \              /         \
    #                \  Higgs_WW  /           nu
    #                  - - - - - -
    #                             \              q
    #                              \  Whadron   /
    #                               ------------
    #                                           \
    #                                            q'
    #

    # object padding
    events = set_ak_column(events, "Jet", ak.pad_none(events.Jet, 2))
    events = set_ak_column(events, "Bjet", ak.pad_none(events.Bjet, 2))
    events = set_ak_column(events, "FatJet", ak.pad_none(events.FatJet, 1))
    events = set_ak_column(events, "HbbJet", ak.pad_none(events.HbbJet, 1))
    events = set_ak_column(events, "Lightjet", ak.pad_none(events.Lightjet, 2))

    if "Whadron" not in events.fields:
        events = set_ak_column(events, "Whadron", events.Lightjet[:, 0] + events.Lightjet[:, 1])
    if "Wlepton" not in events.fields:
        events = set_ak_column(events, "Wlepton", events.Lepton[:, 0] + events.MET[:])
    if "Higgs_WW" not in events.fields:
        events = set_ak_column(events, "Higgs_WW", events.Whadron[:] + events.Wlepton[:])
    if "Higgs_bb" not in events.fields:
        events = set_ak_column(events, "Higgs_bb", events.Bjet[:, 0] + events.Bjet[:, 1])
    if "Teilchen_1" not in events.fields:
        events = set_ak_column(events, "Teilchen_1", events.Bjet[:, 0] + events.Whadron)
    if "Teilchen_2" not in events.fields:
        events = set_ak_column(events, "Teilchen_2", events.Bjet[:, 1] + events.Whadron)
    if "Heavy_Higgs" not in events.fields:
        events = set_ak_column(events, "Heavy_Higgs", events.Higgs_WW + events.Higgs_bb)
    # variables of objects (don't forget to describe them in variables.py and features.py in produces)
    # Wlepton
    events = set_ak_column_f32(events, "pt_Wlepton", events.Wlepton.pt)
    events = set_ak_column_f32(events, "m_Wlepton", events.Wlepton.mass)
    events = set_ak_column_f32(events, "phi_Wlepton", events.Wlepton.phi)
    events = set_ak_column_f32(events, "eta_Wlepton", events.Wlepton.eta)
    # Whadron
    events = set_ak_column_f32(events, "pt_Whadron", events.Whadron.pt)
    events = set_ak_column_f32(events, "m_Whadron", events.Whadron.mass)
    events = set_ak_column_f32(events, "phi_Whadron", events.Whadron.phi)
    events = set_ak_column_f32(events, "eta_Whadron", events.Whadron.eta)
    # Higgs_WW
    events = set_ak_column_f32(events, "pt_Higgs_WW", events.Higgs_WW.pt)
    events = set_ak_column_f32(events, "m_Higgs_WW", events.Higgs_WW.mass)
    events = set_ak_column_f32(events, "eta_Higgs_WW", events.Higgs_WW.eta)
    events = set_ak_column_f32(events, "phi_Higgs_WW", events.Higgs_WW.phi)
    # Higgs_bb
    events = set_ak_column_f32(events, "pt_Higgs_bb", events.Higgs_bb.pt)
    events = set_ak_column_f32(events, "m_Higgs_bb", events.Higgs_bb.mass)
    events = set_ak_column_f32(events, "eta_Higgs_bb", events.Higgs_bb.eta)
    events = set_ak_column_f32(events, "phi_Higgs_bb", events.Higgs_bb.phi)
    # Heavy_Higgs
    events = set_ak_column_f32(events, "pt_Heavy_Higgs", events.Heavy_Higgs.pt)
    events = set_ak_column_f32(events, "m_Heavy_Higgs", events.Heavy_Higgs.mass)
    events = set_ak_column_f32(events, "eta_Heavy_Higgs", events.Heavy_Higgs.eta)
    events = set_ak_column_f32(events, "phi_Heavy_Higgs", events.Heavy_Higgs.phi)

    for col in self.produces:
        events = set_ak_column(events, col, ak.fill_none(ak.nan_to_none(events[col]), EMPTY_FLOAT))

    # undo object padding
    for obj in ["Jet", "Lightjet", "Bjet", "FatJet"]:
        events = set_ak_column(events, obj, events[obj][~ak.is_none(events[obj], axis=1)])
    return events


from hbw.config.sl_res.variables import add_resonant_variables


@resonant_features.init
def sl_res_features_init(self: Producer) -> None:
    # add variable instances to config
    add_resonant_variables(self.config_inst)
