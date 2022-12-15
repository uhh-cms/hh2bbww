# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.production.util import attach_coffea_behavior

from hbw.production.weights import event_weights
from hbw.production.prepare_objects import prepare_objects
from hbw.selection.general import jet_energy_shifts

ak = maybe_import("awkward")
np = maybe_import("numpy")


@producer(
    uses={
        event_weights, prepare_objects.USES, prepare_objects.PRODUCES,
    },
    produces={
        attach_coffea_behavior, event_weights,
        # explicitly save Lepton fields for ML and plotting since they don't exist in ReduceEvents output
        "Lepton.pt", "Lepton.eta", "Lepton.phi", "Lepton.mass", "Lepton.charge", "Lepton.pdgId",
    },
    shifts={
        jet_energy_shifts,
    },
)
def ml_inputs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # use Jets, Electrons and Muons to define Bjets, Lightjets and Lepton
    events = self[prepare_objects](events, **kwargs)

    # jets in general
    events = set_ak_column(events, "mli_ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "mli_n_jet", ak.num(events.Jet.pt, axis=1))

    # all possible jet pairs
    jet_pairs = ak.combinations(events.Jet, 2)
    dr = jet_pairs[:, :, "0"].delta_r(jet_pairs[:, :, "1"])
    events = set_ak_column(events, "mindr_jj", ak.min(dr, axis=1))

    # bjets in general
    wp_med = self.config_inst.x.btag_working_points.deepjet.medium
    events = set_ak_column(events, "mli_n_deepjet", ak.num(events.Jet[events.Jet.btagDeepFlavB > wp_med], axis=1))
    events = set_ak_column(events, "mli_deepjetsum", ak.sum(events.Jet.btagDeepFlavB, axis=1))
    events = set_ak_column(events, "mli_b_deepjetsum", ak.sum(events.Bjet.btagDeepFlavB, axis=1))
    events = set_ak_column(events, "mli_l_deepjetsum", ak.sum(events.Lightjet.btagDeepFlavB, axis=1))

    # hbb features
    events = set_ak_column(events, "mli_dr_bb", events.Bjet[:, 0].delta_r(events.Bjet[:, 1]))
    events = set_ak_column(events, "mli_dphi_bb", abs(events.Bjet[:, 0].delta_phi(events.Bjet[:, 1])))

    hbb = events.Bjet[:, 0] + events.Bjet[:, 1]
    events = set_ak_column(events, "mli_mbb", hbb.mass)

    mindr_lb = ak.min(events.Bjet.delta_r(events.Lepton[:, 0]), axis=-1)
    events = set_ak_column(events, "mli_mindr_lb", mindr_lb)

    # wjj features
    events = set_ak_column(events, "mli_dr_jj", events.Lightjet[:, 0].delta_r(events.Lightjet[:, 1]))
    events = set_ak_column(events, "mli_dphi_jj", abs(events.Lightjet[:, 0].delta_phi(events.Lightjet[:, 1])))

    wjj = events.Lightjet[:, 0] + events.Lightjet[:, 1]
    events = set_ak_column(events, "mli_mjj", wjj.mass)

    mindr_lj = ak.min(events.Lightjet.delta_r(events.Lepton[:, 0]), axis=-1)
    events = set_ak_column(events, "mli_mindr_lj", mindr_lj)

    # wlnu features
    wlnu = events.MET + events.Lepton[:, 0]
    events = set_ak_column(events, "mli_dphi_lnu", abs(events.Lepton[:, 0].delta_phi(events.MET)))
    events = set_ak_column(events, "mli_mlnu", wlnu.mass)

    # hww features
    hww = wlnu + wjj
    hww_vis = events.Lepton[:, 0] + wjj

    events = set_ak_column(events, "mli_mjjlnu", hww.mass)
    events = set_ak_column(events, "mli_mjjl", hww_vis.mass)

    # angles
    events = set_ak_column(events, "mli_dphi_bb_jjlnu", abs(hbb.delta_phi(hww)))
    events = set_ak_column(events, "mli_dr_bb_jjlnu", hbb.delta_r(hww))

    events = set_ak_column(events, "mli_dphi_bb_jjl", abs(hbb.delta_phi(hww_vis)))
    events = set_ak_column(events, "mli_dr_bb_jjl", hbb.delta_r(hww_vis))

    events = set_ak_column(events, "mli_dphi_bb_nu", abs(hbb.delta_phi(events.MET)))
    events = set_ak_column(events, "mli_dphi_jj_nu", abs(wjj.delta_phi(events.MET)))
    events = set_ak_column(events, "mli_dr_bb_l", hbb.delta_r(events.MET))
    events = set_ak_column(events, "mli_dr_jj_l", hbb.delta_r(events.MET))

    # hh features
    hh = hbb + hww
    hh_vis = hbb + hww_vis

    events = set_ak_column(events, "mli_mbbjjlnu", hh.mass)
    events = set_ak_column(events, "mli_mbbjjl", hh_vis.mass)

    s_min = (
        2 * events.MET.pt * ((hh_vis.mass ** 2 + hh_vis.energy ** 2) ** 0.5 -
        hh_vis.pt * np.cos(hh_vis.delta_phi(events.MET)) + hh_vis.mass ** 2)
    ) ** 0.5
    events = set_ak_column(events, "mli_s_min", s_min)

    # fill none values of all produced columns
    for col in self.ml_columns:
        events = set_ak_column(events, col, ak.fill_none(events[col], EMPTY_FLOAT))

    # add event weights
    events = self[event_weights](events, **kwargs)

    return events


@ml_inputs.init
def ml_inputs_init(self: Producer) -> None:
    # define ML input separately to self.produces
    self.ml_columns = {
        "mli_ht", "mli_n_jet", "mli_n_deepjet",
        "mli_deepjetsum", "mli_b_deepjetsum", "mli_l_deepjetsum",
        "mli_dr_bb", "mli_dphi_bb", "mli_mbb", "mli_mindr_lb", "mli_dr_jj", "mli_dphi_jj", "mli_mjj", "mli_mindr_lj",
        "mli_dphi_lnu", "mli_mlnu", "mli_mjjlnu", "mli_mjjl", "mli_dphi_bb_jjlnu", "mli_dr_bb_jjlnu",
        "mli_dphi_bb_jjl", "mli_dr_bb_jjl", "mli_dphi_bb_nu", "mli_dphi_jj_nu", "mli_dr_bb_l", "mli_dr_jj_l",
        "mli_mbbjjlnu", "mli_mbbjjl", "mli_s_min",
    }
    self.produces |= self.ml_columns
