# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.production.util import attach_coffea_behavior

from hbw.production.weights import event_weights
from hbw.selection.general import jet_energy_shifts

ak = maybe_import("awkward")
coffea = maybe_import("coffea")
np = maybe_import("numpy")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={
        attach_coffea_behavior,  # TODO use
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",  # "Electron.charge", "Electron.pdgId",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",  # "Muon.charge", "Muon.pdgId",
        "MET.pt", "MET.phi",
    },
    produces={"Bjet", "Lightjet", "Lepton", "MET"},
    shifts={jet_energy_shifts},
)
def prepare_objects(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # test that the number of particles is as expected after selection
    if ak.any(ak.num(events.Electron) + ak.num(events.Muon) != 1):
        raise Exception("Number of Leptons not 1")
    if ak.any(ak.num(events.Jet) < 3):
        raise Exception("Number of Jets smaller 3")

    # sort jets after b-score and define b-jets as the two b-score leading jets (bscore-sorted)
    bjet_indices = ak.argsort(events.Jet.btagDeepFlavB, axis=-1, ascending=False)
    events = set_ak_column(events, "Bjet", events.Jet[bjet_indices[:, :2]])

    # define all remaining jets as lightjets (pt-sorted) and pad them to a minimum-length of 2
    lightjets = events.Jet[bjet_indices[:, 2:]]
    lightjets = lightjets[ak.argsort(lightjets.pt, axis=-1, ascending=False)]
    events = set_ak_column(events, "Lightjet", ak.pad_none(lightjets, 2))

    # combine Electron and Muon into a single object (Lepton)
    lepton = ak.concatenate([events.Muon, events.Electron], axis=-1)[:, 0]
    events = set_ak_column(events, "Lepton", lepton)

    # transform MET into 4-vector
    events["MET"] = set_ak_column(events.MET, "mass", 0)
    events["MET"] = set_ak_column(events.MET, "eta", 0)

    # 4-vector behavior
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    for obj in ["Jet", "Bjet", "Lightjet", "Muon", "Electron", "Lepton", "MET"]:
        events[obj] = ak.with_name(events[obj], "PtEtaPhiMLorentzVector")

    return events


@producer(
    uses={
        event_weights, prepare_objects.USES, prepare_objects.PRODUCES,
    },
    produces={
        attach_coffea_behavior, event_weights,
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

    # bjets in general
    wp_med = self.config_inst.x.btag_working_points.deepjet.medium
    events = set_ak_column(events, "mli_n_deepjet", ak.num(events.Jet[events.Jet.btagDeepFlavB > wp_med], axis=1))
    events = set_ak_column(events, "mli_deepjetsum", ak.sum(events.Jet.btagDeepFlavB, axis=1))

    # hbb features
    events = set_ak_column(events, "mli_dr_bb", events.Bjet[:, 0].delta_r(events.Bjet[:, 1]))
    events = set_ak_column(events, "mli_dphi_bb", abs(events.Bjet[:, 0].delta_phi(events.Bjet[:, 1])))

    hbb = events.Bjet[:, 0] + events.Bjet[:, 1]
    events = set_ak_column(events, "mli_mbb", hbb.mass)

    mindr_lb = ak.min(events.Bjet.delta_r(events.Lepton), axis=-1)
    events = set_ak_column(events, "mli_mindr_lb", mindr_lb)

    # wjj features
    events = set_ak_column(events, "mli_dr_jj", events.Lightjet[:, 0].delta_r(events.Lightjet[:, 1]))
    events = set_ak_column(events, "mli_dphi_jj", abs(events.Lightjet[:, 0].delta_phi(events.Lightjet[:, 1])))

    wjj = events.Lightjet[:, 0] + events.Lightjet[:, 1]
    events = set_ak_column(events, "mli_mjj", wjj.mass)

    mindr_lj = ak.min(events.Lightjet.delta_r(events.Lepton), axis=-1)
    events = set_ak_column(events, "mli_mindr_lj", mindr_lj)

    # wlnu features
    wlnu = events.MET + events.Lepton
    events = set_ak_column(events, "mli_dphi_lnu", abs(events.Lepton.delta_phi(events.MET)))
    events = set_ak_column(events, "mli_mlnu", wlnu.mass)

    # hww features
    hww = wlnu + wjj
    hww_vis = events.Lepton + wjj

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
        "mli_ht", "mli_n_jet", "mli_n_deepjet", "mli_deepjetsum",
        "mli_dr_bb", "mli_dphi_bb", "mli_mbb", "mli_mindr_lb", "mli_dr_jj", "mli_dphi_jj", "mli_mjj", "mli_mindr_lj",
        "mli_dphi_lnu", "mli_mlnu", "mli_mjjlnu", "mli_mjjl", "mli_dphi_bb_jjlnu", "mli_dr_bb_jjlnu",
        "mli_dphi_bb_jjl", "mli_dr_bb_jjl", "mli_dphi_bb_nu", "mli_dphi_jj_nu", "mli_dr_bb_l", "mli_dr_jj_l",
        "mli_mbbjjlnu", "mli_mbbjjl", "mli_s_min",
    }
    self.produces |= self.ml_columns
