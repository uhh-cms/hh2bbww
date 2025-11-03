# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from __future__ import annotations

import law
import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import remove_ak_column, set_ak_column

from hbw.production.prepare_objects import prepare_objects
from hbw.production.jets import vbf_candidates
from hbw.config.ml_variables import add_common_ml_variables, add_sl_ml_variables
from hbw.config.dl.variables import add_dl_ml_variables
from columnflow.production.cms.dy import recoil_corrected_met

from hbw.util import MET_COLUMN, IF_DY

ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

ZERO_PADDING_VALUE = -10


@producer(uses={prepare_objects, "*"}, produces={"dummy"})
def check_columns(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Check that all columns are present in the events.
    """
    # apply behavior (for variable reconstruction)
    events = self[prepare_objects](events, **kwargs)

    from hbw.util import debugger
    debugger()
    return events


def check_variable_existence(self: Producer) -> None:
    """
    Helper to check that all requested columns define a variable in the config
    """
    # check that all variables are defined in the config
    for column in self.ml_input_columns:
        if not self.config_inst.has_variable(column):
            raise ValueError(f"Variable {column} is not defined in the config.")


def check_column_bookkeeping(self: Producer, events: ak.Array) -> None:
    """
    Helper to check that all produced "mli" columns are bookkept in the config.
    """
    mli_fields = {field for field in events.fields if "mli_" in field}
    if diff := mli_fields - self.config_inst.x.ml_input_columns:
        raise ValueError(f"Extra fields in events: {diff}")


@producer(
    uses={
        # "*", "*.*",
        prepare_objects, vbf_candidates,
        "FatJet.{msoftdrop,particleNet_XbbVsQCD,particleNetWithMass_HbbvsQCD}",
        "{Electron,Muon,Jet,Bjet,Lightjet,ForwardJet,VBFJet,FatJet}.{pt,eta,phi,mass}",
        "{Electron,Muon}.{pdgId}",
        MET_COLUMN("pt"), MET_COLUMN("phi"), IF_DY("RecoilCorrMET.{pt,phi}"),
    },
    # produced columns set in the init function
)
def common_ml_inputs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    ML input features that can be shared between SL and DL channels.
    """
    # add behavior and define new collections (e.g. Lepton)
    events = self[prepare_objects](events, **kwargs)
    met_name = self.config_inst.x.met_name
    if self.dataset_inst.has_tag("is_dy"):
        met_name = "RecoilCorrMET"

    # vbf with and without forward region
    # NOTE: we need to clear the cache since we have to run the same Producer twice
    _clear_cache = kwargs.pop("_clear_cache", False)
    kwargs["_clear_cache"] = True
    for jet_collection, dst_basename in (
        ("Lightjet", "mli_vbf"),
        ("VBFCandidateJet", "mli_full_vbf"),
    ):
        print(jet_collection, dst_basename)
        events = self[vbf_candidates](events, jet_collection=jet_collection, **kwargs)
        vbfpair = ak.pad_none(events.VBFPair, 1)[:, 0]
        vbf_tags = ak.where(events.VBFPair.mass > 500, 1., 0.)
        events = set_ak_column_f32(events, f"{dst_basename}_tag", ak.num(vbf_tags))
        for col in ("pt", "eta", "phi", "mass", "deta"):
            events = set_ak_column_f32(events, f"{dst_basename}_{col}", vbfpair[col])

        # remove columns produced by vbf_candidates such that the same producer can be used again
        events = remove_ak_column(events, "VBFPair")
        events = remove_ak_column(events, "VBFJet")

    kwargs["_clear_cache"] = _clear_cache

    # object padding
    events = set_ak_column(events, "Lightjet", ak.pad_none(events.Lightjet, 2))
    events = set_ak_column(events, "Bjet", ak.pad_none(events.Bjet, 2))
    events = set_ak_column(events, "ForwardJet", ak.pad_none(events.ForwardJet, 2))
    events = set_ak_column(events, "InclJet", ak.pad_none(events.InclJet, 2))
    events = set_ak_column(events, "VBFCandidateJet", ak.pad_none(events.VBFCandidateJet, 2))
    events = set_ak_column(events, "FatBjet", ak.pad_none(events.FatBjet, 1))

    # setup correct btagging columns
    btag_wp_score = self.config_inst.x.btag_wp_score
    btag_column = self.config_inst.x.btag_column

    events = set_ak_column_f32(events, "Jet.b_score", events.Jet[btag_column])
    events = set_ak_column_f32(events, "Bjet.b_score", events.Bjet[btag_column])
    events = set_ak_column_f32(events, "Lightjet.b_score", events.Lightjet[btag_column])

    # H->bb FatJet
    for var in [
        "pt", "eta", "phi", "mass", "msoftdrop",
        "particleNet_XbbVsQCD", "particleNetWithMass_HbbvsQCD",
    ]:
        events = set_ak_column_f32(events, f"mli_fj_{var}", events.FatBjet[:, 0][var])

    # low-level features
    for var in ["pt", "eta", "b_score"]:
        events = set_ak_column_f32(events, f"mli_b1_{var}", events.Bjet[:, 0][var])
        events = set_ak_column_f32(events, f"mli_b2_{var}", events.Bjet[:, 1][var])
        # even in DL, ~10% of events contain 4 jets, so it might be worth keeping this
        events = set_ak_column_f32(events, f"mli_j1_{var}", events.Lightjet[:, 0][var])
        events = set_ak_column_f32(events, f"mli_j2_{var}", events.Lightjet[:, 1][var])
        # observables for full VBF jets
        if var == "b_score":
            continue
        events = set_ak_column_f32(events, f"mli_vbfcand1_{var}", events.VBFCandidateJet[:, 0][var])
        events = set_ak_column_f32(events, f"mli_vbfcand2_{var}", events.VBFCandidateJet[:, 1][var])

    events = set_ak_column_f32(events, "mli_lep_pt", events.Lepton[:, 0].pt)
    events = set_ak_column_f32(events, "mli_lep_eta", events.Lepton[:, 0].eta)
    events = set_ak_column_f32(events, "mli_met_pt", events[met_name].pt)
    events = set_ak_column_f32(events, "mli_met_phi", events[met_name].phi)

    # general
    events = set_ak_column_f32(events, "mli_ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column_f32(events, "mli_lt", ak.sum(events.Lepton.pt, axis=1) + events[met_name].pt)
    events = set_ak_column_f32(events, "mli_n_jet", ak.num(events.Jet.pt, axis=1))

    events = set_ak_column_f32(events, "mli_ht_alljets", ak.sum(events.InclJet.pt, axis=1))
    events = set_ak_column_f32(events, "mli_n_jet_alljets", ak.num(events.InclJet.pt, axis=1))

    events = set_ak_column_f32(events, "mli_ht_fwjets", ak.sum(events.ForwardJet.pt, axis=1))
    events = set_ak_column_f32(events, "mli_n_jet_fwjets", ak.num(events.ForwardJet.pt, axis=1))

    # bjets in general
    events = set_ak_column_f32(
        events, "mli_n_btag", ak.num(events.Jet[events.Jet.b_score > btag_wp_score], axis=1),
    )
    events = set_ak_column_f32(events, "mli_b_score_sum", ak.sum(events.Jet.b_score, axis=1))
    events = set_ak_column_f32(events, "mli_b_b_score_sum", ak.sum(events.Bjet.b_score, axis=1))
    events = set_ak_column_f32(events, "mli_l_b_score_sum", ak.sum(events.Lightjet.b_score, axis=1))

    # all possible jet pairs
    jet_pairs = ak.combinations(events.Jet, 2)
    dr = jet_pairs[:, :, "0"].delta_r(jet_pairs[:, :, "1"])
    events = set_ak_column_f32(events, "mli_mindr_jj", ak.min(dr, axis=1))
    events = set_ak_column_f32(events, "mli_maxdr_jj", ak.max(dr, axis=1))

    all_jet_pairs = ak.combinations(events.InclJet, 2)
    dr = all_jet_pairs[:, :, "0"].delta_r(all_jet_pairs[:, :, "1"])
    events = set_ak_column_f32(events, "mli_mindr_jj_alljets", ak.min(dr, axis=1))
    events = set_ak_column_f32(events, "mli_maxdr_jj_alljets", ak.max(dr, axis=1))

    # hbb features
    hbb = (events.Bjet[:, 0] + events.Bjet[:, 1]) * 1  # NOTE: *1 so it is a Lorentzvector not a candidate vector
    events = set_ak_column_f32(events, "mli_bb_pt", hbb.pt)
    events = set_ak_column_f32(events, "mli_mbb", hbb.mass)

    events = set_ak_column_f32(events, "mli_dr_bb", events.Bjet[:, 0].delta_r(events.Bjet[:, 1]))
    events = set_ak_column_f32(events, "mli_dphi_bb", abs(events.Bjet[:, 0].delta_phi(events.Bjet[:, 1])))
    events = set_ak_column_f32(events, "mli_deta_bb", abs(events.Bjet[:, 0].eta - (events.Bjet[:, 1]).eta))

    # angles to lepton
    mindr_lb = ak.min(events.Bjet.delta_r(events.Lepton[:, 0]), axis=-1)
    events = set_ak_column_f32(events, "mli_mindr_lb", mindr_lb)

    mindr_lj = ak.min(events.Lightjet.delta_r(events.Lepton[:, 0]), axis=-1)
    events = set_ak_column_f32(events, "mli_mindr_lj", mindr_lj)

    # fill nan/none values of all produced columns
    for col in self.ml_input_columns:
        events = set_ak_column_f32(events, col, ak.fill_none(ak.nan_to_none(events[col]), ZERO_PADDING_VALUE))

    check_column_bookkeeping(self, events)

    return events


@common_ml_inputs.init
def common_ml_inputs_init(self: Producer) -> None:
    btag_column = self.config_inst.x.btag_column
    self.uses |= {f"Jet.{btag_column}", f"Bjet.{btag_column}", f"Lightjet.{btag_column}"}

    # define ML input separately to self.produces
    self.ml_input_columns = {
        # event features
        "mli_ht", "mli_lt", "mli_n_jet",
        "mli_ht_alljets", "mli_n_jet_alljets",
        "mli_ht_fwjets", "mli_n_jet_fwjets",
        "mli_n_btag", "mli_b_score_sum", "mli_b_b_score_sum", "mli_l_b_score_sum",
        # bb system
        "mli_mbb", "mli_bb_pt", "mli_dr_bb", "mli_dphi_bb", "mli_deta_bb",
        # minimum angles
        "mli_mindr_lb", "mli_mindr_lj", "mli_mindr_jj", "mli_maxdr_jj",
        "mli_mindr_jj_alljets", "mli_maxdr_jj_alljets",
        # VBF features
        # "mli_vbf_deta", "mli_vbf_invmass", "mli_vbf_tag",
        # low-level features
        "mli_lep_pt", "mli_lep_eta",
        "mli_met_pt", "mli_met_phi",
    } | set(
        f"mli_{obj}_{var}"
        for obj in ["vbf", "full_vbf"]
        for var in ["pt", "eta", "phi", "mass", "deta", "tag"]
    ) | set(
        f"mli_{obj}_{var}"
        for obj in ["b1", "b2", "j1", "j2"]
        for var in ["b_score", "pt", "eta"]
    ) | set(
        f"mli_{obj}_{var}"
        for obj in ["vbfcand1", "vbfcand2"]
        for var in ["pt", "eta"]
    ) | set(
        f"mli_{obj}_{var}"
        for obj in ["fj"]
        for var in [
            "pt", "eta", "phi", "mass", "msoftdrop",
            "particleNet_XbbVsQCD", "particleNetWithMass_HbbvsQCD",
        ]
    )
    self.produces |= self.ml_input_columns

    # bookkeep used ml_input_columns over multiple Producers
    self.config_inst.x.ml_input_columns = self.config_inst.x("ml_input_columns", set()) | self.ml_input_columns

    # add variable instances to config
    add_common_ml_variables(self.config_inst)
    check_variable_existence(self)


@producer(
    uses={common_ml_inputs},
    produces={common_ml_inputs},
    # produced columns set in the init function
    version=1,
)
def sl_ml_inputs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer used for ML Training in the SL analysis.
    """
    met_name = self.config_inst.x.met_name
    if self.dataset_inst.has_tag("is_dy"):
        met_name = "RecoilCorrMET"

    # produce common input features
    events = self[common_ml_inputs](events, **kwargs)

    # wjj features
    events = set_ak_column_f32(events, "mli_dr_jj", events.Lightjet[:, 0].delta_r(events.Lightjet[:, 1]))
    events = set_ak_column_f32(events, "mli_dphi_jj", abs(events.Lightjet[:, 0].delta_phi(events.Lightjet[:, 1])))

    wjj = (events.Lightjet[:, 0] + events.Lightjet[:, 1]) * 1
    events = set_ak_column_f32(events, "mli_mjj", wjj.mass)

    # wlnu features
    # NOTE: we might want to consider neutrino reconstruction or transverse masses instead when including MET
    wlnu = (events[met_name] + events.Lepton[:, 0]) * 1
    events = set_ak_column_f32(events, "mli_mlnu", wlnu.mass)
    events = set_ak_column_f32(events, "mli_dphi_lnu", abs(events.Lepton[:, 0].delta_phi(events[met_name])))
    events = set_ak_column_f32(events, "mli_dphi_wl", abs(wlnu.delta_phi(events.Lepton[:, 0])))

    # hww features
    hww = wlnu + wjj
    hww_vis = events.Lepton[:, 0] + wjj

    events = set_ak_column_f32(events, "mli_mjjlnu", hww.mass)
    events = set_ak_column_f32(events, "mli_mjjl", hww_vis.mass)

    # hh system angles
    hbb = (events.Bjet[:, 0] + events.Bjet[:, 1]) * 1

    events = set_ak_column_f32(events, "mli_dphi_bb_jjlnu", abs(hbb.delta_phi(hww)))
    events = set_ak_column_f32(events, "mli_dr_bb_jjlnu", hbb.delta_r(hww))

    events = set_ak_column_f32(events, "mli_dphi_bb_jjl", abs(hbb.delta_phi(hww_vis)))
    events = set_ak_column_f32(events, "mli_dr_bb_jjl", hbb.delta_r(hww_vis))

    events = set_ak_column_f32(events, "mli_dphi_bb_nu", abs(hbb.delta_phi(events[met_name])))
    events = set_ak_column_f32(events, "mli_dphi_jj_nu", abs(wjj.delta_phi(events[met_name])))
    events = set_ak_column_f32(events, "mli_dr_bb_l", hbb.delta_r(events.Lepton[:, 0]))
    events = set_ak_column_f32(events, "mli_dr_jj_l", hbb.delta_r(events.Lepton[:, 0]))

    # hh features
    hh = hbb + hww
    hh_vis = hbb + hww_vis

    events = set_ak_column_f32(events, "mli_mbbjjlnu", hh.mass)
    events = set_ak_column_f32(events, "mli_mbbjjl", hh_vis.mass)

    s_min = (
        2 * events[met_name].pt * ((hh_vis.mass ** 2 + hh_vis.energy ** 2) ** 0.5 -
        hh_vis.pt * np.cos(hh_vis.delta_phi(events[met_name])) + hh_vis.mass ** 2)
    ) ** 0.5
    events = set_ak_column_f32(events, "mli_s_min", s_min)

    # TODO: variable to reconstruct top quark resonances (e.g. mT(lepton + met + b))

    # fill nan/none values of all produced columns
    for col in self.ml_input_columns:
        events = set_ak_column_f32(events, col, ak.fill_none(ak.nan_to_none(events[col]), ZERO_PADDING_VALUE))

    check_column_bookkeeping(self, events)
    return events


@sl_ml_inputs.init
def sl_ml_inputs_init(self: Producer) -> None:
    # define ML input separately to self.produces
    self.ml_input_columns = {
        # jj system
        "mli_dr_jj", "mli_dphi_jj", "mli_mjj",
        # lnu system
        "mli_dphi_lnu", "mli_mlnu", "mli_dphi_wl",
        # ww system
        "mli_mjjlnu", "mli_mjjl",
        # HH system
        "mli_dphi_bb_jjlnu", "mli_dr_bb_jjlnu",
        "mli_dphi_bb_jjl", "mli_dr_bb_jjl", "mli_dphi_bb_nu", "mli_dphi_jj_nu", "mli_dr_bb_l", "mli_dr_jj_l",
        "mli_mbbjjlnu", "mli_mbbjjl", "mli_mindr_jj",
        "mli_s_min",
    }
    self.produces |= self.ml_input_columns

    # bookkeep used ml_input_columns over multiple Producers
    self.config_inst.x.ml_input_columns = self.config_inst.x("ml_input_columns", set()) | self.ml_input_columns

    # add variable instances to config
    add_sl_ml_variables(self.config_inst)
    check_variable_existence(self)


@producer(
    uses={common_ml_inputs},
    produces={common_ml_inputs},
    # produced columns set in the init function
    version=law.config.get_expanded("analysis", "dl_ml_inputs_version", 2),
)
def dl_ml_inputs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer used for ML Training in the DL analysis.
    """
    met_name = self.config_inst.x.met_name
    if self.dataset_inst.has_tag("is_dy"):
        met_name = "RecoilCorrMET"

    # produce common input features
    events = self[common_ml_inputs](events, **kwargs)

    # object padding
    events = set_ak_column(events, "Lepton", ak.pad_none(events.Lepton, 2))

    for var in ["pt", "eta"]:
        events = set_ak_column_f32(events, f"mli_lep2_{var}".lower(), events.Lepton[:, 1][var])

    events = set_ak_column_f32(events, "mli_lep_tag", abs(events.Lepton[:, 0]["pdgId"]) == 13)
    events = set_ak_column_f32(events, "mli_lep2_tag", abs(events.Lepton[:, 1]["pdgId"]) == 13)
    events = set_ak_column_f32(events, "mli_mixed_channel", events.mli_lep_tag != events.mli_lep2_tag)

    # create ll object and ll variables
    hll = (events.Lepton[:, 0] + events.Lepton[:, 1])
    events = set_ak_column_f32(events, "mli_ll_pt", hll.pt)
    events = set_ak_column_f32(events, "mli_mll", hll.mass)
    events = set_ak_column_f32(events, "mli_mllMET", (hll + events[met_name][:]).mass)
    events = set_ak_column_f32(events, "mli_dr_ll", events.Lepton[:, 0].delta_r(events.Lepton[:, 1]))
    events = set_ak_column_f32(events, "mli_dphi_ll", abs(events.Lepton[:, 0].delta_phi(events.Lepton[:, 1])))
    events = set_ak_column_f32(events, "mli_deta_ll", abs(events.Lepton[:, 0].eta - (events.Lepton[:, 1]).eta))

    # minimum deltaR between lep and jet
    hbb = (events.Bjet[:, 0] + events.Bjet[:, 1]) * 1  # NOTE: *1 so it is a Lorentzvector not a candidate vector

    llbb_pairs = ak.cartesian([events.Lepton, events.Bjet], axis=1)
    lep, jet = ak.unzip(llbb_pairs)
    min_dr_llbb = (ak.min(lep.delta_r(jet), axis=-1))
    events = set_ak_column_f32(events, "mli_min_dr_llbb", min_dr_llbb)

    events = set_ak_column_f32(events, "mli_dr_ll_bb", hll.delta_r(hbb))

    # hh system
    events = set_ak_column_f32(events, "mli_mbbllMET", (hll + hbb + events[met_name][:]).mass)
    events = set_ak_column_f32(events, "mli_dr_bb_llMET", hbb.delta_r(hll + events[met_name][:]))
    events = set_ak_column_f32(events, "mli_dphi_bb_nu", abs(hbb.delta_phi(events[met_name])))
    events = set_ak_column_f32(events, "mli_dphi_bb_llMET", hbb.delta_phi(hll + events[met_name][:]))

    # fill nan/none values of all produced columns
    for col in self.ml_input_columns:
        events = set_ak_column_f32(events, col, ak.fill_none(ak.nan_to_none(events[col]), ZERO_PADDING_VALUE))
    check_column_bookkeeping(self, events)
    return events


@dl_ml_inputs.init
def dl_ml_inputs_init(self: Producer) -> None:
    # define ML input separately to self.produces
    self.ml_input_columns = {
        # ll system
        "mli_mll", "mli_dr_ll", "mli_dphi_ll", "mli_deta_ll", "mli_ll_pt",
        "mli_min_dr_llbb",
        # hh system
        "mli_dr_ll_bb",
        "mli_dphi_bb_nu", "mli_dphi_bb_llMET", "mli_mllMET",
        "mli_mbbllMET", "mli_dr_bb_llMET",
        # low-level features
        "mli_lep2_pt", "mli_lep2_eta",
        "mli_lep_tag", "mli_lep2_tag", "mli_mixed_channel",
    }
    self.produces |= self.ml_input_columns

    # bookkeep used ml_input_columns over multiple Producers
    self.config_inst.x.ml_input_columns = self.config_inst.x("ml_input_columns", set()) | self.ml_input_columns

    # add variable instances to config
    add_dl_ml_variables(self.config_inst)
    check_variable_existence(self)


@producer(
    uses={MET_COLUMN("{pt,phi}"), IF_DY(recoil_corrected_met)},
    produces={"met_pt_corr", "met_phi_corr"},
)
def METCorr(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    met_name = self.config_inst.x.met_name

    if self.dataset_inst.has_tag("is_dy"):
        events = self[recoil_corrected_met](events, **kwargs)
        met_pt_corr = events.RecoilCorrMET.pt
        met_phi_corr = events.RecoilCorrMET.phi
    else:
        met_pt_corr = events[met_name].pt
        met_phi_corr = events[met_name].phi

    events = set_ak_column_f32(events, "met_pt_corr", met_pt_corr)
    events = set_ak_column_f32(events, "met_phi_corr", met_phi_corr)

    for col in ["met_pt_corr", "met_phi_corr"]:
        events = set_ak_column_f32(events, col, ak.fill_none(ak.nan_to_none(events[col]), ZERO_PADDING_VALUE))

    return events


test_dl_ml_inputs = dl_ml_inputs.derive("test_dl_ml_inputs")
