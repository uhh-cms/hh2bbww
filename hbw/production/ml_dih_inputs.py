# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from __future__ import annotations

import law
import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from hbw.production.prepare_objects import prepare_objects
# from hbw.production.jets import vbf_candidates
from hbw.config.dl.variables import add_dl_ml_variables, add_hh_bjet_variables

from hbw.production.ml_inputs import common_ml_inputs, prepare_bjets  # , vbf_jets, METCorr

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
        prepare_objects,
        "Jet.*",
    },
    produces={
        "{btag_jet1,btag_jet2}.{pt,eta,phi,mass,btagUParTAK4B,discrete_b_score}",
        # "{hbjet1,hbjet2,hbjet3,hbjet4}.{pt,eta,phi,mass,btagUParTAK4B}",
        # "hhh_dr_bb",  # "mli_mindr_bb", "mli_maxdr_bb",
        "Jet.discrete_b_score", "discrete_sum_b_score",
        "check_n_btag",
    },
    version=0,
)
def hh_bjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to extract bjetas and btag properties for HH analysis
    hbjets are just possible jets from which to extract Higgs candidate and corresponding properties.
    btag_jets actually fulfill the btag wp threshold medium.
    """

    # add behavior and define new collections (e.g. Lepton)
    events = self[prepare_objects](events, **kwargs)

    # get discrete btag scores
    discrete_btag_scores = ak.zeros_like(events.Jet.btagUParTAK4B)
    for wp in [0.0246, 0.1272, 0.4648, 0.6298, 0.9739]:  # loose, medium, tight, etc.
        discrete_btag_scores = ak.where(
            events.Jet.btagUParTAK4B >= wp,
            ak.full_like(events.Jet.btagUParTAK4B, wp),
            discrete_btag_scores,
        )
    events = set_ak_column_f32(events, "Jet.discrete_b_score", discrete_btag_scores)
    events = set_ak_column_f32(events, "discrete_sum_b_score", ak.sum(discrete_btag_scores, axis=1))

    # get btagged jets acoording to the medium WP and sort them by pt
    btag_wp = self.config_inst.x.btag_wp_names["UParTAK4"][self.config_inst.x.btag_wp]
    # hhh_btag_mask = events.Jet.btagUParTAK4B > btag_wp
    btag_jet = events.Jet[events.Jet.btagUParTAK4B >= btag_wp]
    btag_jet = btag_jet[ak.argsort(btag_jet.pt, ascending=False)]
    # btag_jet = btag_jet[ak.argsort(btag_jet.discrete_b_score, ascending=False)]
    events = set_ak_column_f32(events, "check_n_btag", ak.sum(events.Jet.btagUParTAK4B >= btag_wp, axis=1))
    # events = set_ak_column(events, "btag_jet", btagged_jets)

    # get low level properties of 4 leading bjets (if less than 4 btagged jets, the rest is filled with None)
    events = set_ak_column(events, "btag_jets", ak.pad_none(btag_jet, 2))
    btag_jets = events.btag_jets
    for i in range(2):
        for col in ("pt", "eta", "phi", "mass", "discrete_b_score", "btagUParTAK4B"):

            events = set_ak_column_f32(
                events, f"btag_jet{i+1}.{col}",
                ak.fill_none(ak.nan_to_none(btag_jets[:, i][col]), ZERO_PADDING_VALUE),
            )

    # hbjets = events.hbjets[ak.argsort(events.hbjets.btagUParTAK4B, ascending=False)]
    # for i in range(4):
    #     for col in ("pt", "eta", "phi", "mass", "btagUParTAK4B"):
    #         events = set_ak_column_f32(events, f"hbjet{i+1}.{col}", hbjets[:, i][col])

    # hbjet_pairs = ak.combinations(btag_jets, 2)
    # dr = hbjet_pairs[:, :, "0"].delta_r(hbjet_pairs[:, :, "1"])
    # events = set_ak_column_f32(events, "mli_mindr_bb", ak.min(dr, axis=1))
    # events = set_ak_column_f32(events, "mli_maxdr_bb", ak.max(dr, axis=1))
    events = set_ak_column_f32(events, "hhh_dr_bb", btag_jets[:, 0].delta_r(btag_jets[:, 1]))

    # for i in range(4):
    #     for col in ["pt", "eta", "phi", "mass", "btagUParTAK4B", "discrete_b_score"]:
    #         events = set_ak_column_f32(events, events[f"btag_jet{i+1}"][col], ak.fill_none(ak.nan_to_none(events[f"btag_jet{i+1}"][col]), ZERO_PADDING_VALUE))  # noqa E501

    # for col in ["hhh_dr_bb", "mli_mindr_bb", "mli_maxdr_bb", "discrete_sum_b_score"]:
    for col in ["hhh_dr_bb", "discrete_sum_b_score"]:
        events = set_ak_column_f32(events, col, ak.fill_none(ak.nan_to_none(events[col]), ZERO_PADDING_VALUE))

    return events


@producer(
    uses={common_ml_inputs, prepare_bjets},
    produces={common_ml_inputs, prepare_bjets},
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
    events = self[prepare_bjets](events, n_btags=2, **kwargs)

    # object padding
    events = set_ak_column(events, "Lepton", ak.pad_none(events.Lepton, 2))
    events = set_ak_column(events, "BJet", ak.pad_none(events.BJet, 2))
    events = set_ak_column(events, "LightJet", ak.pad_none(events.LightJet, 2))

    # add btagging based information
    # the collections with L/B Jet capital "J" are based on the working point btagging
    for var in ["pt", "eta", "b_score", "discrete_b_score"]:
        events = set_ak_column_f32(events, f"mli_b1_{var}", events.BJet[:, 0][var])
        events = set_ak_column_f32(events, f"mli_b2_{var}", events.BJet[:, 1][var])
        # even in DL, ~10% of events contain 4 jets, so it might be worth keeping this
        events = set_ak_column_f32(events, f"mli_j1_{var}", events.LightJet[:, 0][var])
        events = set_ak_column_f32(events, f"mli_j2_{var}", events.LightJet[:, 1][var])

    events = set_ak_column_f32(events, "mli_b_score_sum", ak.sum(events.Jet.b_score, axis=1))
    events = set_ak_column_f32(events, "mli_b_b_score_sum", ak.sum(events.BJet.b_score, axis=1))
    events = set_ak_column_f32(events, "mli_l_b_score_sum", ak.sum(events.LightJet.b_score, axis=1))

    events = set_ak_column_f32(events, "mli_discrete_b_score_sum", ak.sum(events.Jet.discrete_b_score, axis=1))
    events = set_ak_column_f32(events, "mli_b_discrete_b_score_sum", ak.sum(events.BJet.discrete_b_score, axis=1))
    events = set_ak_column_f32(events, "mli_l_discrete_b_score_sum", ak.sum(events.LightJet.discrete_b_score, axis=1))

    # hbb features
    hbb = (events.BJet[:, 0] + events.BJet[:, 1]) * 1  # NOTE: *1 so it is a Lorentzvector not a candidate vector
    events = set_ak_column_f32(events, "mli_bb_pt", hbb.pt)
    events = set_ak_column_f32(events, "mli_mbb", hbb.mass)

    events = set_ak_column_f32(events, "mli_dr_bb", events.BJet[:, 0].delta_r(events.BJet[:, 1]))
    events = set_ak_column_f32(events, "mli_dphi_bb", abs(events.BJet[:, 0].delta_phi(events.BJet[:, 1])))
    events = set_ak_column_f32(events, "mli_deta_bb", abs(events.BJet[:, 0].eta - (events.BJet[:, 1]).eta))

    # angles to lepton
    mindr_lb = ak.min(events.BJet.delta_r(events.Lepton[:, 0]), axis=-1)
    events = set_ak_column_f32(events, "mli_mindr_lb", mindr_lb)

    mindr_lj = ak.min(events.LightJet.delta_r(events.Lepton[:, 0]), axis=-1)
    events = set_ak_column_f32(events, "mli_mindr_lj", mindr_lj)

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

    llbb_pairs = ak.cartesian([events.Lepton, events.BJet], axis=1)
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
        # btagging event features
        "mli_b_score_sum", "mli_b_b_score_sum", "mli_l_b_score_sum",
        "mli_discrete_b_score_sum", "mli_b_discrete_b_score_sum", "mli_l_discrete_b_score_sum",
        # hbb system
        "mli_mbb", "mli_bb_pt", "mli_dr_bb", "mli_dphi_bb", "mli_deta_bb",
        # lb lj system
        "mli_mindr_lb", "mli_mindr_lj",
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
    } | set(
        f"mli_{obj}_{var}"
        for obj in ["b1", "b2", "j1", "j2"]
        for var in ["discrete_b_score", "b_score", "pt", "eta"]
    )
    self.produces |= self.ml_input_columns

    # bookkeep used ml_input_columns over multiple Producers
    self.config_inst.x.ml_input_columns = self.config_inst.x("ml_input_columns", set()) | self.ml_input_columns

    # add variable instances to config
    add_dl_ml_variables(self.config_inst)
    add_hh_bjet_variables(self.config_inst)
    check_variable_existence(self)
