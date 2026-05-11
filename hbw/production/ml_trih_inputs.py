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
from columnflow.production.cms.btag import btag_wp_weights

from hbw.production.prepare_objects import prepare_objects
# from hbw.production.jets import vbf_candidates
from hbw.config.dl.variables import add_dl_ml_variables, add_hhh_dl_ml_variables, add_hhh_bjet_variables
from hbw.production.ml_inputs import common_ml_inputs  # , METCorr, vbf_jets


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
        prepare_objects,
        "Jet.*",
    },
    produces={
        "{btag_jet1,btag_jet2,btag_jet3,btag_jet4}.{pt,eta,phi,mass,btagUParTAK4B,discrete_b_score}",
        # "{hbjet1,hbjet2,hbjet3,hbjet4}.{pt,eta,phi,mass,btagUParTAK4B}",
        "hhh_dr_bb",  # "mli_mindr_bb", "mli_maxdr_bb",
        "Jet.discrete_b_score", "discrete_sum_b_score",
        "check_n_btag",
    },  # noqa E501
    version=2,
)
def hhh_bjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to extract bjetas and btag properties for HHH analysis
    hbjets are just possible jets from which to extract Hiuggs candidate and corresponding properties.
    btag_jets actually fulfill the btag wp threshold medium.
    """

    # add behavior and define new collections (e.g. Lepton)
    events = self[prepare_objects](events, **kwargs)
    # events = set_ak_column(events, "hbjets", ak.pad_none(events.Jet, 4))
    # __import__("IPython").embed()
    # get discrete batg scores
    discrete_btag_scores = ak.zeros_like(events.Jet.btagUParTAK4B)
    for wp in [0.0246, 0.1272, 0.4648, 0.6298, 0.9739]:  # loose, medium, tight, etc.
        discrete_btag_scores = ak.where(events.Jet.btagUParTAK4B >= wp, ak.full_like(events.Jet.btagUParTAK4B, wp), discrete_btag_scores)  # noqa E501
    events = set_ak_column_f32(events, "Jet.discrete_b_score", discrete_btag_scores)
    events = set_ak_column_f32(events, "discrete_sum_b_score", ak.sum(discrete_btag_scores, axis=1))

    # get btagged jets acoording to the medium WP and sort them by pt
    btag_wp = self.config_inst.x.btag_wp_names["UParTAK4"][self.config_inst.x.btag_wp]
    # hhh_btag_mask = events.Jet.btagUParTAK4B > btag_wp
    btag_jet = events.Jet[events.Jet.btagUParTAK4B >= btag_wp]
    btag_jet = btag_jet[ak.argsort(btag_jet.pt, ascending=False)]
    btag_jet = btag_jet[ak.argsort(btag_jet.discrete_b_score, ascending=False)]
    events = set_ak_column_f32(events, "check_n_btag", ak.sum(events.Jet.btagUParTAK4B >= btag_wp, axis=1))
    # events = set_ak_column(events, "btag_jet", btagged_jets)

    # get low level properties of 4 leading bjets (if less than 4 btagged jets, the rest is filled with None)
    events = set_ak_column(events, "btag_jets", ak.pad_none(btag_jet, 4))
    btag_jets = events.btag_jets
    for i in range(4):
        for col in ("pt", "eta", "phi", "mass", "discrete_b_score", "btagUParTAK4B"):

            events = set_ak_column_f32(events, f"btag_jet{i+1}.{col}", ak.fill_none(ak.nan_to_none(btag_jets[:, i][col]), ZERO_PADDING_VALUE)) # noqa E501

    # hbjets = events.hbjets[ak.argsort(events.hbjets.btagUParTAK4B, ascending=False)]
    # for i in range(4):
    #     for col in ("pt", "eta", "phi", "mass", "btagUParTAK4B"):
    #         events = set_ak_column_f32(events, f"hbjet{i+1}.{col}", hbjets[:, i][col])

    hbjet_pairs = ak.combinations(btag_jets, 2)
    dr = hbjet_pairs[:, :, "0"].delta_r(hbjet_pairs[:, :, "1"])
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


@hhh_bjets.init
def hhh_bjets_init(self: Producer) -> None:
    # @call_once_on_config
    # def add_hhh_b_variables(config: law.config.Config) -> None:
    #     from hbw.config.styling import default_var_unit, default_var_title_format, default_var_binning
    #     for i in range(4):
    #         for var in ["pt", "eta", "phi", "mass", "b_score"]:
    #             config.add_variable(
    #                 name=f"hbjet{i+1}_{var}",
    #                 expression=f"hbjet{i+1}.{var}",
    #                 unit=default_var_unit.get(var, "1"),
    #                 binning=default_var_binning[var] if var != "b_score" else (50, 0, 1),
    #                 x_title=f"Bjet (hhh) {i+1} {default_var_title_format.get(var, var)}",
    #             )
    add_hhh_bjet_variables(self.config_inst)


@producer(
    uses={common_ml_inputs, hhh_bjets},
    produces={common_ml_inputs, hhh_bjets},
    # produced columns set in the init function
    version=law.config.get_expanded("analysis", "hhh_dl_ml_inputs_version", 0),
)
def hhh_dl_ml_inputs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer used for ML Training in the DL analysis.
    """
    met_name = self.config_inst.x.met_name
    if self.dataset_inst.has_tag("is_dy"):
        met_name = "RecoilCorrMET"

    # produce common input features
    events = self[common_ml_inputs](events, **kwargs)
    events = self[hhh_bjets](events, **kwargs)

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

    # create bjets combinatorics according to the minimum ∆R
    # events = set_ak_column(events, "hbbjets", ak.pad_none(events.Jet, 4))
    hbbjets = events.btag_jets
    # hbbjets = hbbjets[ak.argsort(hbbjets.b_score, ascending=False)]

    b1 = hbbjets[:, 0]
    b2 = hbbjets[:, 1]
    b3 = hbbjets[:, 2]
    b4 = hbbjets[:, 3]

    # low-level features
    # for var in ["pt", "eta", "b_score"]:
    #     events = set_ak_column_f32(events, f"mli_bj1_{var}", b1[var])
    #     events = set_ak_column_f32(events, f"mli_bj2_{var}", b2[var])
    #     events = set_ak_column_f32(events, f"mli_bj3_{var}", b3[var])
    #     events = set_ak_column_f32(events, f"mli_bj4_{var}", b4[var])

    # I leave this here, because I might need to put fields in first (like b_score sum, e.g.
    # hbbA = (b1 + b2) * 1
    # hbbB = (b1 + b3) * 1
    # hbbC = (b1 + b4) * 1

    A = b1.delta_r(b2) + b3.delta_r(b4)
    B = b1.delta_r(b3) + b2.delta_r(b4)
    C = b1.delta_r(b4) + b2.delta_r(b3)

    combinations = [A, B, C]
    delta_r_sums = ak.Array([list(i) for i in zip(*combinations)])
    min_index = ak.argmin(delta_r_sums, axis=1)

    bjets_template = ak.full_like(((b1 + b2) * 1), 99999)  # same dimension as events.
    # Combination A
    hbb1 = ak.where(min_index == 0, (b1 + b2) * 1, bjets_template)
    hbb2 = ak.where(min_index == 0, ((b3 + b4) * 1), bjets_template)

    # Combination B
    hbb1 = ak.where(min_index == 1, (b1 + b3) * 1, hbb1)
    hbb2 = ak.where(min_index == 1, (b2 + b4) * 1, hbb2)

    # Combimaation C
    hbb1 = ak.where(min_index == 2, (b1 + b4) * 1, hbb1)
    hbb2 = ak.where(min_index == 2, (b2 + b3) * 1, hbb2)

    events = set_ak_column_f32(events, "mli_mbb1", hbb1.mass)
    events = set_ak_column_f32(events, "mli_mbb2", hbb2.mass)
    events = set_ak_column_f32(events, "mli_dr_bb_bb", hbb1.delta_r(hbb2))
    events = set_ak_column_f32(events, "mli_dr_ll_bb1", hll.delta_r(hbb1))
    events = set_ak_column_f32(events, "mli_dr_ll_bb2", hll.delta_r(hbb2))

    events = set_ak_column_f32(events, "mli_mhhh", ((hll + events[met_name][:]) + hbb1 + hbb2).mass)
    events = set_ak_column_f32(events, "mli_m4bllMET", (hll + ((b1 + b2 + b3 + b4) * 1) + events[met_name][:]).mass)
    events = set_ak_column_f32(events, "mli_dr_bb1_llMET", hbb1.delta_r(hll + events[met_name][:]))
    events = set_ak_column_f32(events, "mli_dr_bb2_llMET", hbb2.delta_r(hll + events[met_name][:]))

    # fill nan/none values of all produced columns
    for col in self.ml_input_columns:
        events = set_ak_column_f32(events, col, ak.fill_none(ak.nan_to_none(events[col]), ZERO_PADDING_VALUE))
    check_column_bookkeeping(self, events)
    return events


@hhh_dl_ml_inputs.init
def hhh_dl_ml_inputs_init(self: Producer) -> None:
    # define ML input separately to self.produces
    self.ml_input_columns = {
        # ll system
        "mli_mll", "mli_dr_ll", "mli_dphi_ll", "mli_deta_ll", "mli_ll_pt",
        "mli_mbb1", "mli_mbb2", "mli_dr_bb_bb",
        "mli_dr_ll_bb1", "mli_dr_ll_bb2",
        "mli_mhhh", "mli_m4bllMET",
        "mli_dr_bb1_llMET", "mli_dr_bb2_llMET",

        # "mli_min_dr_llbb",
        # hh system
        # "mli_dr_ll_bb",
        # "mli_dphi_bb_nu", "mli_dphi_bb_llMET",
        "mli_mllMET",
        # "mli_mbbllMET", "mli_dr_bb_llMET",
        # low-level features
        "mli_lep2_pt", "mli_lep2_eta",
        "mli_lep_tag", "mli_lep2_tag", "mli_mixed_channel",
    }
    self.produces |= self.ml_input_columns

    # bookkeep used ml_input_columns over multiple Producers
    self.config_inst.x.ml_input_columns = self.config_inst.x("ml_input_columns", set()) | self.ml_input_columns

    # add variable instances to config
    add_dl_ml_variables(self.config_inst)
    add_hhh_dl_ml_variables(self.config_inst)
    check_variable_existence(self)


@producer(
    uses={
        # "*", "*.*",
        prepare_objects,
        btag_wp_weights,
        "Jet.*",
    },
    produces={"flavour_test", "PT", "btag_weight"},
)
def testing_btag_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Simple Producer to extract pt and eta of the two VBF jets.
    """

    # add behavior and define new collections (e.g. Lepton)
    events = self[prepare_objects](events, **kwargs)
    jet_mask = (events.Jet["pt"] < 10_000) & (abs(events.Jet["eta"]) < 2.5)
    events = self[btag_wp_weights](events, jet_mask=jet_mask, **kwargs)
    events = set_ak_column_f32(events, "flavour_test", events.Jet["hadronFlavour"])
    events = set_ak_column_f32(events, "PT", events.Jet["pt"])
    events = set_ak_column_f32(events, "btag_weight", events.btag_weight)

    return events


# @testing_btag_weights.init
# def testing_btag_weights_init(self: Producer) -> None:

#     # @call_once_on_config
#     # def add_btesting_variables(config: law.config.Config) -> None:
#     #     from hbw.config.styling import default_var_unit, default_var_title_format, default_var_binning
#     #     for i in range(4):
#     #         for var in ["pt", "eta", "phi", "mass", "b_score"]:
#     #             config.add_variable(
#     #                 name=f"hbjet{i+1}_{var}",
#     #                 expression=f"hbjet{i+1}.{var}",
#     #                 unit=default_var_unit.get(var, "1"),
#     #                 binning=default_var_binning[var] if var != "b_score" else (50, 0, 1),
#     #                 x_title=f"Bjet (hhh) {i+1} {default_var_title_format.get(var, var)}",
#     #             )
