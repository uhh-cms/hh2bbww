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

from hbw.production.prepare_objects import prepare_objects, prepare_hhh_bjets
from hbw.config.dl.variables import add_dl_ml_variables, add_hhh_ml_variables
from hbw.production.ml_inputs import common_ml_inputs


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
        __import__("IPython").embed()
        raise ValueError(f"Extra fields in events: {diff}")


@producer(
    uses={common_ml_inputs, prepare_hhh_bjets},
    produces={common_ml_inputs, prepare_hhh_bjets},
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
    events = self[prepare_hhh_bjets](events, **kwargs)

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

    # create bjets combinatorics according to the minimum ∆R and assign reconstructed Higgs candidates
    def get_reco_higgs_properties(jet_collection, jet_collection_name, events=events):

        b1 = jet_collection[:, 0]
        b2 = jet_collection[:, 1]
        b3 = jet_collection[:, 2]
        b4 = jet_collection[:, 3]

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

        return hbb1, hbb2, min_index

    btag_jets = ak.pad_none(events.BtaggedJet, 4)
    hb_candidate = ak.pad_none(events.HBjetCandidate, 4)
    for jet_collection in [btag_jets, hb_candidate]:
        jet_collection_name = "btag" if jet_collection is btag_jets else "hb_candidate"
        # Higss reco properties
        hbb1, hbb2, min_index = get_reco_higgs_properties(jet_collection, jet_collection_name)

        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_mh1", hbb1.mass)
        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_mh2", hbb2.mass)
        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_h1_pt", hbb1.pt)
        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_h2_pt", hbb2.pt)
        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_dr_h1_h2", hbb1.delta_r(hbb2))
        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_dr_ll_h1", hll.delta_r(hbb1))
        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_dr_ll_h2", hll.delta_r(hbb2))
        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_dr_h1_nu", hbb1.delta_r(events[met_name]))
        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_dr_h2_nu", hbb2.delta_r(events[met_name]))

        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_mhhh", ((hll + events[met_name][:]) + hbb1 + hbb2).mass)  # noqa E501
        # events = set_ak_column_f32(events, f"mli_{jet_collection_name}_m4bllMET", (hll + ((b1 + b2 + b3 + b4) * 1) + events[met_name][:]).mass)  # noqa E501
        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_dr_h1_llMET", hbb1.delta_r(hll + events[met_name][:]))  # noqa E501
        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_dr_h2_llMET", hbb2.delta_r(hll + events[met_name][:]))  # noqa E501

        for i in range(0, 4):
            # low-level features
            for var in ["pt", "eta", "discrete_b_score"]:
                events = set_ak_column_f32(events, f"mli_{jet_collection_name}{i+1}_{var}", jet_collection[:, i][var])
        # high-level features
        hbjet_pairs = ak.combinations(jet_collection, 2)
        dr = hbjet_pairs[:, :, "0"].delta_r(hbjet_pairs[:, :, "1"])
        events = set_ak_column_f32(events, "maxdr_bb", ak.max(dr, axis=1))
        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_maxdr_jj", ak.max(dr, axis=1))
        events = set_ak_column_f32(events, f"mli_{jet_collection_name}_mindr_jj", ak.min(dr, axis=1))

    # NOTE Bjet variables that not necesssarily aare related to higgs kinematics, but just ordered, by our bjet logic
    events = set_ak_column_f32(events, "mli_dr_hbb", hb_candidate[:, 0].delta_r(hb_candidate[:, 1]))
    events = set_ak_column_f32(events, "mli_dphi_hbb", hb_candidate[:, 0].delta_phi(hb_candidate[:, 1]))
    events = set_ak_column_f32(events, "mli_discrete_b_score_sum", ak.sum(events.Jet.discrete_b_score, axis=1))

    # NOTE: Lepton and leading bjet properties
    min_dr_lb = ak.min(events.HBjetCandidate.delta_r(events.Lepton[:, 0]), axis=-1)
    events = set_ak_column_f32(events, "mli_min_dr_lb", min_dr_lb)

    hbjet_pairs = ak.combinations(events.HBjetCandidate, 2)
    min_dr_ll_bb = ak.min(hbjet_pairs[:, :, "0"].delta_r(hll), axis=1)
    events = set_ak_column_f32(events, "mli_min_dr_ll_bb", min_dr_ll_bb)

    lb = (events.Lepton[:, 0] + hb_candidate[:, 0] * 1)
    events = set_ak_column_f32(events, "mli_lb_pt", lb.pt)
    events = set_ak_column_f32(events, "mli_lb_mass", lb.mass)
    events = set_ak_column_f32(events, "mli_lb_pt_sum", events.Lepton[:, 0].pt + hb_candidate[:, 0].pt)

    # fill nan/none values of all produced columns
    for col in self.ml_input_columns:
        events = set_ak_column_f32(events, col, ak.fill_none(ak.nan_to_none(events[col]), ZERO_PADDING_VALUE))

    # fill nan/none values of all produced columns
    for col in self.ml_input_columns:
        events = set_ak_column_f32(events, col, ak.fill_none(ak.nan_to_none(events[col]), ZERO_PADDING_VALUE))
    check_column_bookkeeping(self, events)
    return events


@hhh_dl_ml_inputs.init
def hhh_dl_ml_inputs_init(self: Producer) -> None:
    # define ML input separately to self.produces
    self.ml_input_columns = {
        # high-level ll system
        "mli_mll", "mli_ll_pt", "mli_mllMET",
        "mli_dr_ll", "mli_dphi_ll", "mli_deta_ll",
        # low-level features lepton features
        "mli_lep2_pt", "mli_lep2_eta",
        "mli_lep_tag", "mli_lep2_tag", "mli_mixed_channel",
        # additional hhh-specific features
        "mli_dr_hbb", "mli_dphi_hbb", "mli_discrete_b_score_sum",
        "mli_min_dr_lb", "mli_min_dr_ll_bb",
        "mli_lb_pt_sum", "mli_lb_pt", "mli_lb_mass",
    } | set(
        f"mli_{obj}_{var}"
        for obj in ["btag", "hb_candidate"]
        for var in ["maxdr_jj", "mindr_jj"]
    ) | set(
        f"mli_{obj}_{var}"
        for obj in ["btag", "hb_candidate"]
        for var in ["mh1", "mh2", "h1_pt", "h2_pt", "dr_h1_h2", "dr_ll_h1", "dr_ll_h2", "mhhh", "dr_h1_llMET", "dr_h2_llMET", "dr_h1_nu", "dr_h2_nu"]  # "m4bllMET"  # noqa E501
    ) | set(
        f"mli_{obj}_{var}"
        for obj in ["btag1", "btag2", "btag3", "btag4"]
        for var in ["discrete_b_score", "pt", "eta"]
    ) | set(
        f"mli_{obj}_{var}"
        for obj in ["hb_candidate1", "hb_candidate2", "hb_candidate3", "hb_candidate4"]
        for var in ["discrete_b_score", "pt", "eta"]
    )
    self.produces |= self.ml_input_columns

    # bookkeep used ml_input_columns over multiple Producers
    self.config_inst.x.ml_input_columns = self.config_inst.x("ml_input_columns", set()) | self.ml_input_columns

    # add variable instances to config
    add_dl_ml_variables(self.config_inst)
    add_hhh_ml_variables(self.config_inst)
    check_variable_existence(self)
