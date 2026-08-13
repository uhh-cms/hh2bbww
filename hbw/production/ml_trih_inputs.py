# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from __future__ import annotations

import law
import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import set_ak_column

from hbw.production.prepare_objects import prepare_objects, prepare_hhh_bjets
from hbw.config.dl.variables import add_dl_ml_variables, add_hhh_ml_variables, add_gatja_scores_variables
from hbw.production.ml_inputs import common_ml_inputs
from columnflow.production.cms.btag import btag_wp_weights
from hbw.production.weights import event_weights
from hbw.util import IF_GATJA

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


@producer(
    uses={
        prepare_objects,
        btag_wp_weights,
        common_ml_inputs,
        prepare_hhh_bjets,
        event_weights,
        "Jet.*",
    },
    produces={"event_id",
        "jetPT1", "jetPT2", "jetPT3", "jetPT4", "jetPT5", "jetPT6", "jetPT7", "jetPT8",
        "jetEta1", "jetEta2", "jetEta3", "jetEta4", "jetEta5", "jetEta6", "jetEta7", "jetEta8",
        "leptonPT1", "leptonEta1", "leptonPT2", "leptonEta2", "leptonPhi1", "leptonPhi2",
        "bjetAverageMass", "jetAverageMass",
        "bjetAverageMassSqr", "jetHT", "bjetHT", "lightjetHT", "jetNumber", "bjetNumber",
        "jetPhi1", "jetPhi2", "jetPhi3", "jetPhi4", "jetPhi5", "jetPhi6", "jetPhi7", "jetPhi8",
        "averageDeltaEtabb", "minDeltaRjj",
        "minDeltaRbb",
        "maxDeltaEtabb", "maxDeltaEtajj", "maxDeltaEtabj",
        "minDeltaRbj", "averageDeltaEtabj", "averageDeltaRbj", "minDeltaRMassjj", "minDeltaRMassbb", "minDeltaRMassbj",  # noqa E501
        "minDeltaRpTjj", "minDeltaRpTbb", "minDeltaRpTbj", "maxPTmassjjj", "maxPTmassjbb", "met", "metPhi",
        "minDeltaRbb_GATJA",
        "jetMinChiHiggsIndex1", "jetSecMinChiHiggsIndex1", "jetMinChiHiggsIndex2", "jetSecMinChiHiggsIndex2", "jetMinChiHiggsIndex3", "jetSecMinChiHiggsIndex3", "jetMinChiHiggsIndex4", "jetSecMinChiHiggsIndex4",  # noqa E501
        "jetMinChiHiggsIndex5", "jetSecMinChiHiggsIndex5", "jetMinChiHiggsIndex6", "jetSecMinChiHiggsIndex6", "jetMinChiHiggsIndex7", "jetSecMinChiHiggsIndex7", "jetMinChiHiggsIndex8", "jetSecMinChiHiggsIndex8",  # noqa E501
        "jetBTagDisc1", "jetBTagDisc2", "jetBTagDisc3", "jetBTagDisc4", "jetBTagDisc5", "jetBTagDisc6", "jetBTagDisc7", "jetBTagDisc8", "btag_weight", "weights",  # noqa E501
    },  # noqa
)
def gatja_inputs_jet_based_simplified(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    mass_higgs = 125.0
    sigma = 10.0  # PLATZHALTER FÜR DIE MASSENAUFLÖSUNG -> MUSS NOCH BESTIMMT WERDEN

    # produce common input features
    events = self[common_ml_inputs](events, **kwargs)
    events = self[prepare_hhh_bjets](events, **kwargs)
    if self.dataset_inst.is_mc:
        events = self[event_weights](events, **kwargs)
    if self.dataset_inst.is_data:
        events = set_ak_column_f32(events, "weights", ak.ones_like(events.mli_n_jet))
        events = set_ak_column_f32(events, "btag_weight", ak.ones_like(events.mli_n_jet))

    # add behavior and define new collections (e.g. Lepton)
    events = self[prepare_objects](events, **kwargs)
    jet_mask = (events.Jet["pt"] < 10_000) & (abs(events.Jet["eta"]) < 2.5)
    if self.dataset_inst.is_mc:
        events = self[btag_wp_weights](events, jet_mask=jet_mask, **kwargs)
    if self.dataset_inst.is_data:
        events = set_ak_column_f32(events, "btag_weight", ak.ones_like(events.mli_n_jet))
    jets = events.Jet
    padded_jets = ak.pad_none(jets, 8)
    padded_lepton = ak.pad_none(events.Lepton, 2)

    btag_column = self.config_inst.x.btag_column
    btag_wp_score = self.config_inst.x.btag_wp_score
    is_bjet = events.Jet[btag_column] >= btag_wp_score
    bjets = events.Jet[is_bjet]
    n_bjets = ak.num(bjets, axis=1)

    j1, j2 = ak.unzip(ak.combinations(events.Jet, 2))
    d_eta_jet_pairs = (j1.eta - j2.eta)
    deltaR_jj = j1.delta_r(j2)

    b1, b2 = ak.unzip(ak.combinations(bjets, 2))
    d_eta_bjet_pairs = (b1.eta - b2.eta)
    d_phi_bjet_pairs_GATJA = (b1.phi - b2.phi)
    deltaR_bb = b1.delta_r(b2)
    DeltaRbb_GATJA = ((d_eta_bjet_pairs)**2 + (d_phi_bjet_pairs_GATJA)**2)**0.5

    b_mix, j_mix = ak.unzip(ak.cartesian([bjets, events.Jet]))
    deltaR_bj = b_mix.delta_r(j_mix)

    j1_3, j2_3, j3_3 = ak.unzip(ak.combinations(events.Jet, 3))
    j_cart, bb_cart = ak.unzip(ak.cartesian([events.Jet, ak.combinations(bjets, 2)]))
    b1_cart, b2_cart = ak.unzip(bb_cart)
    events = set_ak_column_f32(events, "event_id", events.event)

    for i in range(8):
        events = set_ak_column_f32(events, f"jetPT{i+1}", ak.fill_none(padded_jets["pt"][:, i], -6))
    for i in range(8):
        events = set_ak_column_f32(events, f"jetEta{i+1}", ak.fill_none(padded_jets["eta"][:, i], -6))
    for i in range(2):
        events = set_ak_column_f32(events, f"leptonPT{i + 1}", ak.fill_none(padded_lepton["pt"][:, i], -6))
        events = set_ak_column_f32(events, f"leptonEta{i + 1}", ak.fill_none(padded_lepton["eta"][:, i], -6))
    for i in range(2):
        events = set_ak_column_f32(events, f"leptonPhi{i + 1}", ak.fill_none(padded_lepton["phi"][:, i], -6))
    bjetaveragemass = ak.mean(bjets.mass, axis=1)
    events = set_ak_column_f32(events, "bjetAverageMass", ak.fill_none(ak.nan_to_none(bjetaveragemass), -6))
    jetaveragemass = ak.mean(jets.mass, axis=1)
    events = set_ak_column_f32(events, "jetAverageMass", ak.where(ak.num(events.Jet) > 0, ak.sum(events.Jet.mass, axis=1) / ak.num(events.Jet), 0))  # noqa E501
    events = set_ak_column_f32(events, "bjetAverageMassSqr", ak.fill_none(ak.nan_to_none(bjetaveragemass * bjetaveragemass * ak.num(bjets)), -6))  # noqa E501
    events = set_ak_column_f32(events, "jetAverageMassSqr", ak.fill_none(ak.nan_to_none(jetaveragemass * jetaveragemass * ak.num(jets)), -6))  # noqa E501
    events = set_ak_column_f32(events, "jetHT", ak.sum(jets.pt, axis=1))
    events = set_ak_column_f32(events, "bjetHT", ak.sum(bjets.pt, axis=1))
    events = set_ak_column_f32(events, "lightjetHT", ak.sum(events.Lightjet.pt, axis=1))
    events = ak.with_field(events, ak.num(events.Jet), "jetNumber")
    n_bjets = ak.num(bjets, axis=1)
    events = ak.with_field(events, n_bjets, "bjetNumber")
    for i in range(8):
        events = set_ak_column_f32(events, f"jetPhi{i+1}", ak.fill_none(padded_jets["phi"][:, i], -6))
    events = set_ak_column_f32(events, "averageDeltaEtabb", ak.fill_none(ak.nan_to_none(ak.mean(abs(d_eta_bjet_pairs), axis=1)), -6))  # noqa E501
    events = set_ak_column_f32(events, "minDeltaRjj", ak.fill_none(ak.nan_to_none(ak.min(deltaR_jj, axis=1)), -6))
    events = set_ak_column_f32(events, "minDeltaRbb", ak.fill_none(ak.nan_to_none(ak.min(deltaR_bb, axis=1)), -6))
    events = set_ak_column_f32(events, "minDeltaRbb_GATJA", ak.fill_none(ak.nan_to_none(ak.min(DeltaRbb_GATJA, axis=1)), -6))  # noqa E501

    events = set_ak_column_f32(events, "maxDeltaEtabb", ak.fill_none(ak.nan_to_none(ak.max(abs(d_eta_bjet_pairs), axis=1)), -6))  # noqa E501
    events = set_ak_column_f32(events, "maxDeltaEtajj", ak.fill_none(ak.nan_to_none(ak.max(abs(d_eta_jet_pairs), axis=1)), -6))  # noqa E501
    events = set_ak_column_f32(events, "maxDeltaEtabj", ak.fill_none(ak.nan_to_none(ak.max(abs(b_mix.eta - j_mix.eta), axis=1)), -6))  # noqa E501
    events = set_ak_column_f32(events, "minDeltaRbj", ak.fill_none(ak.nan_to_none(ak.min(deltaR_bj, axis=1)), -6))

    events = set_ak_column_f32(events, "averageDeltaEtabj", ak.fill_none(ak.nan_to_none(ak.mean(abs(b_mix.eta - j_mix.eta), axis=1)), -6))  # noqa E501
    events = set_ak_column_f32(events, "averageDeltaRbj", ak.fill_none(ak.nan_to_none(ak.mean(deltaR_bj, axis=1)), -6))  # noqa E501

    mask_min_dR_jj = deltaR_jj == ak.min(deltaR_jj, axis=1, keepdims=True)
    events = set_ak_column_f32(events, "minDeltaRMassjj", ak.fill_none(ak.firsts((j1 + j2).mass[mask_min_dR_jj]), -6))
    mask_min_dR_bb = deltaR_bb == ak.min(deltaR_bb, axis=1, keepdims=True)
    events = set_ak_column_f32(events, "minDeltaRMassbb", ak.fill_none(ak.firsts((b1 + b2).mass[mask_min_dR_bb]), -6))
    mask_min_dR_bj = deltaR_bj == ak.min(deltaR_bj, axis=1, keepdims=True)
    events = set_ak_column_f32(events, "minDeltaRMassbj", ak.fill_none(ak.firsts((b_mix + j_mix).mass[mask_min_dR_bj]), -6))  # noqa E501
    events = set_ak_column_f32(events, "minDeltaRpTjj", ak.fill_none(ak.firsts((j1.pt + j2.pt)[mask_min_dR_jj]), -6))
    events = set_ak_column_f32(events, "minDeltaRpTbb", ak.fill_none(ak.firsts((b1.pt + b2.pt)[mask_min_dR_bb]), -6))
    events = set_ak_column_f32(events, "minDeltaRpTbj", ak.fill_none(ak.firsts((b_mix.pt + j_mix.pt)[mask_min_dR_bj]), -6))  # noqa E501

    pt_jjj = j1_3.pt + j2_3.pt + j3_3.pt
    mask_max_pT_jjj = (pt_jjj) == ak.max(pt_jjj, axis=1, keepdims=True)
    events = set_ak_column_f32(events, "maxPTmassjjj", ak.fill_none(ak.firsts((j1_3 + j2_3 + j3_3).mass[mask_max_pT_jjj]), -6))  # noqa E501
    pt_jbb = j_cart.pt + b1_cart.pt + b2_cart.pt
    mask_max_pT_jbb = (pt_jbb) == ak.max(pt_jbb, axis=1, keepdims=True)
    events = set_ak_column_f32(events, "maxPTmassjbb", ak.fill_none(ak.firsts((j_cart + b1_cart + b2_cart).mass[mask_max_pT_jbb]), -6))  # noqa E501
    events = set_ak_column_f32(events, "met", events.mli_met_pt)
    events = set_ak_column_f32(events, "metPhi", events.mli_met_phi)

    events = set_ak_column_f32(events, "btag_weight", events.btag_weight)
    if self.dataset_inst.is_mc:
        events = set_ak_column_f32(events, "weights", events.stitched_normalization_weight)

    n_jets = ak.num(jets, axis=1)
    jets_i = padded_jets[:, :, np.newaxis]
    jets_j = padded_jets[:, np.newaxis, :]
    dijet = (jets_i + jets_j)
    dijet_mass = ak.without_parameters(dijet.mass)

    idx = ak.local_index(padded_jets)
    idx_i = idx[:, :, np.newaxis]
    idx_j = idx[:, np.newaxis, :]

    valid_i = ~ak.is_none(jets_i.pt)
    valid_j = ~ak.is_none(jets_j.pt)

    pair_valid = valid_i & valid_j & (idx_i != idx_j)

    chi2_matrix = ((dijet_mass - mass_higgs) / sigma)**2
    chi2_matrix = ak.where(pair_valid, chi2_matrix, np.inf)

    min_idx = ak.argmin(chi2_matrix, axis=2)
    min_idx_filled = ak.fill_none(min_idx, -6)
    mask_sec = idx_j != min_idx_filled[:, :, np.newaxis]
    chi2_matrix_sec = ak.where(mask_sec, chi2_matrix, np.inf)
    sec_min_idx = ak.argmin(chi2_matrix_sec, axis=2)
    sec_min_idx_filled = ak.fill_none(sec_min_idx, -6)

    n_jets_1 = (n_jets == 1)[:, np.newaxis]
    min_idx_final = ak.where(n_jets_1, -999, min_idx_filled)
    sec_min_idx_final = ak.where(n_jets_1, -999, sec_min_idx_filled)

    for i in range(8):
        jet_exists = n_bjets > i

        mi = min_idx_final[:, i]
        si = sec_min_idx_final[:, i]

        mi_out = ak.where(jet_exists, mi, -6)
        si_out = ak.where(jet_exists, si, -6)

        events = set_ak_column_f32(events, f"jetMinChiHiggsIndex{i+1}", mi_out)
        events = set_ak_column_f32(events, f"jetSecMinChiHiggsIndex{i+1}", si_out)

    for i in range(8):
        mass_filled = ak.fill_none(padded_jets.mass[:, i], -6.0)
        events = set_ak_column_f32(events, f"jetMass{i+1}", mass_filled)

    for i in range(8):
        btag_score = ak.fill_none(padded_jets.btagUParTAK4B[:, i], -6.0)
        events = set_ak_column_f32(events, f"jetBTagDisc{i+1}", btag_score)

    events = set_ak_column_f32(events, "btag_weight", events.btag_weight)
    if self.dataset_inst.is_mc:
        events = set_ak_column_f32(events, "weights", events.stitched_normalization_weight)
    return events


@producer(
    uses={
        IF_GATJA(gatja_inputs_jet_based_simplified),
        hhh_dl_ml_inputs,
    },
    produces={
        IF_GATJA(gatja_inputs_jet_based_simplified),
        hhh_dl_ml_inputs,
        IF_GATJA(*{f"gatja_output_{i}" for i in range(24)}),
    },
    # produced columns set in the init function
    version=law.config.get_expanded("analysis", "gatja_scores_version", 1),
    sandbox=dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_plotting.sh"),
)
def gatja_scores_jet_based_full_gatja(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    def _safe_lookup(frame: pd.DataFrame, row_labels: Sequence[int], column_names: Sequence[str]) -> np.ndarray:  # noqa
        if len(row_labels) == 0:
            return np.array([], dtype=float)

        subset = frame.loc[row_labels]
        column_index = subset.columns.get_indexer(column_names)
        if np.any(column_index < 0):
            missing = [column_names[index] for index, value in enumerate(column_index) if value < 0]
            raise KeyError(f"Missing neighbour columns: {missing}")
        row_index = np.arange(len(row_labels))
        return subset.to_numpy()[row_index, column_index]

    def _create_graphs_core(df: pd.DataFrame, index: int, drop_empty: bool = True) -> tuple[np.ndarray, np.ndarray]:
        working = df.copy()
        low_index_column = f"jetMinChiHiggsIndex{index + 1}"
        second_index_column = f"jetSecMinChiHiggsIndex{index + 1}"

        working.loc[working[low_index_column] > 7, low_index_column] = index
        working.loc[working[low_index_column] == -6, low_index_column] = index
        working.loc[working[second_index_column] > 7, second_index_column] = index
        working.loc[working[second_index_column] == -6, second_index_column] = index
        node_cols = [
            "jetPT" + str(index + 1),
            "jetEta" + str(index + 1),
            "jetPhi" + str(index + 1),
            "jetMinChiHiggsIndex" + str(index + 1),
            "jetBTagDisc" + str(index + 1),
        ]
        rest_cols = [
            "jetHT", "bjetHT", "lightjetHT",
            "jetNumber", "jetAverageMass",
            "leptonPT1", "leptonEta1", "leptonPhi1",
            "leptonPT2", "leptonEta2", "leptonPhi2",
            "met",
        ]
        btag_weight = working["btag_weight"].to_numpy()
        node_part = working[node_cols].to_numpy()
        rest_part = working[rest_cols].to_numpy()

        low_partner = (working[low_index_column] + 1).astype(int).astype(str)
        second_partner = (working[second_index_column] + 1).astype(int).astype(str)
        low_rows = pd.Series("jetPT" + low_partner, index=working.index)
        neighbour = [
            _safe_lookup(working, low_rows.index, low_rows),
            _safe_lookup(working, low_rows.index, pd.Series("jetEta" + low_partner, index=working.index)),
            _safe_lookup(working, low_rows.index, pd.Series("jetPhi" + low_partner, index=working.index)),
            _safe_lookup(working, low_rows.index, pd.Series("jetBTagDisc" + low_partner, index=working.index)),
        ]
        second_rows = pd.Series("jetPT" + second_partner, index=working.index)
        neighbour2 = [
            _safe_lookup(working, second_rows.index, second_rows),
            _safe_lookup(working, second_rows.index, pd.Series("jetEta" + second_partner, index=working.index)),
            _safe_lookup(working, second_rows.index, pd.Series("jetPhi" + second_partner, index=working.index)),
            _safe_lookup(working, second_rows.index, pd.Series("jetBTagDisc" + second_partner, index=working.index)),
        ]
        graph_data = np.hstack((btag_weight[:, None], node_part, rest_part, np.array(neighbour).T, np.array(neighbour2).T))  # noqa E501å

        return graph_data

    def create_graphs(df: pd.DataFrame, index: int, drop_empty: bool = True) -> tuple[np.ndarray, np.ndarray]:
        return _create_graphs_core(df, index=index, drop_empty=drop_empty)

    def make_optimizer(config: StudioConfig, train_data_length: int):  # noqa
        steps_per_epoch = m.ceil(train_data_length / config.stage_one_batch_size)  # noqa
        total_steps = config.stage_one_epochs * steps_per_epoch
        warmup_steps = int(total_steps * 0.05)
        lr_schedule = WarmupCosineDecay(  # noqa
            initial_lr=config.stage_one_initial_lr(),
            decay_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_lr=1e-7,
        )
        return tf.keras.optimizers.Lamb(learning_rate=lr_schedule)

    def make_model_gnn(input_shape, index_node: int, index_neigh1: int, index_neigh2: int):

        def dense_layer(values, units: int):
            values = layers.Dense(units)(values)
            values = layers.LeakyReLU()(values)
            return values

        def dropout_layer(values, units: int):
            values = layers.Dense(units)(values)
            values = layers.LeakyReLU()(values)
            values = layers.Dropout(0.15)(values)
            values = layers.Concatenate()([values, x_dense])
            return values

        inputs = keras.Input(shape=input_shape)
        input_node_value = inputs[:, :index_node]
        input_neigh1_value = inputs[:, -(index_neigh1 + index_neigh2): -index_neigh2]
        input_neigh2_value = inputs[:, -index_neigh2:]
        input_rest = inputs[:, index_node:-(index_neigh1 + index_neigh2)]
        node_value = dense_layer(input_node_value, 256)
        node_value = layers.Concatenate()([node_value, input_node_value])
        node_value = dense_layer(node_value, 128)

        neigh1_value = dense_layer(input_neigh1_value, 256)
        neigh1_value = layers.Concatenate()([neigh1_value, input_neigh1_value])
        neigh1_value = dense_layer(neigh1_value, 128)

        neigh2_value = dense_layer(input_neigh2_value, 256)
        neigh2_value = layers.Concatenate()([neigh2_value, input_neigh2_value])
        neigh2_value = dense_layer(neigh2_value, 128)

        weight_main = layers.Softmax()(keras.ops.matmul(keras.ops.transpose(node_value), node_value))
        weight_neigh1 = layers.Softmax()(keras.ops.matmul(keras.ops.transpose(node_value), neigh1_value))
        weight_neigh2 = layers.Softmax()(keras.ops.matmul(keras.ops.transpose(node_value), neigh2_value))

        rest = dense_layer(input_rest, 256)
        rest = layers.Concatenate()([rest, input_rest])
        rest = dense_layer(rest, 128)

        node = node_value * weight_main[:, 0]
        neigh1 = neigh1_value * weight_neigh1[:, 0]
        neigh2 = neigh2_value * weight_neigh2[:, 0]

        max_embed = layers.Concatenate()([node, layers.Maximum()([neigh1, neigh2])])
        x_dense = layers.Concatenate()([rest, max_embed])
        x_dense = layers.Dropout(0.15)(x_dense)

        x = dropout_layer(x_dense, 2048)
        x = dropout_layer(x, 2048)
        x = dropout_layer(x, 2048)
        x = dropout_layer(x, 2048)
        x = dropout_layer(x, 1024)
        x = dropout_layer(x, 512)
        x = dropout_layer(x, 128)
        x = dropout_layer(x, 32)
        outputs = layers.Dense(3, activation="softmax")(x)
        return keras.Model(inputs, outputs)

    def load_gatja_model():
        class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, initial_lr, decay_steps, warmup_steps=2, warmup_lr=1e-7, name=None):
                super().__init__()
                self.initial_lr = float(initial_lr)
                self.decay_steps = int(decay_steps)
                self.warmup_steps = int(warmup_steps)
                self.warmup_lr = float(warmup_lr)
                self.name = name

                self.cosine = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=self.initial_lr,
                    decay_steps=self.decay_steps,
                )

            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                warmup_steps = tf.cast(self.warmup_steps, tf.float32)

                def warmup():
                    # linear warmup von warmup_lr -> initial_lr
                    return self.warmup_lr + (self.initial_lr - self.warmup_lr) * (step / warmup_steps)

                def decay():
                    return self.cosine(step - warmup_steps)

                return tf.cond(step < warmup_steps, warmup, decay)

            def get_config(self):
                # muss serialisierbar sein
                return {
                    "initial_lr": self.initial_lr,
                    "decay_steps": self.decay_steps,
                    "warmup_steps": self.warmup_steps,
                    "warmup_lr": self.warmup_lr,
                    "name": self.name,
                }

        model = tf.keras.models.load_model(
            "/data/dust/user/weidnerb/Code setup after CMS week/New_labels/evaluation_von_4_überarbeitung_der_Inputs_includieren_von_ttbb/Training_1/save_gatja_main_best_v3_jet_based.keras",  # noqa E501
            custom_objects={"WarmupCosineDecay": WarmupCosineDecay},
            compile=False,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Lamb(learning_rate=0.0001),
            loss=tf.keras.losses.CategoricalFocalCrossentropy(
                alpha=[2.5, 0.7, 0.9],
                gamma=1.5,
                from_logits=False,
            ),
            metrics=["accuracy"],
        )
        return model

    def predict_all_jets(events_in: ak.Array, df_all: pd.DataFrame, model, scalers,
                        evt_pos_filtered, n_jets=8):

        robust_scaler, quantile_scaler, minmax_scaler = scalers

        jet_pred_dfs = []

        for jet_idx in range(n_jets):
            jet_pt_col = f"jetPT{jet_idx + 1}"

            keep_mask = df_all[jet_pt_col] != -6
            mask_np = keep_mask.to_numpy()
            if np.sum(mask_np) == 0:
                continue
            else:
                df_kept = df_all.loc[keep_mask].reset_index(drop=True)
                pos_kept = ak.to_numpy(evt_pos_filtered[mask_np]).astype(np.int64)

                sample_block = create_graphs(df_kept, jet_idx, drop_empty=False)

                X_raw = sample_block[:, 1:]  # remove btag_weight
                try:
                    robust_scaler.transform(X_raw)
                except ValueError as e:
                    __import__("IPython").embed()
                    raise e
                X_scaled = minmax_scaler.transform(
                    quantile_scaler.transform(
                        robust_scaler.transform(X_raw),
                    ),
                )
                y_pred_prob = model.predict(X_scaled, batch_size=4096, verbose=0)
                jet_pred_df = pd.DataFrame({
                    "evt_pos": pos_kept,
                    "jet_idx": jet_idx,
                    "prob_higgs": y_pred_prob[:, 0],
                    "prob_top": y_pred_prob[:, 1],
                    "prob_other": y_pred_prob[:, 2],
                })
                jet_pred_dfs.append(jet_pred_df)

        return pd.concat(jet_pred_dfs, ignore_index=True)

    def attach_outputs(events_in: ak.Array, pred_df: pd.DataFrame, n_jets=8):
        n_events = len(events_in)
        out_arrays = {i: np.full(n_events, -10.0, dtype=np.float32) for i in range(n_jets * 3)}

        jets_allowed = np.asarray(ak.to_numpy(events_in.jetNumber) >= 3, dtype=bool)

        # pred_df contains evt_pos in [0, n_events)
        for row in pred_df.itertuples(index=False):
            ievent = int(row.evt_pos)
            if ievent < 0 or ievent >= n_events:
                continue
            if not jets_allowed[ievent]:
                continue

            j = int(row.jet_idx)
            out_arrays[j * 3 + 0][ievent] = float(row.prob_higgs)
            out_arrays[j * 3 + 1][ievent] = float(row.prob_top)
            out_arrays[j * 3 + 2][ievent] = float(row.prob_other)

        events_out = events_in
        for out_i, arr in out_arrays.items():
            events_out = set_ak_column_f32(events_out, f"gatja_output_{out_i}", arr)
        return events_out

    def load_scalers():
        import pickle
        robust_scaler = pickle.load(open("/data/dust/user/markusla/public/hh2bbww/gatja_scaler/robust_scaler.pkl", "rb"))  # noqa E501
        quantile_scaler = pickle.load(open("/data/dust/user/markusla/public/hh2bbww/gatja_scaler/quantile_scaler.pkl", "rb"))  # noqa E501
        minmax_scaler = pickle.load(open("/data/dust/user/markusla/public/hh2bbww/gatja_scaler/minmax_scaler.pkl", "rb"))  # noqa E501
        return robust_scaler, quantile_scaler, minmax_scaler

    events = self[hhh_dl_ml_inputs](events, **kwargs)
    if self.has_dep(gatja_inputs_jet_based_simplified):
        events = self[gatja_inputs_jet_based_simplified](events, **kwargs)
        evt_pos = ak.local_index(events.jetNumber)
        keep_events = events.jetNumber >= 3
        events_filtered = events[keep_events]
        evt_pos_filtered = evt_pos[keep_events]
        zero_padding = ak.full_like(events.jetNumber, -6)
        idx = ak.local_index(events.jetNumber)
        gatja_idx = ak.where((events.jetNumber >= 3), idx, zero_padding)
        events = set_ak_column_f32(events, "gatja_idx", gatja_idx)
        gatja_input_list = [
            "weights", "btag_weight", "jetPT1", "jetPT2", "jetPT3", "jetPT4", "jetPT5", "jetPT6", "jetPT7", "jetPT8", "jetEta1", "jetEta2",  # noqa E501
            "jetEta3", "jetEta4", "jetEta5", "jetEta6", "jetEta7", "jetEta8", "jetBTagDisc1", "jetBTagDisc2", "jetBTagDisc3", "jetBTagDisc4", "jetBTagDisc5", "jetBTagDisc6",  # noqa E501
            "jetBTagDisc7", "jetBTagDisc8", "jetMinChiHiggsIndex1", "jetMinChiHiggsIndex2", "jetMinChiHiggsIndex3", "jetMinChiHiggsIndex4", "jetMinChiHiggsIndex5", "jetMinChiHiggsIndex6", "jetMinChiHiggsIndex7",  # noqa E501
            "jetMinChiHiggsIndex8", "leptonPT1", "leptonEta1", "leptonPT2",
            "leptonEta2", "leptonPhi1", "leptonPhi2", "bjetAverageMass"," jetAverageMass", "bjetAverageMassSqr"," jetHT", "bjetHT", "lightjetHT"," jetNumber","bjetNumber",  # noqa E501
            "jetPhi1", "jetPhi2", "jetPhi3", "jetPhi4", "jetPhi5", "jetPhi6", "jetPhi7", "jetPhi8", "averageDeltaEtabb", "minDeltaRjj", "minDeltaRbb", "maxDeltaEtabb","maxDeltaEtajj", "maxDeltaEtabj", "minDeltaRbj",  # noqa E501
            "averageDeltaEtabj", "averageDeltaRbj", "minDeltaRMassjj", "minDeltaRMassbb", "minDeltaRMassbj", "minDeltaRpTjj", "minDeltaRpTbb", "minDeltaRpTbj", "maxPTmassjjj", "maxPTmassjbb", "met", "metPhi",  # noqa E501
            "jetSecMinChiHiggsIndex1", "jetSecMinChiHiggsIndex2", "jetSecMinChiHiggsIndex3", "jetSecMinChiHiggsIndex4", "jetSecMinChiHiggsIndex5", "jetSecMinChiHiggsIndex6", "jetSecMinChiHiggsIndex7", "jetSecMinChiHiggsIndex8",  # noqa E501
        ]

        import pandas as pd
        event_id_arr = ak.to_numpy(events_filtered["event_id"]).astype(np.int64)

        data = {"event_id": event_id_arr}

        for col in gatja_input_list:
            data[col] = ak.to_numpy(events_filtered[col])

        df_all = pd.DataFrame(data)

        # import sklearn
        robust_scaler, quantile_scaler, minmax_scaler = load_scalers()

        scalers = (robust_scaler, quantile_scaler, minmax_scaler)
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        model = load_gatja_model()

        pred_df = predict_all_jets(events_filtered, df_all, model, scalers, evt_pos_filtered)
        events = attach_outputs(events, pred_df)

    else:
        output_cols = [f"gatja_output_{i}" for i in range(23)]
        for col in output_cols:
            events = set_ak_column_f32(events, col, ak.full_like(events.mli_n_jet, -10))  # noqa E501

    return events


@gatja_scores_jet_based_full_gatja.init
def gatja_scores_jet_based_full_gatja_init(self: Producer) -> None:
    add_gatja_scores_variables(self.config_inst)
