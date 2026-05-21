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

from hbw.tasks.ml import ProduceColumnsTF


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
        # "*", "*.*",
        prepare_objects,
        btag_wp_weights,
        common_ml_inputs,
        prepare_hhh_bjets,
        event_weights,
        #b_gen_matching,
        "Jet.*",
        #"GenJet.*", 
        # "Jet.bjetHiggsMatched", "Jet.bjetTopMatched", "Jet.bjetZMatched",
    },
    produces={"bjetPT1", "bjetPT2", "bjetPT3", "bjetPT4", "bjetPT5", "bjetPT6", "bjetPT7", "bjetPT8",
              "bjetEta1", "bjetEta2", "bjetEta3", "bjetEta4", "bjetEta5", "bjetEta6", "bjetEta7", "bjetEta8",
              "leptonPT1", "leptonEta1", "leptonPT2", "leptonEta2", "leptonPhi1", "leptonPhi2",
              "bjetAverageMass", "jetAverageMass", 
              "bjetAverageMassSqr", "jetHT", "bjetHT", "lightjetHT", "jetNumber", "bjetNumber",
              "bjetPhi1", "bjetPhi2", "bjetPhi3", "bjetPhi4", "bjetPhi5", "bjetPhi6", "bjetPhi7", "bjetPhi8",
              "averageDeltaEtabb", "minDeltaRjj", 
              "minDeltaRbb", 
              "maxDeltaEtabb", "maxDeltaEtajj", "maxDeltaEtabj",
              "minDeltaRbj", "averageDeltaEtabj", "averageDeltaRbj", "minDeltaRMassjj", "minDeltaRMassbb", "minDeltaRMassbj",
              "minDeltaRpTjj", "minDeltaRpTbb", "minDeltaRpTbj", "maxPTmassjjj", "maxPTmassjbb", "met", "metPhi",
              #bjetSecMinChiHiggsIndex
              #bjetTopMatched
              "minDeltaRbb_GATJA",
              "bjetMinChiHiggsIndex1", "bjetSecMinChiHiggsIndex1", "bjetMinChiHiggsIndex2", "bjetSecMinChiHiggsIndex2", "bjetMinChiHiggsIndex3", "bjetSecMinChiHiggsIndex3", "bjetMinChiHiggsIndex4", "bjetSecMinChiHiggsIndex4",
              "bjetMinChiHiggsIndex5", "bjetSecMinChiHiggsIndex5", "bjetMinChiHiggsIndex6", "bjetSecMinChiHiggsIndex6", "bjetMinChiHiggsIndex7", "bjetSecMinChiHiggsIndex7", "bjetMinChiHiggsIndex8", "bjetSecMinChiHiggsIndex8",
              "bjetBTagDisc1", "bjetBTagDisc2", "bjetBTagDisc3", "bjetBTagDisc4", "bjetBTagDisc5", "bjetBTagDisc6", "bjetBTagDisc7", "bjetBTagDisc8", "btag_weight", "weights",
            #   "bjetTopMatched1", "bjetTopMatched2", "bjetTopMatched3", "bjetTopMatched4", "bjetTopMatched5", "bjetTopMatched6","bjetTopMatched7", "bjetTopMatched8", 
            #   "bjetHiggsMatched1", "bjetHiggsMatched2", "bjetHiggsMatched3", "bjetHiggsMatched4", "bjetHiggsMatched5", "bjetHiggsMatched6", "bjetHiggsMatched7", "bjetHiggsMatched8",
            #   "bjetZMatched1", "bjetZMatched2", "bjetZMatched3", "bjetZMatched4", "bjetZMatched5", "bjetZMatched6", "bjetZMatched7", "bjetZMatched8", 
              "bjetsMass12", "bjetsMass13", "bjetsMass14", "bjetsMass15", "bjetsMass16", "bjetsMass17", "bjetsMass18",
              "bjetsMass23", "bjetsMass24", "bjetsMass25", "bjetsMass26", "bjetsMass27", "bjetsMass28",
              "bjetsMass34", "bjetsMass35", "bjetsMass36", "bjetsMass37", "bjetsMass38",
              "bjetsMass45", "bjetsMass46", "bjetsMass47", "bjetsMass48",
              "bjetsMass56", "bjetsMass57", "bjetsMass58",
              "bjetsMass67", "bjetsMass68",
              "bjetsMass78"
    },
)

def gatja_inputs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    mass_higgs = 125.0
    sigma = 10.0 #PLATZHALTER FÜR DIE MASSENAUFLÖSUNG -> MUSS NOCH BESTIMMT WERDEN
    

    # produce common input features
    events = self[common_ml_inputs](events, **kwargs)
    events = self[prepare_hhh_bjets](events, **kwargs)
    events = self[event_weights](events, **kwargs)
    #events = self[b_gen_matching](events, **kwargs)

    #from IPython import embed
    #embed()

    # add behavior and define new collections (e.g. Lepton)
    events = self[prepare_objects](events, **kwargs)
    #__import__("IPython").embed()
    jet_mask = (events.Jet["pt"] < 10_000) & (abs(events.Jet["eta"]) < 2.5) #Prüfen, ob die Selection auch bei GATJA angewendet wird
    events = self[btag_wp_weights](events, jet_mask=jet_mask, **kwargs)
    padded_jets = ak.pad_none(events.Jet, 8)
    padded_lepton = ak.pad_none(events.Lepton, 2)
    #padded_bjets = ak.pad_none(events.BtaggedJet, 8)
    padded_bjets = ak.pad_none(events.BtaggedJet[:,:8], 8, axis=1)
    
    #__import__("IPython").embed()

    j1, j2 = ak.unzip(ak.combinations(events.Jet,2))
    d_eta_jet_pairs = (j1.eta - j2.eta)
    d_phi_jet_pairs = (j1.delta_phi(j2))
    d_phi_jet_pairs_GATJA = (j1.phi - j2.phi)
    deltaR_jj = j1.delta_r(j2)

    b1, b2 = ak.unzip(ak.combinations(events.BtaggedJet, 2))
    d_eta_bjet_pairs = (b1.eta - b2.eta)
    d_phi_bjet_pairs = (b1.delta_phi(b2))
    d_phi_bjet_pairs_GATJA = (b1.phi - b2.phi)  
    deltaR_bb = b1.delta_r(b2)
    DeltaRbb_GATJA = ((d_eta_bjet_pairs)**2 + (d_phi_bjet_pairs_GATJA)**2)**0.5

   

    b_mix, j_mix = ak.unzip(ak.cartesian([events.BtaggedJet, events.Jet]))
    deltaR_bj = b_mix.delta_r(j_mix)

    j1_3, j2_3, j3_3 = ak.unzip(ak.combinations(events.Jet, 3))
    
    j_cart, bb_cart = ak.unzip(ak.cartesian([events.Jet,ak.combinations(events.BtaggedJet, 2)]))
    b1_cart, b2_cart = ak.unzip(bb_cart)
  
    #__import__("IPython").embed()

    for i in range(8):
        events = set_ak_column_f32(events, f"bjetPT{i+1}", ak.fill_none(padded_bjets["pt"][:, i], -6))
    for i in range(8):
        events = set_ak_column_f32(events, f"bjetEta{i+1}", ak.fill_none(padded_bjets["eta"][:, i], -6))
    #bjetBTagDisc
    #bjetMinChiHiggsIndex
    #bjetHiggsMatched
    for i in range(2):
        events = set_ak_column_f32(events, f"leptonPT{i+1}", ak.fill_none(padded_lepton["pt"][:,i], -6))
        events = set_ak_column_f32(events, f"leptonEta{i+1}", ak.fill_none(padded_lepton["eta"][:,i], -6))
    for i in range(2):
        events = set_ak_column_f32(events, f"leptonPhi{i+1}", ak.fill_none(padded_lepton["phi"][:,i], -6))
    bjetaveragemass=ak.mean(events.BtaggedJet.mass, axis=1)
    events = set_ak_column_f32(events, "bjetAverageMass", ak.fill_none(ak.nan_to_none(bjetaveragemass), -6))
    events = set_ak_column_f32(events, "jetAverageMass", ak.where(ak.num(events.Jet)>0, ak.sum(events.Jet.mass, axis=1)/ak.num(events.Jet), 0))
    events = set_ak_column_f32(events, "bjetAverageMassSqr", ak.fill_none(ak.nan_to_none(bjetaveragemass*bjetaveragemass*ak.num(events.BtaggedJet)), -6))
    events = set_ak_column_f32(events, "jetHT", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column_f32(events, "bjetHT", ak.sum(events.BtaggedJet.pt, axis=1))
    events = set_ak_column_f32(events, "lightjetHT", ak.sum(events.Lightjet.pt, axis=1))
    events = ak.with_field(events, ak.num(events.Jet), "jetNumber")
    events = ak.with_field(events, ak.num(events.BtaggedJet), "bjetNumber")
    for i in range(8):
        events = set_ak_column_f32(events, f"bjetPhi{i+1}", ak.fill_none(padded_bjets["phi"][:, i], -6))
    events = set_ak_column_f32(events, "averageDeltaEtabb", ak.fill_none(ak.nan_to_none(ak.mean(abs(d_eta_bjet_pairs), axis=1)), -6))
    events = set_ak_column_f32(events, "minDeltaRjj", ak.fill_none(ak.nan_to_none(ak.min(deltaR_jj, axis=1)), -6))
    events = set_ak_column_f32(events, "minDeltaRbb", ak.fill_none(ak.nan_to_none(ak.min(deltaR_bb, axis=1)), -6))
    events = set_ak_column_f32(events, "minDeltaRbb_GATJA", ak.fill_none(ak.nan_to_none(ak.min(DeltaRbb_GATJA, axis=1)), -6))

    events = set_ak_column_f32(events, "maxDeltaEtabb", ak.fill_none(ak.nan_to_none(ak.max(abs(d_eta_bjet_pairs), axis=1)), -6))
    events = set_ak_column_f32(events, "maxDeltaEtajj", ak.fill_none(ak.nan_to_none(ak.max(abs(d_eta_jet_pairs), axis=1)), -6))
    events = set_ak_column_f32(events, "maxDeltaEtabj", ak.fill_none(ak.nan_to_none(ak.max(abs(b_mix.eta - j_mix.eta), axis=1)), -6))
    events = set_ak_column_f32(events, "minDeltaRbj", ak.fill_none(ak.nan_to_none(ak.min(deltaR_bj, axis=1)), -6))

    events = set_ak_column_f32(events, "averageDeltaEtabj", ak.fill_none(ak.nan_to_none(ak.mean(abs(b_mix.eta - j_mix.eta), axis=1)), -6))
    events = set_ak_column_f32(events, "averageDeltaRbj", ak.fill_none(ak.nan_to_none(ak.mean(deltaR_bj, axis=1)), -6))

    
    
    mask_min_dR_jj = deltaR_jj == ak.min(deltaR_jj, axis=1, keepdims=True)
    events = set_ak_column_f32(events, "minDeltaRMassjj", ak.fill_none(ak.firsts((j1+j2).mass[mask_min_dR_jj]), -6)) #Invariant mass of jet pair with smallest ΔR
    mask_min_dR_bb = deltaR_bb == ak.min(deltaR_bb, axis=1, keepdims=True)
    events = set_ak_column_f32(events, "minDeltaRMassbb", ak.fill_none(ak.firsts((b1+b2).mass[mask_min_dR_bb]), -6)) #Invariant mass of b-jet pair with smallest ΔR
    mask_min_dR_bj = deltaR_bj == ak.min(deltaR_bj, axis=1, keepdims=True)
    events = set_ak_column_f32(events, "minDeltaRMassbj", ak.fill_none(ak.firsts((b_mix + j_mix).mass[mask_min_dR_bj]), -6)) #Invariant mass of jet+bjet-pair pair with

    events = set_ak_column_f32(events, "minDeltaRpTjj", ak.fill_none(ak.firsts((j1.pt + j2.pt)[mask_min_dR_jj]), -6)) #Combined transverse momentum of jet pair with smallest ΔR
    events = set_ak_column_f32(events, "minDeltaRpTbb", ak.fill_none(ak.firsts((b1.pt + b2.pt)[mask_min_dR_bb]), -6)) #Combined transverse momentum of b-jet pair with smallest ΔR 
    events = set_ak_column_f32(events, "minDeltaRpTbj", ak.fill_none(ak.firsts((b_mix.pt + j_mix.pt)[mask_min_dR_bj]), -6)) #Combined transverse momentum of jet+bjet-pair pair with smallest ΔR

    pt_jjj = j1_3.pt + j2_3.pt + j3_3.pt
    mask_max_pT_jjj = (pt_jjj) == ak.max(pt_jjj, axis=1, keepdims=True)
    events = set_ak_column_f32(events, "maxPTmassjjj", ak.fill_none(ak.firsts((j1_3+j2_3+j3_3).mass[mask_max_pT_jjj]), -6)) #Mass of 3-jet system with highest total pT (boosted object candidate)
    pt_jbb = j_cart.pt + b1_cart.pt + b2_cart.pt
    mask_max_pT_jbb = (pt_jbb) == ak.max(pt_jbb, axis=1, keepdims=True)
    events = set_ak_column_f32(events, "maxPTmassjbb", ak.fill_none(ak.firsts((j_cart + b1_cart + b2_cart).mass[mask_max_pT_jbb]), -6)) # Mass of system (1 jet + 2 b-jets) with highest total pT

    events = set_ak_column_f32(events, "met", events.mli_met_pt)
    events = set_ak_column_f32(events, "metPhi", events.mli_met_phi)

    for i in range(8):
        events = set_ak_column_f32(events, f"bjetBTagDisc{i+1}", padded_bjets.b_score[:,i])
        
    events = set_ak_column_f32(events, "btag_weight", events.btag_weight)   
    events = set_ak_column_f32(events, "weights", events. stitched_normalization_weight) 

    
    min_3_bjets = ak.num(events.BtaggedJet, axis=1) >= 3
    

    bjets_i = padded_bjets[:, :, np.newaxis] # (Events, Jets, 1) Spaltenvektor
    bjets_j = padded_bjets[:, np.newaxis, :] # (Events, 1, Jets) Zeilenvektor

    dibjet = (bjets_i + bjets_j)
    dibjet_mass = ak.without_parameters(dibjet.mass)

    idx = ak.local_index(padded_bjets) #Generate local index for each bjet in each event to aviod later that one bjet is combined with itself
    idx_i = idx[:, :, np.newaxis]
    idx_j = idx[:, np.newaxis, :]

    valid_i = ~ak.is_none(bjets_i.pt)
    valid_j = ~ak.is_none(bjets_j.pt)

    pair_valid = valid_i & valid_j & (idx_i != idx_j)

    chi2_matrix = ((dibjet_mass - mass_higgs) / sigma)**2
    chi2_matrix = ak.where(pair_valid, chi2_matrix, np.inf)


    min_idx = ak.argmin(chi2_matrix, axis=2)
    min_idx_filled = ak.fill_none(min_idx, -6)
    
    
    mask_sec = idx_j != min_idx_filled[:, :, np.newaxis]
    chi2_matrix_sec = ak.where(mask_sec, chi2_matrix, np.inf)
    sec_min_idx = ak.argmin(chi2_matrix_sec, axis=2)
    sec_min_idx_filled = ak.fill_none(sec_min_idx,-6)

    event_mask_2d = min_3_bjets[:, np.newaxis]
    min_idx_final = ak.where(event_mask_2d, min_idx_filled, -999)
    sec_min_idx_final = ak.where(event_mask_2d, sec_min_idx_filled, -999)

    real_jet = ~ak.is_none(padded_bjets.pt)
    
    n_bjets = ak.num(events.BtaggedJet, axis=1)
    min_3_bjets = n_bjets >= 3

   

    for i in range(8):
        jet_exists = n_bjets > i

        mi = min_idx_final[:,i]
        si = sec_min_idx_final[:,i]

        #real = n_bjets > i

        mi_out = ak.where(jet_exists, mi, -6)
        si_out = ak.where(jet_exists, si, -6)

        mi_out = ak.where(min_3_bjets, mi_out, -999)
        si_out = ak.where(min_3_bjets, si_out, -999)

        #mi_final = ak.where(min_3_bjets, mi_out, -999)
        #si_final = ak.where(min_3_bjets, si_out, -999)

        events = set_ak_column_f32(events, f"bjetMinChiHiggsIndex{i+1}", mi_out)
        events = set_ak_column_f32(events, f"bjetSecMinChiHiggsIndex{i+1}", si_out)
   

    for i in range(8):
        mass_filled = ak.fill_none(padded_bjets.mass[:, i], -6.0)
        mass_final = ak.where(min_3_bjets, mass_filled, -999.0)
        events = set_ak_column_f32(events, f"b_jet_mass{i+1}", mass_final)
    #from IPython import embed
    #embed()
    for i in range(8):
        btag_score = ak.fill_none(padded_bjets.b_score[:, i], -6.0)
        btag_score_final = ak.where(min_3_bjets, btag_score, -999.0)
        events = set_ak_column_f32(events, f"bjetBTagDisc{i+1}", btag_score_final)
        #events = set_ak_column_f32(events, f"bjetBTagDisc{i+1}", bjets_btag[:,i])
        
    events = set_ak_column_f32(events, f"btag_weight", events.btag_weight)   
    events = set_ak_column_f32(events, f"weights", events.stitched_normalization_weight)

    #from IPython import embed
    #embed()
    # top = ak.fill_none(ak.pad_none(events.BtaggedJet.bjetTopMatched, 8, clip=True), 0)
    # z = ak.fill_none(ak.pad_none(events.BtaggedJet.bjetZMatched, 8, clip=True), 0)
    # higgs = ak.fill_none(ak.pad_none(events.BtaggedJet.bjetHiggsMatched, 8, clip=True), 0)

   

    # for i in range(8): 
    #     events = set_ak_column_f32(events, f"bjetTopMatched{i+1}", top[:,i])
    #     events = set_ak_column_f32(events, f"bjetZMatched{i+1}", z[:,i])
    #     events = set_ak_column_f32(events, f"bjetHiggsMatched{i+1}", higgs[:,i])

    
    pair_mass = (b1+ b2).mass
    pair_mass = ak.pad_none(pair_mass, 28, axis=1)
    pair_mass = ak.fill_none(pair_mass, -6)

    k = 0
    
    for i in range(1,9):
        for j in range(i+1,9):
            col_name = (f"bjetsMass{i}{j}")
            current_masses = pair_mass[:,k]
            events = set_ak_column_f32(events, col_name, current_masses)

            k += 1

    # __import__("IPython").embed()
    return events

# @gatja_inputs.init
# def gatja_inputs_init(self: Producer) -> None:
    # add_variable_matching_GATJA2(self.config_inst)


@producer(
    uses={
        IF_GATJA(gatja_inputs), 
        hhh_dl_ml_inputs,
    },
    produces={
        IF_GATJA(gatja_inputs), 
        hhh_dl_ml_inputs,
        IF_GATJA(*{f"gatja_output_{i}" for i in range(23)}),
        # Here die scores die produced werden
    },
    # produced columns set in the init function
    version=law.config.get_expanded("analysis", "gatja_scores_version", 0),
)
def gatja_scores(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    def create_graphs(df, index, drop_empty=True):
        import tensorflow
        import pandas as pd
        # Get the column names for the minimum and second minimum Higgs chi2 matching indices
        col_lp = "bjetMinChiHiggsIndex" + str(index + 1)
        col_sp = "bjetSecMinChiHiggsIndex" + str(index + 1)

        # If the min and second min do not exist or are not among the first 8 btagged jets,
        # assign the jet itself as its neighbour.
        df.loc[df[col_lp] > 7, col_lp] = index
        df.loc[df[col_lp] == -6, col_lp] = index
        df.loc[df[col_sp] > 7, col_sp] = index
        df.loc[df[col_sp] == -6, col_sp] = index

        # Recalculate index series as strings after adjustment
        index_lp = (df[col_lp] + 1).astype(int).astype(str)
        index_sp = (df[col_sp] + 1).astype(int).astype(str)

        # Build the main index list with explicit column names
        index_main = [
            "bjetPT" + str(index + 1),
            "bjetEta" + str(index + 1),
            "bjetPhi" + str(index + 1),
            "bjetMinChiHiggsIndex" + str(index + 1),
            "bjetBTagDisc" + str(index + 1),
            "jetAverageMass", "bjetAverageMassSqr",
            "jetHT", "bjetHT", "lightjetHT", "jetNumber", "bjetNumber",
            "leptonPT1", "leptonEta1", "leptonPhi1",
            "leptonPT2", "leptonEta2", "leptonPhi2",
            "met",
        ]

        main = df[index_main].copy()
        na_index = []
        if drop_empty:
            query_str = "-6 == bjetPT" + str(index + 1)
            na_index = main.query(query_str).index
        main = main.drop(na_index)

        print("index is : ", index_main)
        # Replace np.char.add with a list comprehension to ensure consistent string types.
        combined = np.array(["bjetPT" + s for s in index_lp])
        print("the dataframe : ", np.sum(combined == "bjetPT0"))

        # # Prepare label series and drop the same indices
        # label_higgs = df["bjetHiggsMatched" + str(index + 1)].drop(na_index)
        # label_top = df["bjetTopMatched" + str(index + 1)].drop(na_index)
        # label_others = ~np.logical_or(label_top, label_higgs)

        # Function to replace df.lookup using index positions.
        def safe_lookup(df, row_labels, col_series):
            subset = df.loc[row_labels]
            col_idx = subset.columns.get_indexer(col_series.values)
            row_idx = np.arange(len(row_labels))
            return subset.to_numpy()[row_idx, col_idx]

        # Function to obtain neighbour values given a column prefix and an index series.
        def get_neighbour_values(prefix, idx_series):
            # Use simple string concatenation here.
            col_series = pd.Series(prefix + idx_series, index=df.index).drop(na_index)
            return safe_lookup(df, col_series.index, col_series)

        # Obtain neighbour values for the min index.
        neighbour = [
            get_neighbour_values("bjetPT", index_lp),
            get_neighbour_values("bjetEta", index_lp),
            get_neighbour_values("bjetPhi", index_lp),
            get_neighbour_values("bjetBTagDisc", index_lp)
        ]

        # Obtain neighbour values for the second min index.
        neighbour2 = [
            get_neighbour_values("bjetPT", index_sp),
            get_neighbour_values("bjetEta", index_sp),
            get_neighbour_values("bjetPhi", index_sp),
            get_neighbour_values("bjetBTagDisc", index_sp)
        ]

        graph_data = np.hstack((main.to_numpy(),
                                np.array(neighbour).T,
                                np.array(neighbour2).T))
        

        return graph_data


    def make_model_gatja(input_shape, index_node, index_neigh1, index_neigh2):
        import tensorflow
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Input, Dense, Dropout
        inputs = keras.Input(shape=input_shape)

        # Extract node, neighbor1 and neighbor2 values from inputs using Lambda layers
        input_node_value = layers.Lambda(lambda x: x[:, :index_node])(inputs)
        input_neigh1_value = layers.Lambda(lambda x: x[:, -(index_neigh1+index_neigh2):-index_neigh2])(inputs)
        input_neigh2_value = layers.Lambda(lambda x: x[:, -index_neigh2:])(inputs)

        # Define a function to create dense layers with LeakyReLU activation
        def dense_layer(x, units):
            x = layers.Dense(units)(x)
            x = layers.LeakyReLU()(x)
            return x

        # Process node embedding
        node_value = dense_layer(input_node_value, 256)
        node_value = layers.Concatenate()([node_value, input_node_value])
        node_value = dense_layer(node_value, 128)

        # Process neighbor1 embedding
        neigh1_value = dense_layer(input_neigh1_value, 256)
        neigh1_value = layers.Concatenate()([neigh1_value, input_neigh1_value])
        neigh1_value = dense_layer(neigh1_value, 128)

        # Process neighbor2 embedding
        neigh2_value = dense_layer(input_neigh2_value, 256)
        neigh2_value = layers.Concatenate()([neigh2_value, input_neigh2_value])
        neigh2_value = dense_layer(neigh2_value, 128)

        # Compute attention scores per sample using element-wise dot products
        # For each sample, compute a scalar score by taking the dot product along the features.
        node_score = layers.Lambda(
            lambda x: tensorflow.reduce_sum(x[0] * x[1], axis=-1, keepdims=True),
            output_shape=(1,)
        )([node_value, node_value])
        neigh1_score = layers.Lambda(
            lambda x: tensorflow.reduce_sum(x[0] * x[1], axis=-1, keepdims=True),
            output_shape=(1,)
        )([node_value, neigh1_value])
        neigh2_score = layers.Lambda(
            lambda x: tensorflow.reduce_sum(x[0] * x[1], axis=-1, keepdims=True),
            output_shape=(1,)
        )([node_value, neigh2_value])

        # Concatenate scores to shape (batch_size, 3) and apply softmax
        scores = layers.Concatenate(axis=-1)([node_score, neigh1_score, neigh2_score])
        attention_weights = layers.Softmax()(scores)

        # Extract individual attention weights; each will have shape (batch_size, 1)
        node_weight = layers.Lambda(lambda x: x[:, 0:1])(attention_weights)
        neigh1_weight = layers.Lambda(lambda x: x[:, 1:2])(attention_weights)
        neigh2_weight = layers.Lambda(lambda x: x[:, 2:3])(attention_weights)

        # Apply the attention weights (element-wise multiplication)
        node = layers.Multiply()([node_value, node_weight])
        neigh1 = layers.Multiply()([neigh1_value, neigh1_weight])
        neigh2 = layers.Multiply()([neigh2_value, neigh2_weight])

        # Concatenate node embedding with the maximum of neighbor embeddings
        max_embed = layers.Concatenate()([node, layers.Maximum()([neigh1, neigh2])])

        # Extract the rest of the input features using a Lambda layer
        input_rest = layers.Lambda(lambda x: x[:, index_node:-(index_neigh1+index_neigh2)])(inputs)
        rest = dense_layer(input_rest, 256)
        rest = layers.Concatenate()([rest, input_rest])
        rest = dense_layer(rest, 128)

        # Concatenate rest with the attended node and neighbor features
        x_dense = layers.Concatenate()([rest, max_embed])
        x_dense = layers.Dropout(0.15)(x_dense)

        # Define a function to create dropout layers with LeakyReLU activation and concatenation
        def dropout_layer(x, units):
            x_new = layers.Dense(units)(x)
            x_new = layers.LeakyReLU()(x_new)
            x_new = layers.Dropout(0.15)(x_new)
            # Concatenate with the original x_dense for residual-like connection
            x_new = layers.Concatenate()([x_new, x_dense])
            return x_new

        x = dropout_layer(x_dense, 512)
        x = dropout_layer(x, 128)
        x = dropout_layer(x, 32)


        # Final output layer with softmax activation for 3 classes
        outputs = layers.Dense(3, activation="softmax")(x)

        return keras.Model(inputs, outputs)

    events = self[hhh_dl_ml_inputs](events, **kwargs)
    if self.has_dep(gatja_inputs):
        import tensorflow
        events = self[gatja_inputs](events, **kwargs)

        zero_padding = ak.full_like(events.bjetNumber, -6)
        idx = ak.local_index(events.bjetNumber)
        gatja_idx = ak.where((events.bjetNumber >= 3), idx, zero_padding)
        events = set_ak_column_f32(events, "gatja_idx", gatja_idx)

        # logger.info(f"evaluating model {str(self.ml_model_inst)} for process {process} and fold {self.fold}")
        gatja_model = make_model_gatja((27,),5,4,4)
        gatja_model.load_weights("/data/dust/user/markusla/analysis/dilepton/hh2bbww/tutorial_onwn_Data_hhh.weights.h5")

        gatja_input_list = [
            'bjetPT1',
            'bjetPT2',
            'bjetPT3',
            'bjetPT4',
            'bjetPT5',
            'bjetPT6',
            'bjetPT7',
            'bjetPT8',
            'bjetEta1',
            'bjetEta2',
            'bjetEta3',
            'bjetEta4',
            'bjetEta5',
            'bjetEta6',
            'bjetEta7',
            'bjetEta8',
            'bjetBTagDisc1',
            'bjetBTagDisc2',
            'bjetBTagDisc3',
            'bjetBTagDisc4',
            'bjetBTagDisc5',
            'bjetBTagDisc6',
            'bjetBTagDisc7',
            'bjetBTagDisc8',
            'leptonPT1',
            'leptonEta1',
            'leptonPT2',
            'leptonEta2',
            'leptonPhi1',
            'leptonPhi2',
            'bjetAverageMass',
            'jetAverageMass',
            'bjetAverageMassSqr',
            'jetHT',
            'bjetHT',
            'lightjetHT',
            'jetNumber',
            'bjetNumber',
            'bjetPhi1',
            'bjetPhi2',
            'bjetPhi3',
            'bjetPhi4',
            'bjetPhi5',
            'bjetPhi6',
            'bjetPhi7',
            'bjetPhi8',
            'averageDeltaEtabb',
            'minDeltaRjj',
            'minDeltaRbb',
            'maxDeltaEtabb',
            'maxDeltaEtajj',
            'maxDeltaEtabj',
            'averageDeltaEtabj',
            'averageDeltaRbj',
            'minDeltaRMassjj',
            'minDeltaRMassbb',
            'minDeltaRMassbj',
            'minDeltaRpTjj',
            'minDeltaRpTbb',
            'minDeltaRpTbj',
            'maxPTmassjjj',
            'maxPTmassjbb',
            'met',
            'bjetMinChiHiggsIndex1',
            'bjetMinChiHiggsIndex2',
            'bjetMinChiHiggsIndex3',
            'bjetMinChiHiggsIndex4',
            'bjetMinChiHiggsIndex5',
            'bjetMinChiHiggsIndex6',
            'bjetMinChiHiggsIndex7',
            'bjetMinChiHiggsIndex8',
            'bjetSecMinChiHiggsIndex1',
            'bjetSecMinChiHiggsIndex2',
            'bjetSecMinChiHiggsIndex3',
            'bjetSecMinChiHiggsIndex4',
            'bjetSecMinChiHiggsIndex5',
            'bjetSecMinChiHiggsIndex6',
            'bjetSecMinChiHiggsIndex7',
            'bjetSecMinChiHiggsIndex8',
            'gatja_idx'
        ]

        df_sample = events[events.bjetNumber >= 3]
        df_sample = df_sample[gatja_input_list]
        df_sample = ak.to_dataframe(df_sample)
        # for df_sample in background_processes + signal_processes:
        samples = []
        for i in range(8):
            s = create_graphs(df_sample, i, drop_empty=False)
            samples.append(s)
        
        sample = np.concatenate(samples)
        
        # train_mask = (df_sample['dataset_split'] == 'train').values
        # train_mask_concat = np.tile(train_mask, 8) # Da sample 8x so lang ist
        
        max_sample = np.max(sample, axis=0)
        min_sample = np.min(sample, axis=0)
        
        den = (max_sample - min_sample)
        den_safe = np.where(den == 0, 1.0, den)
        
        sample = (sample - min_sample) / den_safe
        sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)

        sample = sample.astype(np.float32)
        
        gatja_output = gatja_model.predict(sample, batch_size=512)
        
        # Aufspalten und mergen der 8 Jets
        jet_sample = np.split(gatja_output, 8, axis=0)
        gatja_output = np.concatenate(jet_sample, axis=1)

        new_columns = [f'gatja_output_{i}' for i in range(gatja_output.shape[1])]
        df_sample[new_columns] = gatja_output

        import pandas as pd

        event_idx = ak.to_numpy(events["gatja_idx"]).astype(np.int64)
        n_events = len(events)

        output_cols = [c for c in df_sample.columns if c.startswith("gatja_output_")]

        for col in output_cols:

            mapping = pd.Series(
                df_sample[col].values,
                index=df_sample["gatja_idx"].astype(np.int64)
            )

            full_arr = np.full(n_events, -10, dtype=np.float32)

            mask = np.isin(event_idx, mapping.index)

            full_arr[mask] = mapping.loc[event_idx[mask]].values

            events = set_ak_column_f32(events, col, full_arr)

        # events = set_ak_column_f32(events, f"gatja_score1", events.btag_weight) 
    return events

@gatja_scores.init
def gatja_scores_init(self: Producer) -> None:
    add_gatja_scores_variables(self.config_inst)