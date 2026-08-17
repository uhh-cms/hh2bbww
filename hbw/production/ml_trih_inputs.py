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
    uses={"Jet.*", "GenPart.*", "GenJet.*"},  # "genTtbarId", "GenJet.*"} --> can only be used for ttbar processes
    produces={
        "GenJet.matchClass", "GenJet.matchDR", "GenJet.matchAmbiguous",
        "Jet.matchClass", "Jet.matchDR", "Jet.matchAmbiguous",
    },
)
def jet_gen_matching_8(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Gen Matching of jets to gen particles (Higgs, Top, Z) and GenJets. These are required for the GATJA training.
    # The matching is done by the minimum ∆R between the jet and the gen particle or genjet
    # The matching is only done for jets that have a genJetIdx >= 0 and < n_genjets.
    # The matching is done for all jets in the event, not just the b-tagged jets
    gp = events.GenPart
    absid_all = abs(gp.pdgId)

    # hard = gp.hasFlags("isHardProcess")

    # isH = hard & (absid_all == 25)
    # isT = hard & (absid_all == 6)
    # isZ = hard & (absid_all == 23)

    is_b = (absid_all == 5)
    isParton = is_b & gp.hasFlags("fromHardProcess") & gp.hasFlags("isLastCopy")
    partons = gp[isParton]

    def gather_flag(flags, idx):
        safe_idx = ak.where(idx < 0, 0, idx)
        return ak.where(idx < 0, False, flags[safe_idx])

    def gather_int(vals, idx, fill):
        safe_idx = ak.where(idx < 0, 0, idx)
        return ak.where(idx < 0, fill, vals[safe_idx])

    anc = partons.genPartIdxMother
    origin = ak.values_astype(ak.full_like(anc, -1), np.int8)
    resolved = ak.zeros_like(anc, dtype=bool)

    max_depth = 50
    for _ in range(max_depth):
        if ak.all(ak.flatten(anc) < 0):
            break

        anc_absid = gather_int(absid_all, anc, -1)
        is_bcopy = (anc_absid == 5)

        at_H = (anc_absid == 25)  # Higgs
        at_T = (anc_absid == 6)  # Top
        at_Z = (anc_absid == 23)  # Z

        decide = (~resolved) & (anc >= 0) & (~is_bcopy)
        origin = ak.where(decide & at_H, 1, origin)
        origin = ak.where(decide & at_T, 2, origin)
        origin = ak.where(decide & at_Z, 3, origin)

        resolved = resolved | decide

        next_anc = gp.genPartIdxMother[ak.where(anc >= 0, anc, 0)]
        anc = ak.where(anc >= 0, next_anc, -1)

    part_origin = origin

    partH = partons[part_origin == 1]
    partT = partons[part_origin == 2]
    partZ = partons[part_origin == 3]

    genjets = events.GenJet

    def min_dr_between(jets, ref_particles):
        pairs = ak.cartesian({"obj": jets, "ref": ref_particles}, nested=True)
        dr = pairs["obj"].delta_r(pairs["ref"])
        return ak.fill_none(ak.min(dr, axis=-1), 999.0)

    dr_stack = ak.concatenate(
        [
            min_dr_between(genjets, partH)[..., np.newaxis],
            min_dr_between(genjets, partT)[..., np.newaxis],
            min_dr_between(genjets, partZ)[..., np.newaxis],
        ],
        axis=-1,
    )

    best_idx = ak.argmin(dr_stack, axis=-1) + 1
    dr_sorted = ak.sort(dr_stack, axis=-1)
    best_dr = dr_sorted[..., 0]
    second_dr = dr_sorted[..., 1]

    matched = best_dr < 0.4
    matchClass = ak.values_astype(ak.where(matched, best_idx, -1), np.int8)
    matchDR = ak.values_astype(ak.where(matched, best_dr, 999.0), np.float32)

    matchAmbiguous = matched & (second_dr < 0.4)

    genjets = ak.with_field(genjets, matchClass, "matchClass")
    genjets = ak.with_field(genjets, matchDR, "matchDR")
    genjets = ak.with_field(genjets, matchAmbiguous, "matchAmbiguous")
    events["GenJet"] = genjets

    jets = events.Jet
    n_genjets = ak.num(events.GenJet, axis=1)

    valid = (jets.genJetIdx >= 0) & (jets.genJetIdx < n_genjets)
    safe_idx = ak.where(valid, jets.genJetIdx, 0)
    matched_genjets = events.GenJet[safe_idx]

    jets = ak.with_field(
        jets,
        ak.values_astype(ak.where(valid, matched_genjets.matchClass, -1),
        np.int8),
        "matchClass",
    )

    jets = ak.with_field(
        jets,
        ak.values_astype(ak.where(valid, matched_genjets.matchDR, 999.0), np.float32),
        "matchDR",
    )

    jets = ak.with_field(
        jets,
        ak.where(valid, matched_genjets.matchAmbiguous, False),
        "matchAmbiguous",
    )

    events["Jet"] = jets

    print("H:", ak.sum(events.Jet.matchClass == 1),
        "T:", ak.sum(events.Jet.matchClass == 2),
        "Z:", ak.sum(events.Jet.matchClass == 3),
        "Others:", ak.sum(events.Jet.matchClass == -1))

    return events


from columnflow.production.normalization import normalization_weights


@producer(
    uses={normalization_weights},
    produces={normalization_weights, "event_weight"},
    mc_only=True,
)
def gatja_event_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[normalization_weights](events, **kwargs)
    events = set_ak_column(events, "event_weight", events.normalization_weight, value_type=np.float32)
    return events


from columnflow.selection.stats import increment_stats


@producer(
    uses={increment_stats, "process_id", "fold_indices", "event_weight", "jetNumber"},
)
def gatja_prepml(
    self: Producer,
    events: ak.Array,
    task: law.Task,
    stats: dict = {},
    fold_indices: ak.Array | None = None,
    ml_model_inst=None,
    **kwargs,
) -> ak.Array:

    # gatja_prepml is an adapted copy of prepml
    # Two adaotations: 1) jetNumber >= 3 instead of the signal-region
    # categorizer because GATJA graph needs a node jet plus its two chi2 neighbours
    # 2) Weights: reads the pre-computed event_weight column (normalization weight) instead of
    # calling default_hist_producer (Hist_producer caused an error).
    # These weights do not enter the GATJA loss since the traning uses btag_weight as sample_weight

    if task.task_family == "cf.PrepareMLEvents":
        mask = events.jetNumber >= 3  # Select events with at least 3 jets for gatja training
        events = events[mask]

    weight_map = {
        "num_events": Ellipsis,
    }
    if task.dataset_inst.is_mc:
        weight = events.event_weight
        stats["sum_weights"] += float(ak.sum(weight, axis=0))
        weight_map["sum_weights"] = weight
        weight_map["sum_pos_weights"] = (weight, weight > 0)
        weight_map["sum_abs_weights"] = np.abs(weight)
        weight_map["num_events_pos_weights"] = weight > 0

    group_map = {
        "process": {
            "values": events.process_id,
            "mask_fn": (lambda v: events.process_id == v),
        },
        "fold": {
            "values": events.fold_indices,
            "mask_fn": (lambda v: events.fold_indices == v),
            "combinations_only": True,
        },
    }

    self[increment_stats](
        events,
        None,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        group_combinations=[("process", "fold")],
        **kwargs,
    )

    for key in list(weight_map.keys()):
        stats.pop(key, None)

    return events


# Input features which are required for the GATJA traning.
# The producer includes all features which are also used in the first version of the tutorial of
# Oszguar and Gamze (not all are used for the training at the moment)
# Furhtermore some additonal features are added
@producer(
    uses={
        prepare_objects,
        btag_wp_weights,
        common_ml_inputs,
        prepare_hhh_bjets,
        event_weights,
        "Jet.*",
        "Jet.matchClass",
    },
    produces={
        "event_id", "jetPT1", "jetPT2", "jetPT3", "jetPT4", "jetPT5", "jetPT6", "jetPT7", "jetPT8",
        "bjetPT1", "bjetPT2", "bjetPT3", "bjetPT4", "bjetPT5", "bjetPT6", "bjetPT7", "bjetPT8",
        "jetEta1", "jetEta2", "jetEta3", "jetEta4", "jetEta5", "jetEta6", "jetEta7", "jetEta8",
        "bjetEta1", "bjetEta2", "bjetEta3", "bjetEta4", "bjetEta5", "bjetEta6", "bjetEta7", "bjetEta8",
        "leptonPT1", "leptonEta1", "leptonPT2", "leptonEta2", "leptonPhi1", "leptonPhi2",
        "bjetAverageMass", "jetAverageMass", "jetAverageMassSqr",
        "bjetAverageMassSqr", "jetHT", "bjetHT", "lightjetHT", "jetNumber", "bjetNumber",
        "jetPhi1", "jetPhi2", "jetPhi3", "jetPhi4", "jetPhi5", "jetPhi6", "jetPhi7", "jetPhi8",
        "bjetPhi1", "bjetPhi2", "bjetPhi3", "bjetPhi4", "bjetPhi5", "bjetPhi6", "bjetPhi7", "bjetPhi8",
        "averageDeltaEtabb", "minDeltaRjj", "minDeltaRbb",
        "maxDeltaEtabb", "maxDeltaEtajj", "maxDeltaEtabj",
        "minDeltaRbj", "averageDeltaEtabj", "averageDeltaRbj", "minDeltaRMassjj",
        "minDeltaRMassbb", "minDeltaRMassbj",
        "minDeltaRpTjj", "minDeltaRpTbb", "minDeltaRpTbj", "maxPTmassjjj", "maxPTmassjbb", "met", "metPhi",
        "jetMinChiHiggsIndex1", "jetSecMinChiHiggsIndex1", "jetMinChiHiggsIndex2", "jetSecMinChiHiggsIndex2",
        "jetMinChiHiggsIndex3", "jetSecMinChiHiggsIndex3", "jetMinChiHiggsIndex4", "jetSecMinChiHiggsIndex4",
        "jetMinChiHiggsIndex5", "jetSecMinChiHiggsIndex5", "jetMinChiHiggsIndex6", "jetSecMinChiHiggsIndex6",
        "jetMinChiHiggsIndex7", "jetSecMinChiHiggsIndex7", "jetMinChiHiggsIndex8", "jetSecMinChiHiggsIndex8",
        "bjetMinChiHiggsIndex1", "bjetSecMinChiHiggsIndex1", "bjetMinChiHiggsIndex2", "bjetSecMinChiHiggsIndex2",
        "bjetMinChiHiggsIndex3", "bjetSecMinChiHiggsIndex3", "bjetMinChiHiggsIndex4", "bjetSecMinChiHiggsIndex4",
        "bjetMinChiHiggsIndex5", "bjetSecMinChiHiggsIndex5", "bjetMinChiHiggsIndex6", "bjetSecMinChiHiggsIndex6",
        "bjetMinChiHiggsIndex7", "bjetSecMinChiHiggsIndex7", "bjetMinChiHiggsIndex8", "bjetSecMinChiHiggsIndex8",
        "jetBTagDisc1", "jetBTagDisc2", "jetBTagDisc3", "jetBTagDisc4", "jetBTagDisc5",
        "jetBTagDisc6", "jetBTagDisc7", "jetBTagDisc8",
        "jetBTagDisDisc1", "jetBTagDisDisc2", "jetBTagDisDisc3", "jetBTagDisDisc4",
        "jetBTagDisDisc5", "jetBTagDisDisc6", "jetBTagDisDisc7", "jetBTagDisDisc8",
        "bjetBTagDisc1", "bjetBTagDisc2", "bjetBTagDisc3", "bjetBTagDisc4",
        "bjetBTagDisc5", "bjetBTagDisc6", "bjetBTagDisc7", "bjetBTagDisc8",
        "bjetBTagDisDisc1", "bjetBTagDisDisc2", "bjetBTagDisDisc3", "bjetBTagDisDisc4",
        "bjetBTagDisDisc5", "bjetBTagDisDisc6", "bjetBTagDisDisc7", "bjetBTagDisDisc8",
        "btag_weight", "weights",
        "jetTopMatched1", "jetTopMatched2", "jetTopMatched3", "jetTopMatched4",
        "jetTopMatched5", "jetTopMatched6", "jetTopMatched7", "jetTopMatched8",
        "bjetTopMatched1", "bjetTopMatched2", "bjetTopMatched3", "bjetTopMatched4",
        "bjetTopMatched5", "bjetTopMatched6", "bjetTopMatched7", "bjetTopMatched8",
        "jetHiggsMatched1", "jetHiggsMatched2", "jetHiggsMatched3", "jetHiggsMatched4",
        "jetHiggsMatched5", "jetHiggsMatched6", "jetHiggsMatched7", "jetHiggsMatched8",
        "bjetHiggsMatched1", "bjetHiggsMatched2", "bjetHiggsMatched3", "bjetHiggsMatched4",
        "bjetHiggsMatched5", "bjetHiggsMatched6", "bjetHiggsMatched7", "bjetHiggsMatched8",
        "jetZMatched1", "jetZMatched2", "jetZMatched3", "jetZMatched4",
        "jetZMatched5", "jetZMatched6", "jetZMatched7", "jetZMatched8",
        "bjetZMatched1", "bjetZMatched2", "bjetZMatched3", "bjetZMatched4",
        "bjetZMatched5", "bjetZMatched6", "bjetZMatched7", "bjetZMatched8",
        "jetsMass12", "jetsMass13", "jetsMass14", "jetsMass15", "jetsMass16", "jetsMass17", "jetsMass18",
        "jetsMass23", "jetsMass24", "jetsMass25", "jetsMass26", "jetsMass27", "jetsMass28",
        "jetsMass34", "jetsMass35", "jetsMass36", "jetsMass37", "jetsMass38",
        "jetsMass45", "jetsMass46", "jetsMass47", "jetsMass48", "jetsMass56", "jetsMass57", "jetsMass58",
        "jetsMass67", "jetsMass68", "jetsMass78",
        "bjetsMass12", "bjetsMass13", "bjetsMass14", "bjetsMass15", "bjetsMass16", "bjetsMass17", "bjetsMass18",
        "bjetsMass23", "bjetsMass24", "bjetsMass25", "bjetsMass26", "bjetsMass27", "bjetsMass28",
        "bjetsMass34", "bjetsMass35", "bjetsMass36", "bjetsMass37", "bjetsMass38",
        "bjetsMass45", "bjetsMass46", "bjetsMass47", "bjetsMass48", "bjetsMass56", "bjetsMass57", "bjetsMass58",
        "bjetsMass67", "bjetsMass68", "bjetsMass78",
        "jetMass1", "jetMass2", "jetMass3", "jetMass4", "jetMass5", "jetMass6", "jetMass7", "jetMass8",
        "bjetMass1", "bjetMass2", "bjetMass3", "bjetMass4", "bjetMass5", "bjetMass6", "bjetMass7", "bjetMass8",
    },
)
def gatja_inputs_jet_based_plus_b_jet_inputs_corrected_Higgs_Index_discrete_b(
    self: Producer, events: ak.Array, **kwargs,
) -> ak.Array:
    mass_higgs = 125.0
    sigma = 10.0  # PLATZHALTER FÜR DIE MASSENAUFLÖSUNG -> MUSS NOCH BESTIMMT WERDEN
    # produce common input features
    events = self[common_ml_inputs](events, **kwargs)
    events = self[prepare_hhh_bjets](events, **kwargs)
    events = self[event_weights](events, **kwargs)
    # add behavior and define new collections (e.g. Lepton)
    events = self[prepare_objects](events, **kwargs)
    jet_mask = (events.Jet["pt"] < 10_000) & (abs(events.Jet["eta"]) < 2.5)
    events = self[btag_wp_weights](events, jet_mask=jet_mask, **kwargs)
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
    # d_phi_bjet_pairs = (b1.delta_phi(b2))
    deltaR_bb = b1.delta_r(b2)

    b_mix, j_mix = ak.unzip(ak.cartesian([bjets, events.Jet]))
    deltaR_bj = b_mix.delta_r(j_mix)

    j1_3, j2_3, j3_3 = ak.unzip(ak.combinations(events.Jet, 3))
    j_cart, bb_cart = ak.unzip(ak.cartesian([events.Jet, ak.combinations(bjets, 2)]))
    b1_cart, b2_cart = ak.unzip(bb_cart)

    events = set_ak_column_f32(events, "event_id", events.event)

    for i in range(8):
        events = set_ak_column_f32(events, f"jetPT{i+1}", ak.fill_none(padded_jets["pt"][:, i], -6.0))
        events = set_ak_column_f32(events, f"bjetPT{i+1}", ak.fill_none(padded_bjets["pt"][:, i], -6.0))

    for i in range(8):
        events = set_ak_column_f32(events, f"jetEta{i+1}", ak.fill_none(padded_jets["eta"][:, i], -6.0))
        events = set_ak_column_f32(events, f"bjetEta{i+1}", ak.fill_none(padded_bjets["eta"][:, i], -6.0))

    for i in range(2):
        events = set_ak_column_f32(events, f"leptonPT{i+1}", ak.fill_none(padded_lepton["pt"][:, i], -6.0))
        events = set_ak_column_f32(events, f"leptonEta{i+1}", ak.fill_none(padded_lepton["eta"][:, i], -6.0))

    for i in range(2):
        events = set_ak_column_f32(events, f"leptonPhi{i+1}", ak.fill_none(padded_lepton["phi"][:, i], -6.0))
    bjetaveragemass = ak.mean(bjets.mass, axis=1)
    events = set_ak_column_f32(events, "bjetAverageMass", ak.fill_none(ak.nan_to_none(bjetaveragemass), -6.0))
    jetaveragemass = ak.mean(jets.mass, axis=1)
    events = set_ak_column_f32(
        events, "jetAverageMass", ak.where(
            ak.num(events.Jet) > 0, ak.sum(events.Jet.mass, axis=1) / ak.num(events.Jet), 0,
        ),
    )
    events = set_ak_column_f32(
        events, "bjetAverageMassSqr", ak.fill_none(
            ak.nan_to_none(bjetaveragemass * bjetaveragemass * ak.num(bjets)), -6.0,
        ),
    )
    events = set_ak_column_f32(
        events, "jetAverageMassSqr", ak.fill_none(ak.nan_to_none(jetaveragemass * jetaveragemass * ak.num(jets)), -6.0),
    )
    events = set_ak_column_f32(events, "jetHT", ak.sum(jets.pt, axis=1))
    events = set_ak_column_f32(events, "bjetHT", ak.sum(bjets.pt, axis=1))
    events = set_ak_column_f32(events, "lightjetHT", ak.sum(events.Lightjet.pt, axis=1))
    events = ak.with_field(events, ak.num(events.Jet), "jetNumber")
    n_bjets = ak.num(bjets, axis=1)
    events = ak.with_field(events, n_bjets, "bjetNumber")
    for i in range(8):
        events = set_ak_column_f32(events, f"jetPhi{i+1}", ak.fill_none(padded_jets["phi"][:, i], -6.0))
        events = set_ak_column_f32(events, f"bjetPhi{i+1}", ak.fill_none(padded_bjets["phi"][:, i], -6.0))
    events = set_ak_column_f32(
        events, "averageDeltaEtabb", ak.fill_none(ak.nan_to_none(ak.mean(abs(d_eta_bjet_pairs), axis=1)), -6.0),
    )
    events = set_ak_column_f32(events, "minDeltaRjj", ak.fill_none(ak.nan_to_none(ak.min(deltaR_jj, axis=1)), -6.0))
    events = set_ak_column_f32(events, "minDeltaRbb", ak.fill_none(ak.nan_to_none(ak.min(deltaR_bb, axis=1)), -6.0))
    events = set_ak_column_f32(
        events, "maxDeltaEtabb", ak.fill_none(ak.nan_to_none(ak.max(abs(d_eta_bjet_pairs), axis=1)), -6.0),
    )
    events = set_ak_column_f32(
        events, "maxDeltaEtajj", ak.fill_none(ak.nan_to_none(ak.max(abs(d_eta_jet_pairs), axis=1)), -6.0),
    )
    events = set_ak_column_f32(
        events, "maxDeltaEtabj", ak.fill_none(ak.nan_to_none(ak.max(abs(b_mix.eta - j_mix.eta), axis=1)), -6.0),
    )
    events = set_ak_column_f32(events, "minDeltaRbj", ak.fill_none(ak.nan_to_none(ak.min(deltaR_bj, axis=1)), -6.0))
    events = set_ak_column_f32(
        events, "averageDeltaEtabj", ak.fill_none(ak.nan_to_none(ak.mean(abs(b_mix.eta - j_mix.eta), axis=1)), -6.0),
    )
    events = set_ak_column_f32(
        events, "averageDeltaRbj", ak.fill_none(ak.nan_to_none(ak.mean(deltaR_bj, axis=1)), -6.0),
    )

    mask_min_dR_jj = deltaR_jj == ak.min(deltaR_jj, axis=1, keepdims=True)
    # Invariant mass of jet pair with smallest ΔR
    events = set_ak_column_f32(events, "minDeltaRMassjj", ak.fill_none(ak.firsts((j1 + j2).mass[mask_min_dR_jj]), -6.0))
    mask_min_dR_bb = deltaR_bb == ak.min(deltaR_bb, axis=1, keepdims=True)
    mask_min_dR_bj = deltaR_bj == ak.min(deltaR_bj, axis=1, keepdims=True)
    # Invariant mass of b-jet pair with smallest ΔR
    events = set_ak_column_f32(
        events, "minDeltaRMassbb", ak.fill_none(ak.firsts((b1 + b2).mass[mask_min_dR_bb]), -6.0),
    )
    # Invariant mass of jet+bjet-pair pair with
    events = set_ak_column_f32(
        events, "minDeltaRMassbj", ak.fill_none(ak.firsts((b_mix + j_mix).mass[mask_min_dR_bj]), -6.0),
    )
    # Combined transverse momentum of jet pair with smallest ΔR
    events = set_ak_column_f32(events, "minDeltaRpTjj", ak.fill_none(ak.firsts((j1.pt + j2.pt)[mask_min_dR_jj]), -6.0))
    # Combined transverse momentum of b-jet pair with smallest ΔR
    events = set_ak_column_f32(events, "minDeltaRpTbb", ak.fill_none(ak.firsts((b1.pt + b2.pt)[mask_min_dR_bb]), -6.0))
    # Combined transverse momentum of jet+bjet-pair pair with smallest ΔR
    events = set_ak_column_f32(
        events, "minDeltaRpTbj", ak.fill_none(ak.firsts((b_mix.pt + j_mix.pt)[mask_min_dR_bj]), -6.0),
    )
    pt_jjj = j1_3.pt + j2_3.pt + j3_3.pt
    mask_max_pT_jjj = (pt_jjj) == ak.max(pt_jjj, axis=1, keepdims=True)
    # Mass of 3-jet system with highest total pT (boosted object candidate)
    events = set_ak_column_f32(
        events, "maxPTmassjjj", ak.fill_none(ak.firsts((j1_3 + j2_3 + j3_3).mass[mask_max_pT_jjj]), -6.0),
    )
    pt_jbb = j_cart.pt + b1_cart.pt + b2_cart.pt
    mask_max_pT_jbb = (pt_jbb) == ak.max(pt_jbb, axis=1, keepdims=True)
    # Mass of system (1 jet + 2 b-jets) with highest total pT
    events = set_ak_column_f32(
        events, "maxPTmassjbb", ak.fill_none(ak.firsts((j_cart + b1_cart + b2_cart).mass[mask_max_pT_jbb]), -6.0),
    )

    events = set_ak_column_f32(events, "met", events.mli_met_pt)
    events = set_ak_column_f32(events, "metPhi", events.mli_met_phi)

    events = set_ak_column_f32(events, "btag_weight", events.btag_weight)
    events = set_ak_column_f32(events, "weights", events.stitched_normalization_weight)

    def chi2_higgs_indices(events, objs, padded_objs, prefix, n_pad=8):
        n_objs = ak.num(objs, axis=1)

        objs_i = padded_objs[:, :, np.newaxis]
        objs_j = padded_objs[:, np.newaxis, :]

        dipair_mass = ak.without_parameters((objs_i + objs_j).mass)

        idx = ak.local_index(padded_objs, axis=1)
        idx_i = idx[:, :, np.newaxis]
        idx_j = idx[:, np.newaxis, :]

        valid = ~ak.is_none(padded_objs.pt, axis=1)
        valid_i = valid[:, :, np.newaxis]
        valid_j = valid[:, np.newaxis, :]

        pair_valid = valid_i & valid_j & (idx_i != idx_j)

        chi2_matrix = ((dipair_mass - mass_higgs) / sigma) ** 2
        chi2_matrix = ak.where(pair_valid, ak.fill_none(chi2_matrix, np.inf), np.inf)

        min_idx = ak.argmin(chi2_matrix, axis=2)
        min_idx_filled = ak.fill_none(min_idx, -6.0)

        mask_sec = idx_j != min_idx_filled[:, :, np.newaxis]
        chi2_matrix_sec = ak.where(mask_sec, chi2_matrix, np.inf)
        sec_min_idx_filled = ak.fill_none(ak.argmin(chi2_matrix_sec, axis=2), -6.0)

        min_chi = ak.min(chi2_matrix, axis=2)
        min_idx_final = ak.where(min_chi != np.inf, min_idx_filled, -6)

        sec_chi = ak.min(chi2_matrix_sec, axis=2)
        sec_min_idx_final = ak.where(sec_chi != np.inf, sec_min_idx_filled, -6.0)

        for i in range(n_pad):
            obj_exists = n_objs > i
            mi_out = ak.where(obj_exists, min_idx_final[:, i], -6.0)
            si_out = ak.where(obj_exists, sec_min_idx_final[:, i], -6.0)
            events = set_ak_column_f32(events, f"{prefix}MinChiHiggsIndex{i+1}", mi_out)
            events = set_ak_column_f32(events, f"{prefix}SecMinChiHiggsIndex{i+1}", si_out)
        return events

    events = chi2_higgs_indices(events, jets, padded_jets, "jet")
    events = chi2_higgs_indices(events, bjets, padded_bjets, "bjet")

    for i in range(8):
        mass_filled = ak.fill_none(padded_jets.mass[:, i], -6.0)
        # mass_final = ak.where(min_3_bjets, mass_filled, -999.0)
        events = set_ak_column_f32(events, f"jetMass{i+1}", mass_filled)

    for i in range(8):
        mass_filled = ak.fill_none(padded_bjets.mass[:, i], -6.0)
        # mass_final = ak.where(min_3_bjets, mass_filled, -999.0)
        events = set_ak_column_f32(events, f"bjetMass{i+1}", mass_filled)

    for i in range(8):
        events = set_ak_column_f32(events, f"jetBTagDisc{i+1}", ak.fill_none(padded_jets.b_score[:, i], -6.0))
        events = set_ak_column_f32(events, f"bjetBTagDisc{i+1}", ak.fill_none(padded_bjets.b_score[:, i], -6.0))
        events = set_ak_column_f32(
            events, f"jetBTagDisDisc{i+1}", ak.fill_none(padded_jets.discrete_b_score[:, i], -6.0),
        )
        events = set_ak_column_f32(
            events, f"bjetBTagDisDisc{i+1}", ak.fill_none(padded_bjets.discrete_b_score[:, i], -6.0),
        )

    match_class = ak.fill_none(ak.pad_none(jets.matchClass, 8, clip=True), -1)

    top = ak.values_astype(match_class == 2, "float32")
    z = ak.values_astype(match_class == 3, "float32")
    higgs = ak.values_astype(match_class == 1, "float32")

    for i in range(8):
        events = set_ak_column_f32(events, f"jetTopMatched{i+1}", top[:, i])
        events = set_ak_column_f32(events, f"jetZMatched{i+1}", z[:, i])
        events = set_ak_column_f32(events, f"jetHiggsMatched{i+1}", higgs[:, i])

    match_class_bjet = ak.fill_none(ak.pad_none(bjets.matchClass, 8, clip=True), -1)

    top_bjet = ak.values_astype(match_class_bjet == 2, "float32")
    z_bjet = ak.values_astype(match_class_bjet == 3, "float32")
    higgs_bjet = ak.values_astype(match_class_bjet == 1, "float32")

    for i in range(8):
        events = set_ak_column_f32(events, f"bjetTopMatched{i+1}", top_bjet[:, i])
        events = set_ak_column_f32(events, f"bjetZMatched{i+1}", z_bjet[:, i])
        events = set_ak_column_f32(events, f"bjetHiggsMatched{i+1}", higgs_bjet[:, i])

    pair_mass_matrix = ak.without_parameters(
        (padded_jets[:, :, np.newaxis] + padded_jets[:, np.newaxis, :]).mass,
    )
    for i in range(8):
        for j in range(i + 1, 8):
            events = set_ak_column_f32(
                events, f"jetsMass{i+1}{j+1}",
                ak.fill_none(pair_mass_matrix[:, i, j], -6.0),
            )

    pair_bmass_matrix = ak.without_parameters(
        (padded_bjets[:, :, np.newaxis] + padded_bjets[:, np.newaxis, :]).mass,
    )
    for i in range(8):
        for j in range(i + 1, 8):
            events = set_ak_column_f32(
                events, f"bjetsMass{i+1}{j+1}",
                ak.fill_none(pair_bmass_matrix[:, i, j], -6.0),
            )

    return events


# This producer is used to compute the GATJA scores for a network which is trained
# in a Jupyter Notebook instead of columnflow
# When using this producer, one needs to take into account that there could be an overlapp between training/testing set
@producer(
    uses={
        IF_GATJA(gatja_inputs_jet_based_plus_b_jet_inputs_corrected_Higgs_Index_discrete_b),
        hhh_dl_ml_inputs,
    },
    produces={
        IF_GATJA(gatja_inputs_jet_based_plus_b_jet_inputs_corrected_Higgs_Index_discrete_b),
        hhh_dl_ml_inputs,
        IF_GATJA(*{f"gatja_output_{i}" for i in range(24)}),
    },
    # produced columns set in the init function
    # version=law.config.get_expanded(
    #     "analysis", "gatja_scores_version",
    #     "New_labels_evaluation_von_4_ueberarbeitung_der_Inputs_includieren_von_ttbb_Training_1",
    # ),
    version=law.config.get_expanded(
        "analysis", "gatja_scores_version", "Training_31_Training_24_additional_Top_other_Balance",
    ),
    sandbox=dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_plotting.sh"),
)
def gatja_scores_jet_based_full_gatja_corrected_Higgs_Index(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # import tensorflow as tf

    import pickle
    import joblib
    import pandas as pd
    import tensorflow as tf

    # N_JETS = 8
    # NO_SCORE = -10.0
    # MAX_JET_INDEX = N_JETS - 1

    # path = "/data/dust/user/weidnerb/Code_setup_after_CMS_week/New_labels/evaluation_von_4_ueberarbeitung_der_Inputs_includieren_von_ttbb/Training_1" # GATJA 2
    path = "/data/dust/user/weidnerb/New_era/Training_31_Training_24_additional_Top_other_Balance"  # GATJA 3
    model_file = f"{path}/save_gatja_main_best_v3_jet_based.keras"
    scaler_files = {name: f"{path}/{name}.joblib" for name in ("robust_scaler", "quantile_scaler", "minmax_scaler")}

    rest_cols = [
        "jetHT",  # GATJA 3
        # "jetHT", "bjetHT", "lightjetHT", # GATJA 3
        "jetNumber", "jetAverageMass",
        "leptonPT1", "leptonEta1", "leptonPhi1",
        "leptonPT2", "leptonEta2", "leptonPhi2",
        "met",
    ]

    def real_jet_mask(df: pd.DataFrame, slot: int) -> np.ndarray:
        return df[f"jetPT{slot}"].to_numpy() > 0.0

    def load_gatja_model():
        return tf.keras.models.load_model(model_file, compile=False)

    def load_scalers():
        scalers = []
        for name in ("robust_scaler", "quantile_scaler", "minmax_scaler"):
            path = scaler_files[name]
            if path.endswith(".joblib"):
                scalers.append(joblib.load(path))
            else:
                with open(path, "rb") as f:
                    scalers.append(pickle.load(f))
        return tuple(scalers)

    def _safe_lookup(frame: pd.DataFrame, row_labels, column_names) -> np.ndarray:
        if len(row_labels) == 0:
            return np.array([], dtype=float)
        subset = frame.loc[row_labels]
        column_index = subset.columns.get_indexer(column_names)
        if np.any(column_index < 0):
            missing = [column_names[i] for i, v in enumerate(column_index) if v < 0]
            raise KeyError(f"Missing neighbour columns: {missing}")
        return subset.to_numpy()[np.arange(len(row_labels)), column_index]

    # N_JETS = 8
    # JET_CLASSES = ("higgs", "top", "other")

    def compute_padding_mask(working, index):
        slot = index + 1
        # by_count = slot > working["jetNumber"].to_numpy()
        by_sentinel = ~real_jet_mask(working, slot)
        return by_sentinel

    def _create_graphs_core(
        df: pd.DataFrame, index: int, drop_empty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        working = df.copy()

        if drop_empty:
            working = working.loc[real_jet_mask(working, index + 1)].copy()

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
        btag_weight = working["btag_weight"].to_numpy()

        node_part = working[node_cols].to_numpy()
        rest_part = working[rest_cols].to_numpy()

        low_partner = (working[low_index_column] + 1).astype(int).astype(str)
        second_partner = (working[second_index_column] + 1).astype(int).astype(str)

        # print("index is : ", index_main)
        # print("the dataframe : ", np.sum(("jetPT" + low_partner) == "jetPT0"))

        label_higgs = working[f"jetHiggsMatched{index + 1}"].to_numpy(dtype=bool)  # .drop(empty_index)
        label_top = working[f"jetTopMatched{index + 1}"].to_numpy(dtype=bool)  # .drop(empty_index)
        # label_sample = working["sample"].drop(empty_index)
        label_others = (~np.logical_or(label_top, label_higgs)).astype(int)

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

        graph_data = np.hstack((
            btag_weight[:, None], node_part, rest_part, np.array(neighbour).T, np.array(neighbour2).T,
        ))
        # graph_data = np.hstack((main.to_numpy(), np.array(neighbour).T, np.array(neighbour2).T))
        labels = np.vstack(
            (
                label_higgs.astype(int),
                label_top.astype(int),
                label_others.astype(int),
                # label_sample.to_numpy(),
            ),
        )
        padding_mask = compute_padding_mask(working, index)
        return graph_data, labels, padding_mask

    def create_graphs(
        df: pd.DataFrame, index: int, drop_empty: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _create_graphs_core(df, index=index, drop_empty=drop_empty)

    def predict_all_jets(df_all, model, scalers, evt_pos_filtered, expected_features, n_jets=8):
        robust_scaler, quantile_scaler, minmax_scaler = scalers
        jet_pred_dfs = []

        for jet_idx in range(n_jets):
            mask_np = real_jet_mask(df_all, jet_idx + 1)
            if not np.any(mask_np):
                continue

            df_kept = df_all.loc[mask_np].reset_index(drop=True)
            pos_kept = ak.to_numpy(evt_pos_filtered[mask_np]).astype(np.int64)

            sample_block, _, padding_mask = create_graphs(df_kept, jet_idx, drop_empty=False)
            x_raw = sample_block[:, 1:]  # btag_weight abtrennen
            x_scaled = minmax_scaler.transform(
                quantile_scaler.transform(robust_scaler.transform(x_raw)),
            )
            y_pred = model.predict(x_scaled, batch_size=4096, verbose=0)

            jet_pred_dfs.append(pd.DataFrame({
                "evt_pos": pos_kept,
                "jet_idx": jet_idx,
                "prob_higgs": y_pred[:, 0],
                "prob_top": y_pred[:, 1],
                "prob_other": y_pred[:, 2],
            }))

        return pd.concat(jet_pred_dfs, ignore_index=True)

    def attach_outputs(events_in, pred_df, n_jets=8):
        n_events = len(events_in)
        out_arrays = {i: np.full(n_events, -10.0, dtype=np.float32) for i in range(n_jets * 3)}
        jets_allowed = np.asarray(ak.to_numpy(events_in.jetNumber) >= 3, dtype=bool)

        for row in pred_df.itertuples(index=False):
            ievent = int(row.evt_pos)
            if ievent < 0 or ievent >= n_events or not jets_allowed[ievent]:
                continue
            j = int(row.jet_idx)
            out_arrays[j * 3 + 0][ievent] = float(row.prob_higgs)
            out_arrays[j * 3 + 1][ievent] = float(row.prob_top)
            out_arrays[j * 3 + 2][ievent] = float(row.prob_other)

        events_out = events_in
        for out_i, arr in out_arrays.items():
            events_out = set_ak_column_f32(events_out, f"gatja_output_{out_i}", arr)
        return events_out

    events = self[hhh_dl_ml_inputs](events, **kwargs)
    if not self.has_dep(gatja_inputs_jet_based_plus_b_jet_inputs_corrected_Higgs_Index_discrete_b):
        output_cols = [f"gatja_output_{i}" for i in range(23)]
        for col in output_cols:
            events = set_ak_column_f32(events, col, ak.full_like(events.mli_n_jet, -10))
        return events

    events = self[gatja_inputs_jet_based_plus_b_jet_inputs_corrected_Higgs_Index_discrete_b](events, **kwargs)

    evt_pos = ak.local_index(events.jetNumber)
    keep_events = events.jetNumber >= 3
    events_filtered = events[keep_events]
    evt_pos_filtered = evt_pos[keep_events]

    gatja_input_list = ["btag_weight", *rest_cols]
    for i in range(1, 9):
        gatja_input_list += [
            f"jetPT{i}", f"jetEta{i}", f"jetPhi{i}", f"jetBTagDisc{i}",
            f"jetMinChiHiggsIndex{i}", f"jetSecMinChiHiggsIndex{i}", f"jetHiggsMatched{i}", f"jetTopMatched{i}",
        ]

    df_all = pd.DataFrame({
        col: ak.to_numpy(events_filtered[col])
        for col in dict.fromkeys(gatja_input_list)
    })

    model = load_gatja_model()
    scalers = load_scalers()
    expected_features = model.input_shape[-1]

    pred_df = predict_all_jets(df_all, model, scalers, evt_pos_filtered, expected_features)
    events = attach_outputs(events, pred_df)

    return events


@gatja_scores_jet_based_full_gatja_corrected_Higgs_Index.init
def gatja_scores_jet_based_full_gatja_corrected_Higgs_Index_init(self: Producer) -> None:
    add_gatja_scores_variables(self.config_inst)
