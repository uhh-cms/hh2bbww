# coding: utf-8

"""
Selection modules for HH(bbWW) that are used for both SL and DL.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Tuple

import law

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.production.util import attach_coffea_behavior

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.categories import category_ids
from columnflow.production.processes import process_ids

from hbw.production.weights import event_weights_to_normalize, large_weights_killer
from hbw.selection.stats import hbw_increment_stats
from hbw.selection.cutflow_features import cutflow_features
from hbw.util import four_vec

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


@selector(
    uses=four_vec("Jet", {"btagDeepFlavB"}),
    exposed=False,
)
def vbf_jet_selection(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:

    # assign local index to all Jets
    events = set_ak_column(events, "Jet.local_index", ak.local_index(events.Jet))

    # default requirements for vbf jets (pt, eta and no H->bb jet)
    # NOTE: we might also want to remove the two H->jj jet candidates
    # TODO: how to get the object mask from the object indices in a more convenient way?
    b_indices = ak.where(
        results.steps.HbbJet,
        ak.fill_none(ak.pad_none(results.objects.Jet.HbbSubJet, 2), -1),
        ak.fill_none(ak.pad_none(results.objects.Jet.Bjet, 2), -1),
    )
    vbf_jets = events.Jet[(events.Jet.local_index != b_indices[:, 0]) & (events.Jet.local_index != b_indices[:, 1])]
    vbf_jets = vbf_jets[(vbf_jets.pt > 30) & (abs(vbf_jets.eta < 4.7))]

    # build all possible pairs of jets fulfilling the `vbf_jet_mask` requirement
    vbf_pairs = ak.combinations(vbf_jets, 2)
    vbf1, vbf2 = ak.unzip(vbf_pairs)

    # define requirements for vbf pair candidates
    vbf_pairs["deta"] = abs(vbf1.eta - vbf2.eta)
    vbf_pairs["invmass"] = (vbf1 + vbf2).mass
    vbf_mask = (vbf_pairs.deta > 3) & (vbf_pairs.invmass > 500)

    # event selection: at least one vbf pair present (TODO: use it for categorization)
    vbf_selection = ak.sum(vbf_mask >= 1, axis=-1) >= 1

    # apply requirements to vbf pairs
    vbf_pairs = vbf_pairs[vbf_mask]

    # choose the vbf pair based on maximum delta eta
    chosen_vbf_pair = vbf_pairs[ak.singletons(ak.argmax(vbf_pairs.deta, axis=1))]

    # get the local indices (pt sorted)
    vbf1, vbf2 = [chosen_vbf_pair[i] for i in ["0", "1"]]
    vbf_jets = ak.concatenate([vbf1, vbf2], axis=1)
    vbf_jets = vbf_jets[ak.argsort(vbf_jets.pt, ascending=False)]

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={"VBFJetPair": vbf_selection},
        objects={"Jet": {
            "VBFJet": vbf_jets.local_index,
        }},
    )


@selector(
    uses=(
        four_vec(
            {"Jet", "Electron", "Muon"},
        ) | {"Jet.jetId"} |
        four_vec(
            "FatJet", {"msoftdrop", "jetId", "subJetIdx1", "subJetIdx2", "tau1", "tau2"},
        )
    ),
    produces=(
        {"cutflow.n_fatjet", "cutflow.n_hbbjet"} |
        four_vec(
            {"FatJet", "HbbJet"},
            {"n_subjets", "n_separated_jets", "max_dr_ak4"},
            skip_defaults=True,
        )
    ),
    exposed=False,
    single_lepton_selector=True,
)
def sl_boosted_jet_selection(
    self: Selector,
    events: ak.Array,
    lepton_results: SelectionResult,
    jet_results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """
    HH -> bbWW(qqlnu) boosted selection
    TODO: separate common parts from SL dependant
    """
    # assign local index to all Jets
    events = set_ak_column(events, "Jet.local_index", ak.local_index(events.Jet))
    events = set_ak_column(events, "FatJet.local_index", ak.local_index(events.FatJet))

    # get leptons and jets with object masks applied
    electron = events.Electron[lepton_results.objects.Electron.Electron]
    muon = events.Muon[lepton_results.objects.Muon.Muon]
    ak4_jets = events.Jet[jet_results.objects.Jet.Jet]

    # get separation info between FatJets and AK4 Jets
    dr_fatjet_ak4 = events.FatJet.metric_table(ak4_jets)
    events = set_ak_column(events, "FatJet.n_subjets", ak.sum(dr_fatjet_ak4 < 0.8, axis=2))
    events = set_ak_column(events, "FatJet.n_separated_jets", ak.sum(dr_fatjet_ak4 > 1.2, axis=2))
    events = set_ak_column(events, "FatJet.max_dr_ak4", ak.max(dr_fatjet_ak4, axis=2))

    # baseline fatjet selection
    fatjet_mask = (
        (events.FatJet.pt > 200) &
        (abs(events.FatJet.eta) < 2.4) &
        (events.FatJet.jetId == 6) &
        (ak.all(events.FatJet.metric_table(electron) > 0.8, axis=2)) &
        (ak.all(events.FatJet.metric_table(muon) > 0.8, axis=2))
    )
    events = set_ak_column(events, "cutflow.n_fatjet", ak.sum(fatjet_mask, axis=1))

    # H->bb fatjet definition based on Aachen analysis
    hbbJet_mask = (
        fatjet_mask &
        (events.FatJet.msoftdrop > 30) &
        (events.FatJet.msoftdrop < 210) &
        (events.FatJet.subJetIdx1 >= 0) &
        (events.FatJet.subJetIdx2 >= 0) &
        (events.FatJet.subJetIdx1 < ak.num(events.Jet)) &
        (events.FatJet.subJetIdx2 < ak.num(events.Jet)) &
        (events.FatJet.tau2 / events.FatJet.tau1 < 0.75)
    )
    # for the SL analysis, we additionally require one AK4 jet that is separated with dR > 1.2
    # (as part of the HbbJet definition)
    if self.single_lepton_selector:
        hbbJet_mask = (
            hbbJet_mask &
            (events.FatJet.n_separated_jets >= 1)
        )

    # create temporary object with fatjet mask applied and get the subjets
    hbbjets = events.FatJet[hbbJet_mask]
    subjet1 = events.Jet[hbbjets.subJetIdx1]
    subjet2 = events.Jet[hbbjets.subJetIdx2]

    # requirements on H->bb subjets (without b-tagging)
    subjets_mask_no_bjet = (
        (abs(subjet1.eta) < 2.4) & (abs(subjet2.eta) < 2.4) &
        (subjet1.pt > 20) & (subjet2.pt > 20) &
        ((subjet1.pt > 30) | (subjet2.pt > 30))
    )
    hbbjets_no_bjet = hbbjets[subjets_mask_no_bjet]
    hbbjet_sel_no_bjet = ak.num(hbbjets_no_bjet, axis=1) >= 1

    # requirements on H->bb subjets (with b-tagging)
    wp_med = self.config_inst.x.btag_working_points.deepjet.medium
    subjets_mask = (
        (abs(subjet1.eta) < 2.4) & (abs(subjet2.eta) < 2.4) &
        (subjet1.pt > 20) & (subjet2.pt > 20) &
        (
            ((subjet1.pt > 30) & (subjet1.btagDeepFlavB > wp_med)) |
            ((subjet2.pt > 30) & (subjet2.btagDeepFlavB > wp_med))
        )
    )

    # apply subjets requirements on hbbjets and pt-sort
    hbbjets = hbbjets[subjets_mask]
    hbbjets = hbbjets[ak.argsort(hbbjets.pt, ascending=False)]
    events = set_ak_column(events, "HbbJet", hbbjets)

    # number of hbbjets fulfilling all criteria
    events = set_ak_column(events, "cutflow.n_hbbjet", ak.num(hbbjets, axis=1))
    hbbjet_sel = events.cutflow.n_hbbjet >= 1

    # define HbbSubJet collection (TODO: pt-sort or b-score sort)
    # TODO: when we get multiple HbbJets, we also define more than 2 hbbSubJets
    hbbSubJet_indices = ak.concatenate([hbbjets.subJetIdx1, hbbjets.subJetIdx2], axis=1)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "HbbJet_no_bjet": hbbjet_sel_no_bjet,
            "HbbJet": hbbjet_sel,
        },
        objects={
            "FatJet": {
                "FatJet": masked_sorted_indices(fatjet_mask, events.FatJet.pt),
                "HbbJet": hbbjets.local_index,
            },
            "Jet": {"HbbSubJet": hbbSubJet_indices},
        },
    )


# boosted selection for the DL channel (only one parameter needs to be changed)
dl_boosted_jet_selection = sl_boosted_jet_selection.derive(
    "dl_boosted_jet_selection",
    cls_dict={"single_lepton_selector": False},
)


@selector(
    exposed=False,
)
def noise_filter(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    mask = ak.Array(np.ones(len(events), dtype=bool))
    for flag in self.noise_filter:
        mask = mask & events.Flag[flag]

    results.steps["noise_filter"] = mask
    return events, results


@noise_filter.init
def noise_filter_init(self: Selector):
    if not getattr(self, "dataset_inst", None):
        return

    # TODO: make campaign dependent
    self.noise_filter = {
        "goodVertices",
        "globalSuperTightHalo2016Filter",
        "HBHENoiseFilter",
        "HBHENoiseIsoFilter",
        "EcalDeadCellTriggerPrimitiveFilter",
        "BadPFMuonFilter",
        "BadPFMuonDzFilter",
        # "hfNoisyHitsFilter",  # optional for UL
        "eeBadScFilter",  # might be data only
        # "ecalBadCalibReducedMINIAODFilter",  # 2017 and 2018 only, only in MiniAOD
    }

    if self.dataset_inst.has_tag("is_hbw") and self.config_inst.has_tag("is_run2"):
        # missing in MiniAOD HH samples
        self.noise_filter.remove("BadPFMuonDzFilter")

    if self.config_inst.has_tag("is_run3"):
        self.noise_filter.add("ecalBadCalibFilter")
    # if self.dataset_inst.is_data:
    #     self.noise_filter.add("eeBadScFilter")

    self.uses = {f"Flag.{flag}" for flag in self.noise_filter}


@selector(
    uses={"PV.npvsGood"},
    exposed=False,
)
def primary_vertex(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """ requires at least one good primary vertex """
    results.steps["good_vertex"] = events.PV.npvsGood >= 1
    return events, results


@selector(
    uses={
        noise_filter, primary_vertex,
        process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,
    },
    produces={
        process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,
    },
    exposed=False,
)
def pre_selection(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """ Methods that are called for both SL and DL before calling the selection modules """

    # temporary fix for optional types from Calibration (e.g. events.Jet.pt --> ?float32)
    # TODO: remove as soon as possible as it might lead to weird bugs when there are none entries in inputs
    events = ak.fill_none(events, EMPTY_FLOAT)

    # mc weight
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
        events = self[large_weights_killer](events, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # apply some general quality criteria on events
    events, results = self[noise_filter](events, results, **kwargs)
    events, results = self[primary_vertex](events, results, **kwargs)

    return events, results


@selector(
    uses={
        category_ids, hbw_increment_stats,
    },
    produces={
        category_ids, hbw_increment_stats,
    },
    exposed=False,
)
def post_selection(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """ Methods that are called for both SL and DL after calling the selection modules """

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # produce event weights
    if self.dataset_inst.is_mc:
        events = self[event_weights_to_normalize](events, results=results, **kwargs)

    # increment stats
    self[hbw_increment_stats](events, results, stats, **kwargs)

    def log_fraction(stats_key: str, msg: str | None = None):
        if not stats.get(stats_key):
            return
        if not msg:
            msg = "Fraction of {stats_key}"
        logger.info(f"{msg}: {(100 * stats[stats_key] / stats['num_events']):.2f}%")

    log_fraction("num_negative_weights", "Fraction of negative weights")
    log_fraction("num_pu_0", "Fraction of events with pu_weight == 0")
    log_fraction("num_pu_100", "Fraction of events with pu_weight >= 100")

    # add cutflow features
    if self.config_inst.x("do_cutflow_features", False):
        events = self[cutflow_features](events, results=results, **kwargs)

    # temporary fix for optional types from Calibration (e.g. events.Jet.pt --> ?float32)
    # TODO: remove as soon as possible as it might lead to weird bugs when there are none entries in inputs
    events = ak.fill_none(events, EMPTY_FLOAT)

    return events, results


@post_selection.init
def post_selection_init(self: Selector) -> None:
    if self.config_inst.x("do_cutflow_features", False):
        self.uses.add(cutflow_features)
        self.produces.add(cutflow_features)

    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return

    self.uses.add(event_weights_to_normalize)
    self.produces.add(event_weights_to_normalize)
