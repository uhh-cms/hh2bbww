# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from __future__ import annotations

from functools import partial

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper functions
set_ak_bool = partial(set_ak_column, value_type=np.bool_)
set_ak_f32 = partial(set_ak_column, value_type=np.float32)


@producer(
    jet_collection="Jet",
)
def jetId_v12(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer that extracts correct jet ids for Nano v12
    https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV?rev=21
    NOTE: this receipe seems to be looser that the "correct" JetId receipe (at least in NanoV13).
    therefore, this should only be used where really necessary (Nano V12).
    In Nano V13 and forward, we can use the columnflow.production.cms.jet.jet_id Producer, which
    recalculates the jetId from scratch using a centrally provided json file.
    """
    jets = events[self.jet_collection]
    abseta = abs(jets.eta)

    # baseline mask (abseta < 2.7)
    passJetId_Tight = (jets.jetId & 2 == 2)

    passJetId_Tight = ak.where(
        (abseta > 2.7) & (abseta <= 3.0),
        passJetId_Tight & (jets.neHEF < 0.99),
        passJetId_Tight,
    )
    passJetId_Tight = ak.where(
        abseta > 3.0,
        passJetId_Tight & (jets.neEmEF < 0.4),
        passJetId_Tight,
    )

    passJetId_TightLepVeto = ak.where(
        abseta <= 2.7,
        passJetId_Tight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8),
        passJetId_Tight,
    )

    events = set_ak_bool(events, "Jet.TightId", passJetId_Tight)
    events = set_ak_bool(events, "Jet.TightLepVeto", passJetId_TightLepVeto)

    return events


@jetId_v12.init
def jetId_v12_init(self: Producer) -> None:
    config_inst = getattr(self, "config_inst", None)

    if config_inst and config_inst.campaign.x.version != 12:
        raise NotImplementedError("jetId_v12 Producer only recommended for Nano v12")


@jetId_v12.post_init
def jetId_v12_post_init(self: Producer, **kwargs) -> None:
    self.uses = {f"{self.jet_collection}.{{pt,eta,phi,mass,jetId,neHEF,neEmEF,muEF,chEmEF}}"}
    self.produces = {f"{self.jet_collection}.{{TightId,TightLepVeto}}"}


# NOTE: in NanoV12, the FatJet.{chEmEf,muEF,neEmEF,neHEF} columns are not available,
# so this Producer is not useable and we cannot recalculate the jetId in NanoV12.
fatjetId_v12 = jetId_v12.derive("fatjetId_v12", cls_dict={"jet_collection": "FatJet"})


@producer(
    # uses defined in TAF that defines jet_collection
    produces={
        "VBFJet.{pt,eta,phi,mass}",
        "VBFPair.{pt,eta,phi,mass,deta}",
    },
)
def vbf_candidates(
    self: Producer,
    events: ak.Array,
    jet_collection: ak.Array | str = "VBFCandidateJet",
    deta_cut: float | None = None,
    invmass_cut: float | None = 500,
    **kwargs,
) -> ak.Array:
    if isinstance(jet_collection, str):
        print(f"Runninng vbf_candidates with {jet_collection}")
        jet_collection = events[jet_collection]

    vbf_pairs = ak.combinations(jet_collection, 2)
    vbf1, vbf2 = ak.unzip(vbf_pairs)

    # define requirements for vbf pair candidates
    vbf_pairs["deta"] = abs(vbf1.eta - vbf2.eta)
    vbf_pairs["invmass"] = (vbf1 + vbf2).mass

    if invmass_cut:
        vbf_mask = (vbf_pairs.invmass > invmass_cut)
        vbf_pairs = vbf_pairs[vbf_mask]
    if deta_cut:
        vbf_mask = (vbf_pairs.deta > deta_cut)
        vbf_pairs = vbf_pairs[vbf_mask]

    # choose the vbf pair based on maximum delta eta
    chosen_vbf_pair = vbf_pairs[ak.singletons(ak.argmax(vbf_pairs.deta, axis=1))]

    # single collection of chosen vbf jets
    vbf1, vbf2 = [chosen_vbf_pair[i] for i in ["0", "1"]]
    vbf_jets = ak.concatenate([vbf1, vbf2], axis=1)
    vbf_jets = vbf_jets[ak.argsort(vbf_jets.pt, ascending=False)]

    # get rid of Nones
    vbf_jets = vbf_jets[ak.fill_none(vbf_jets.pt > 0, False)]

    # store columns of interest
    events = set_ak_column(events, "VBFJet", vbf_jets)
    vbf_pair = vbf1 + vbf2
    for field in ("pt", "eta", "phi", "mass"):
        events = set_ak_f32(events, f"VBFPair.{field}", getattr(vbf_pair, field))
    for field in ("deta",):
        events = set_ak_f32(events, f"VBFPair.{field}", getattr(chosen_vbf_pair, field))

    return events


@producer(
    uses={
        "{Jet,ForwardJet}.{pt,eta,phi,mass}",
    },
    produces={
        "njet_for_recoil",
    },
)
def njet_for_recoil(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Helper producer for njet used in recoil_corrected_met, assuming only the jet pt is different and jet definition
    See https://cms-higgs-leprare.docs.cern.ch/htt-common/V_recoil for more info and explicit jet definition
    """

    # Count the number of selected jets per event
    njet = ak.num(events.Jet[events.Jet.pt > 30], axis=1) + ak.num(events.ForwardJet[events.ForwardJet.pt > 50], axis=1)

    # Set the njet_for_recoil column
    events = set_ak_column(events, "njet_for_recoil", njet, value_type=np.float32)

    return events


@njet_for_recoil.skip
def njet_for_recoil_skip(self: Producer) -> bool:
    """
    Skip if running on anything except ttbar MC simulation.
    """
    # never skip when there is no dataset
    if not getattr(self, "dataset_inst", None):
        return False

    return self.dataset_inst.is_data or not (
        self.dataset_inst.has_tag("is_dy")
    )
