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
set_ak_column_f32 = partial(set_ak_column, value_type=np.float32)
set_ak_bool = partial(set_ak_column, value_type=np.bool_)
set_ak_f32 = partial(set_ak_column, value_type=np.float32)

ZERO_PADDING_VALUE = -10


@producer(
    uses={"Jet.{pt,eta,phi,mass,jetId,neHEF,neEmEF,muEF,chEmEF}"},
    produces={"Jet.TightId", "Jet.TightLepVeto"},
)
def jetId(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer that extracts correct jet ids for Nano v12
    https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV?rev=21
    """
    abseta = abs(events.Jet.eta)

    # baseline mask (abseta < 2.7)
    passJetId_Tight = (events.Jet.jetId & 2 == 2)

    passJetId_Tight = ak.where(
        (abseta > 2.7) & (abseta <= 3.0),
        passJetId_Tight & (events.Jet.neHEF < 0.99),
        passJetId_Tight,
    )
    passJetId_Tight = ak.where(
        abseta > 3.0,
        passJetId_Tight & (events.Jet.neEmEF < 0.4),
        passJetId_Tight,
    )

    passJetId_TightLepVeto = ak.where(
        abseta <= 2.7,
        passJetId_Tight & (events.Jet.muEF < 0.8) & (events.Jet.chEmEF < 0.8),
        passJetId_Tight,
    )

    events = set_ak_bool(events, "Jet.TightId", passJetId_Tight)
    events = set_ak_bool(events, "Jet.TightLepVeto", passJetId_TightLepVeto)

    return events


@jetId.init
def jetId_init(self: Producer) -> None:
    config_inst = getattr(self, "config_inst", None)

    if config_inst and config_inst.campaign.x.version != 12:
        raise NotImplementedError("jetId Producer only recommended for Nano v12")


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
