# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from __future__ import annotations

from functools import partial

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.selection import SelectionResult

from hbw.config.variables import add_vbf_variables
from hbw.production.prepare_objects import prepare_objects

ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper functions
set_ak_column_f32 = partial(set_ak_column, value_type=np.float32)
set_ak_bool = partial(set_ak_column, value_type=np.bool_)

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
    print("start")
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
    # produces in init
    produces=set(
        f"{obj}_{var}"
        for obj in ["vbf1", "vbf2", "customjet0", "customjet1"]
        for var in ["pt", "eta", "phi", "mass"]
    ),
    # region=str,
)
def vbf_candidates(
    self: Producer, 
    events: ak.Array, 
    jet_collection: ak.Array,
    deta_cut = 3,
    invmass_cut = 500,
    region="",
    **kwargs
) -> ak.Array:

    # if jet_collection: jet_collection = events.Lightjet TODO:

    jet_collection = jet_collection[ak.argsort(jet_collection.pt, ascending=False)]
    jet_collection = ak.pad_none(jet_collection, 2)

    for i in [0,1]:
        for var in ["pt", "eta", "phi", "mass"]:
            events = set_ak_column_f32(events, f"customjet{i}_{var}", jet_collection[:, i][var])
    #events = set_ak_column(events, "Forwardjet", ak.pad_none(events.Lightjet, 2))
    # gp_ghost = ak.zip({f: EMPTY_FLOAT for f in sec.fields}, with_name="GenParticle")
    vbf_pairs = ak.combinations((jet_collection), 2)
    vbf1, vbf2 = ak.unzip(vbf_pairs)

    # define requirements for vbf pair candidates
    vbf_pairs["deta"] = abs(vbf1.eta - vbf2.eta)
    vbf_pairs["invmass"] = (vbf1 + vbf2).mass

    vbf_mask = (vbf_pairs.deta > deta_cut) & (vbf_pairs.invmass > invmass_cut)
    vbf_pairs = vbf_pairs[vbf_mask]

    # choose the vbf pair based on maximum delta eta
    chosen_vbf_pair = vbf_pairs[ak.singletons(ak.argmax(vbf_pairs.deta, axis=1))]

    # get the local indices (pt sorted)
    vbf1, vbf2 = [chosen_vbf_pair[i] for i in ["0", "1"]]
    vbf_jets = ak.concatenate([vbf1, vbf2], axis=1)
    vbf_jets = vbf_jets[ak.argsort(vbf_jets.pt, ascending=False)]
    vbf_jets = ak.pad_none(vbf_jets, 2)

    # low level features of VBF canoddates
    for i in [1,2]:
        for var in ["pt", "eta", "phi", "mass"]:
            events = set_ak_column_f32(events, f"vbf{i}_{var}", vbf_jets[:, i-1][var])

    vbfpair = (vbf_jets[:,0]+vbf_jets[:,1])
    vbfpair["deta"] = abs(vbf_jets[:,0].eta-vbf_jets[:,1].eta)
    vbf_tag = ak.sum(vbf_jets.pt > 0, axis=1) >= 2 
    vbfpair["tag"] = ak.fill_none(ak.nan_to_none(vbf_tag), 0)

    events = set_ak_column(events, f"vbf_invmass{region}", vbfpair.mass)
    events = set_ak_column(events, f"vbf_deta{region}", vbfpair.deta)
    events = set_ak_column(events, f"vbf_tag{region}", vbfpair.tag)

    #TODO: padden 
    # __import__("IPython").embed()
    columns = self.produces | set([f"vbf_invmass{region}",f"vbf_deta{region}",f"vbf_tag{region}"])
    for col in columns:
        events = set_ak_column_f32(events, col, ak.fill_none(ak.nan_to_none(events[col]), ZERO_PADDING_VALUE))

    return events

@producer(
    uses={prepare_objects,
          "{Forwardjet,Lightjet}.{pt,eta,phi,mass}",
          vbf_candidates,
    },
    produces={vbf_candidates},
    region="",
    forward_pt_cut=50,
    barrel_pt_cut=30,
)
def vbf_candidates_incl(self:Producer, events: ak.Array, **kwargs)-> ak.Array:

    events = self[prepare_objects](events, **kwargs)

    jet_collection = ak.concatenate([events.Lightjet,events.Forwardjet], axis=-1)
    events = self[vbf_candidates](events,jet_collection=jet_collection,deta_cut=3,invmass_cut=500,region=self.region,**kwargs)

    return events


@vbf_candidates_incl.init
def vbf_candidates_incl_init(self: Producer) -> None:
    
    self.produces.add(f"vbf_deta{self.region}")
    self.produces.add(f"vbf_invmass{self.region}")
    self.produces.add(f"vbf_tag{self.region}")
    # self.uses |= {"Forwardjet.{pt,eta,phi,mass}", "Lightjet.{pt,eta,phi,mass}"}
    add_vbf_variables(self.config_inst)

@producer(
    uses={prepare_objects,
          "Lightjet.{pt,eta,phi,mass}",
          vbf_candidates,
    },
    produces={vbf_candidates},
    region="_barrel",
)
def vbf_candidates_barrel(self:Producer, events: ak.Array, **kwargs)-> ak.Array:

    events = self[prepare_objects](events, **kwargs)

    jet_collection = events.Lightjet
    events = self[vbf_candidates](
        events,jet_collection=jet_collection,deta_cut=3,invmass_cut=500,region=self.region,**kwargs
        )

    return events


@vbf_candidates_barrel.init
def vbf_candidates_barrel_init(self: Producer) -> None:
    
    self.produces.add(f"vbf_deta{self.region}")
    self.produces.add(f"vbf_invmass{self.region}")
    self.produces.add(f"vbf_tag{self.region}")
    # self.uses |= {"Forwardjet.{pt,eta,phi,mass}", "Lightjet.{pt,eta,phi,mass}"}
    add_vbf_variables(self.config_inst)