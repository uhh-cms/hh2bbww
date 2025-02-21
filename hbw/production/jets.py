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
