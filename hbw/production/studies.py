
"""
Column production methods related to higher-level features.
"""

from __future__ import annotations

import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from hbw.production.prepare_objects import prepare_objects
from columnflow.production.cms.dy import recoil_corrected_met

from hbw.util import MET_COLUMN, IF_MC

ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

ZERO_PADDING_VALUE = -10


# TODO: this producer can only be called, when upara/uperp are saved in events.RecoilCorrMET
# from cf.production.cms.dy recoil_corrected_met
@producer(
    uses={
        prepare_objects,
        "{Electron,Muon,Jet,ForwardJet}.{pt,eta,phi,mass}",
        MET_COLUMN("{pt,phi}"),
        IF_MC(recoil_corrected_met),
    },
    produces={
        "met_pt_corr", "met_phi_corr",
        "upara", "uperp",
    },
)
def U_vectorV3(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    events = self[prepare_objects](events, **kwargs)
    met_name = self.config_inst.x.met_name

    if self.dataset_inst.has_tag("is_dy"):

        recoil_corrected_met.jet_name = "InclJet"
        events = self[recoil_corrected_met](events, **kwargs)

        met_pt_corr = events.RecoilCorrMET.pt
        met_phi_corr = events.RecoilCorrMET.phi
        events = set_ak_column_f32(events, "upara", events.RecoilCorrMET.upara)
        events = set_ak_column_f32(events, "uperp", events.RecoilCorrMET.uperp)

    else:
        # Compute the recoil vector U = MET + vis - full
        hll = (events.Lepton[:, 0] + events.Lepton[:, 1])
        upara = events[met_name].pt * np.cos(events[met_name].phi - hll.phi)
        uperp = events[met_name].pt * np.sin(events[met_name].phi - hll.phi)
        events = set_ak_column_f32(events, "upara", upara)
        events = set_ak_column_f32(events, "uperp", uperp)

        met_pt_corr = events[met_name].pt
        met_phi_corr = events[met_name].phi

    events = set_ak_column_f32(events, "met_pt_corr", met_pt_corr)
    events = set_ak_column_f32(events, "met_phi_corr", met_phi_corr)

    for col in ["met_pt_corr", "met_phi_corr", "upara", "uperp"]:
        events = set_ak_column_f32(events, col, ak.fill_none(ak.nan_to_none(events[col]), ZERO_PADDING_VALUE))

    return events
