# coding: utf-8

"""
Selectors to set ak columns for gen particles of hh2bbww
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, Route, EMPTY_FLOAT
from columnflow.selection import Selector, SelectionResult, selector
from hbw.production.gen_hbw_decay import gen_hbw_decay_products

ak = maybe_import("awkward")


@selector(
    uses={
        "gen_hbw_decay"
    },
    produces={
        "cutflow.h1_pt", "cutflow.h2_pt", "cutflow.b1_pt", "cutflow.b2_pt",
        "cutflow.wlep_pt", "cutflow.whad_pt", "cutflow.l_pt", "cutflow.nu_pt",
        "cutflow.q1_pt", "cutflow.q2_pt", "cutflow.foo_pt"
    },
)
def gen_hbw_decay_features(self: Selector, events: ak.Array, **kwargs) -> ak.Array:

    events = set_ak_column(events, "cutflow.h1_pt", events.gen_hbw_decay.h1.pt)
    events = set_ak_column(events, "cutflow.h2_pt", events.gen_hbw_decay.h2.pt)
    events = set_ak_column(events, "cutflow.b1_pt", events.gen_hbw_decay.b1.pt)
    events = set_ak_column(events, "cutflow.b2_pt", events.gen_hbw_decay.b2.pt)
    events = set_ak_column(events, "cutflow.wlep_pt", events.gen_hbw_decay.wlep.pt)
    events = set_ak_column(events, "cutflow.whad_pt", events.gen_hbw_decay.whad.pt)
    events = set_ak_column(events, "cutflow.l_pt", events.gen_hbw_decay.l.pt)
    events = set_ak_column(events, "cutflow.nu_pt", events.gen_hbw_decay.nu.pt)
    events = set_ak_column(events, "cutflow.q1_pt", events.gen_hbw_decay.q1.pt)
    events = set_ak_column(events, "cutflow.q2_pt", events.gen_hbw_decay.q2.pt)
    # events = set_ak_column(events, "cutflow.foo1_pt", events.gen_hbw_decay.foo.pt[: ,0])
    
    return events


# gen_hbw_decay = ak.zip({
#     "h1": h[:, 0],
#     "h2": h[:, 1],
#     "b1": b1,
#     "b2": b2,
#     "wlep": wlep,
#     "whad": whad,
#     "l": lepton,
#     "nu": neutrino,
#     "q1": q_dtype,
#     "q2": q_utype,
#     "foo": foo,
# })
