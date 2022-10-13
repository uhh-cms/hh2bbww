# coding: utf-8

"""
Selectors to set ak columns for gen particles of hh2bbww
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, Route, EMPTY_FLOAT
from columnflow.selection import Selector, SelectionResult, selector

ak = maybe_import("awkward")


@selector(
    uses={
        "gen_hbw_decay"
    },
    produces={
        "gen_hbw_decay.h1_pt", "gen_hbw_decay.h2_pt", "gen_hbw_decay.b1_pt", "gen_hbw_decay.b2_pt",
        "gen_hbw_decay.wlep_pt", "gen_hbw_decay.whad_pt", "gen_hbw_decay.l_pt", "gen_hbw_decay.nu_pt",
        "gen_hbw_decay.q1_pt", "gen_hbw_decay.q2_pt", "gen_hbw_decay.foo_pt"
    },
)
def gen_hbw_decay_features(self: Selector, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:

    jets = events.Jet[jet_indices]
    events = set_ak_column(events, "gen_hbw_decay.h1_pt", gen_hbw_decay.h1.pt)
    events = set_ak_column(events, "gen_hbw_decay.h2_pt", gen_hbw_decay.h2.pt)
    events = set_ak_column(events, "gen_hbw_decay.b1_pt", gen_hbw_decay.b1.pt)
    events = set_ak_column(events, "gen_hbw_decay.b2_pt", gen_hbw_decay.b2.pt)
    events = set_ak_column(events, "gen_hbw_decay.wlep1_pt", gen_hbw_decay.wlep1.pt)
    events = set_ak_column(events, "gen_hbw_decay.whad_pt", gen_hbw_decay.whad.pt)
    events = set_ak_column(events, "gen_hbw_decay.l_pt", gen_hbw_decay.l.pt)
    events = set_ak_column(events, "gen_hbw_decay.nu_pt", gen_hbw_decay.nu.pt)
    events = set_ak_column(events, "gen_hbw_decay.q1_pt", gen_hbw_decay.q1.pt)
    events = set_ak_column(events, "gen_hbw_decay.g2_pt", gen_hbw_decay.g2.pt)
    events = set_ak_column(events, "gen_hbw_decay.foo1_pt", gen_hbw_decay.foo.pt[: ,0])

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
