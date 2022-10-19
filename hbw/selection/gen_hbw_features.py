# coding: utf-8

"""
Selectors to set ak columns for gen particles of hh2bbww
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column  # , Route, EMPTY_FLOAT
from columnflow.selection import Selector, selector

ak = maybe_import("awkward")


@selector(
    uses={
        "gen_hbw_decay",
    },
    produces=set(
        f"cutflow.{gp}_{var}"
        for gp in ["h1", "h2", "b1", "b2", "wlep", "whad", "l", "nu", "q1", "q2", "sec"]  # , "foo1", "foo2"]
        for var in ["pt", "eta", "phi", "mass"]
    ),
)
def gen_hbw_decay_features(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    print("features")
    for var in ["pt", "eta", "phi", "mass"]:
        for gp in ["h1", "h2", "b1", "b2", "wlep", "whad", "l", "nu", "q1", "q2", "sec"]:
            events = set_ak_column(events, f"cutflow.{gp}_{var}", events.gen_hbw_decay[gp][var])

        """
        from IPython import embed; embed()
        for i in range(2):
            events = set_ak_column(
                events, f"cutflow.foo{i}_{var}",
                Route(f"foo[:, {i}, {var}]").apply(events.gen_hbw_decay, EMPTY_FLOAT),
            )
        """
    return events
