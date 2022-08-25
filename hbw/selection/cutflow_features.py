# coding: utf-8

"""
Selectors to set ak columns for cutflow features
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, Route, EMPTY_FLOAT
from columnflow.selection import Selector, selector

ak = maybe_import("awkward")


@selector(uses={"Jet.pt"}, produces={"cutflow.jet1_pt"})
def cutflow_features(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    # TODO: we would probably like to determine these variables for objects after object definition,
    #       but for now, this takes the unsorted and unmasked object as input
    events = set_ak_column(events, "cutflow.jet1_pt", Route("Jet.pt[:, 0]").apply(events, EMPTY_FLOAT))
    return events
