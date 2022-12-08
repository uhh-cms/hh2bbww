# coding: utf-8

"""
Column production methods related to generic event weights.
"""

from columnflow.util import maybe_import
from columnflow.selection import SelectionResult
from columnflow.production import Producer, producer
from columnflow.production.pileup import pu_weight
from columnflow.production.normalization import normalization_weights
from columnflow.production.electron import electron_weights
from columnflow.production.muon import muon_weights
from columnflow.production.btag import btag_weights
# TODO: move them to columnflow
from hbw.production.scale_pdf_weights import pdf_weights, scale_weights, murmuf_weights

ak = maybe_import("awkward")


@producer(
    uses={pu_weight, btag_weights, scale_weights, murmuf_weights, pdf_weights},
    produces={pu_weight, btag_weights, scale_weights, murmuf_weights, pdf_weights},
)
def event_weights_to_normalize(self: Producer, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers that are typically called as part of SelectEvents
    since it is required to normalize them before applying certain event selections.
    """
    # compute pu weights
    events = self[pu_weight](events, **kwargs)

    # compute btag SF weights
    events = self[btag_weights](events, jet_mask=results.aux["jet_mask"], **kwargs)

    # compute scale weights (TODO testing)
    events = self[scale_weights](events, **kwargs)

    # read out mur and weights (TODO testing)
    events = self[murmuf_weights](events, **kwargs)

    # compute pdf weights (TODO: testing)
    events = self[pdf_weights](events, **kwargs)

    return events


@producer(
    uses={normalization_weights, electron_weights, muon_weights},
    produces={normalization_weights, electron_weights, muon_weights},
)
def event_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers that are typically called in ProduceColumns.
    """
    # compute normalization weights
    events = self[normalization_weights](events, **kwargs)

    # compute electron and muon SF weights
    events = self[electron_weights](events, **kwargs)
    events = self[muon_weights](events, **kwargs)

    return events
