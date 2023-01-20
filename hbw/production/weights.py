# coding: utf-8

"""
Column production methods related to generic event weights.
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.selection import SelectionResult
from columnflow.production import Producer, producer
from columnflow.production.pileup import pu_weight
from columnflow.production.normalization import normalization_weights
from columnflow.production.electron import electron_weights
from columnflow.production.muon import muon_weights
from columnflow.production.btag import btag_weights
from columnflow.production.scale import murmuf_weights, murmuf_envelope_weights
from columnflow.production.pdf import pdf_weights
from hbw.production.normalized_weights import normalized_weight_factory
from hbw.production.normalized_btag import normalized_btag_weights

ak = maybe_import("awkward")


@producer(
    uses={pu_weight, btag_weights, murmuf_envelope_weights, murmuf_weights, pdf_weights},
    # don't save btag_weights to save storage space, since we can reproduce them in ProduceColumns
    produces={pu_weight, murmuf_envelope_weights, murmuf_weights, pdf_weights},
)
def event_weights_to_normalize(self: Producer, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers that are typically called as part of SelectEvents
    since it is required to normalize them before applying certain event selections.
    """
    if self.dataset_inst.is_data:
        raise Exception("attempt to compute event weights in data")

    # compute pu weights
    events = self[pu_weight](events, **kwargs)

    # compute btag SF weights (for renormalization tasks)
    events = self[btag_weights](events, jet_mask=results.aux["jet_mask"], **kwargs)

    # ugly hack for st_schannel_had_amcatnlo sample (nominal entry missing in LHEScaleWeights)
    # TODO: there is still some issue with that sample...
    if ak.all(ak.num(events.LHEScaleWeight, axis=1) == 8):
        print(
            f"In dataset {self.dataset_inst.name}: number of LHEScaleWeights is always " +
            "8 instead of the expected 9. It is assumed, that the missing entry is the " +
            "nominal one and all other entries are in correct order",
        )
        scale_tmp = ak.to_regular(events.LHEScaleWeight)
        events = set_ak_column(events, "LHEScaleWeight", ak.concatenate(
            [scale_tmp[:, :4], ak.ones_like(scale_tmp[:, [0]]), scale_tmp[:, 4:]], axis=1,
        ))

    # compute scale weights
    events = self[murmuf_envelope_weights](events, **kwargs)

    # read out mur and weights
    events = self[murmuf_weights](events, **kwargs)

    # compute pdf weights
    events = self[pdf_weights](events, **kwargs)

    return events


normweights = normalized_weight_factory(
    producer_name="normweights",
    weight_producers={event_weights_to_normalize},
)


@producer(
    uses={
        normalization_weights, electron_weights, muon_weights, btag_weights,
        normweights, normalized_btag_weights,
    },
    produces={
        normalization_weights, electron_weights, muon_weights, btag_weights,
        normweights, normalized_btag_weights,
    },
)
def event_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers that are typically called in ProduceColumns.
    """
    if self.dataset_inst.is_data:
        raise Exception("attempt to compute event weights in data")

    # compute normalization weights
    events = self[normalization_weights](events, **kwargs)

    # compute btag SF weights
    events = self[btag_weights](events, **kwargs)

    # compute electron and muon SF weights
    events = self[electron_weights](events, **kwargs)
    events = self[muon_weights](events, **kwargs)

    # normalize event weights using stats
    events = self[normweights](events, **kwargs)
    events = self[normalized_btag_weights](events, **kwargs)

    return events
