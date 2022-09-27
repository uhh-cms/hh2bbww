# coding: utf-8

"""
Calibration methods.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.jets import jets
from columnflow.production.mc_weight import mc_weight
from columnflow.production.seeds import deterministic_seeds
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

from hbw.calibration.jet import jet_energy

ak = maybe_import("awkward")


@calibrator(
    uses={"mc_weight"},
    produces={"mc_weight"},
)
def large_weights_killer(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Simple calibrator that sets eventweights to 0 when too large.
    """
    if getattr(self, "dataset_inst", None) and self.dataset_inst.is_data:
        return events

    # TODO: figure out a good threshold when events are considered unphysical
    median_weight = ak.sort(abs(events.mc_weight))[int(len(events) / 2)]
    weight_too_large = abs(events.mc_weight) > 1000 * median_weight
    events = set_ak_column(events, "mc_weight", ak.where(weight_too_large, 0, events.mc_weight))

    return events


@calibrator(
    uses={mc_weight, deterministic_seeds, jets, large_weights_killer},
    produces={mc_weight, deterministic_seeds, jets, large_weights_killer},
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    events = self[mc_weight](events, **kwargs)
    events = self[deterministic_seeds](events, **kwargs)
    events = self[jets](events, **kwargs)
    events = self[large_weights_killer](events, **kwargs)

    return events


@calibrator(
    uses={mc_weight, deterministic_seeds, jet_energy, large_weights_killer},
    produces={mc_weight, deterministic_seeds, jet_energy, large_weights_killer},
)
def skip_jecunc(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """ only uses jec_nominal for test purposes """
    events = self[mc_weight](events, **kwargs)
    events = self[deterministic_seeds](events, **kwargs)
    events = self[jet_energy](events, **kwargs)
    events = self[large_weights_killer](events, **kwargs)

    return events
