# coding: utf-8

"""
Calibration methods.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.jets import jec, jer
from columnflow.production import producer
from columnflow.production.mc_weight import mc_weight
from columnflow.production.seeds import deterministic_seeds
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

from hbw.calibration.jet import jec_nominal

ak = maybe_import("awkward")


@producer(
    uses={"mc_weight"},
    produces={"mc_weight"},
)
def large_weights_killer(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Simple producer that sets eventweights to 0 when too large.
    """
    if self.dataset_inst.is_data:
        raise Exception("large_weights_killer is only callable for MC")

    # TODO: figure out a good threshold when events are considered unphysical
    median_weight = ak.sort(abs(events.mc_weight))[int(len(events) / 2)]
    weight_too_large = abs(events.mc_weight) > 1000 * median_weight
    events = set_ak_column(events, "mc_weight", ak.where(weight_too_large, 0, events.mc_weight))

    return events


@calibrator(
    uses={deterministic_seeds},
    produces={deterministic_seeds},
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
        events = self[large_weights_killer](events, **kwargs)

    events = self[deterministic_seeds](events, **kwargs)

    if self.dataset_inst.is_data:
        events = self[jec_nominal](events, **kwargs)
    else:
        events = self[jec](events, **kwargs)
        events = self[jer](events, **kwargs)

    return events


@default.init
def default_init(self: Calibrator) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    if self.dataset_inst.is_data:
        calibrators = {jec_nominal}
    else:
        calibrators = {mc_weight, jec, jer, large_weights_killer}

    self.uses |= calibrators
    self.produces |= calibrators


@calibrator(
    uses={deterministic_seeds},
    produces={deterministic_seeds},
)
def skip_jecunc(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """ only uses jec_nominal for test purposes """
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
        events = self[large_weights_killer](events, **kwargs)

    events = self[deterministic_seeds](events, **kwargs)

    if self.dataset_inst.is_data:
        events = self[jec_nominal](events, **kwargs)
    else:
        events = self[jec_nominal](events, **kwargs)
        events = self[jer](events, **kwargs)

    return events


@skip_jecunc.init
def skip_jecunc_init(self: Calibrator) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    if self.dataset_inst.is_data:
        calibrators = {jec_nominal}
    else:
        calibrators = {mc_weight, jec_nominal, jer, large_weights_killer}

    self.uses |= calibrators
    self.produces |= calibrators
