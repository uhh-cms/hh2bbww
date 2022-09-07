# coding: utf-8

"""
Calibration methods.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.production.mc_weight import mc_weight
from columnflow.production.seeds import deterministic_seeds
from columnflow.calibration.jets import jec, jer
from columnflow.util import maybe_import

ak = maybe_import("awkward")


# custom jec calibrator that only runs nominal correction
jec_nominal = jec.derive("jec_nominal", cls_dict={"uncertainty_sources": []})


@calibrator(
    uses={mc_weight, deterministic_seeds, jec_nominal, jer},
    produces={mc_weight, deterministic_seeds, jec_nominal, jer},
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    events = self[mc_weight](events, **kwargs)
    events = self[deterministic_seeds](events, **kwargs)
    events = self[jec_nominal](events, **kwargs)
    events = self[jer](events, **kwargs)

    return events
