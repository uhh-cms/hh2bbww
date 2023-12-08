# coding: utf-8

"""
Custom jet energy calibration methods that disable data uncertainties (for searches).
"""

from __future__ import annotations

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.calibration.cms.jets import jec
from columnflow.calibration import calibrator, Calibrator

# custom jec calibrator that only runs nominal correction
jec_nominal = jec.derive("jec_nominal", cls_dict={"uncertainty_sources": []})

ak = maybe_import("awkward")


@calibrator(
    uses={
        "Jet.pt", "Jet.mass", "Jet.bRegCorr",  # "Jet.bRegRes",
    },
    produces={"Jet.pt"},
)
def bjet_regression(
    self: Calibrator,
    events: ak.Array,
    jet_mask: ak.Array | None = None,
    **kwargs,
):
    """
    Calibrator to apply the bjet regression.

    This Calibrator should be applied after the jet energy corrections (jec) but before the jet
    energy smearing (jer).

    Documentation: https://twiki.cern.ch/twiki/bin/view/Main/BJetRegression

    :param events: Awkward array containing events to process
    :param jet_mask: Optional awkward array containing a mask on which to apply the bjet regression.
        A Jet.pt > 20 is always required

    """

    # apply regression only for jet pt > 20 (docu: https://twiki.cern.ch/twiki/bin/view/Main/BJetRegression)
    if jet_mask:
        jet_mask = jet_mask & (events.Jet.pt > 20)
    else:
        jet_mask = (events.Jet.pt > 20)

    jet_pt = ak.where(jet_mask, events.Jet.pt * events.Jet.bRegCorr, events.Jet.pt)
    jet_mass = ak.where(jet_mask, events.Jet.mass * events.Jet.bRegCorr, events.Jet.mass)
    events = set_ak_column(events, "Jet.pt", jet_pt)
    events = set_ak_column(events, "Jet.mass", jet_mass)

    return events
