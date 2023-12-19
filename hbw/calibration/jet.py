# coding: utf-8

"""
Custom jet energy calibration methods that disable data uncertainties (for searches).
"""

from __future__ import annotations

import functools

from columnflow.util import maybe_import, test_float
from columnflow.columnar_util import set_ak_column
from columnflow.calibration.cms.jets import jec
from columnflow.calibration import calibrator, Calibrator


np = maybe_import("numpy")
ak = maybe_import("awkward")


#
# helper functions
#

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


# custom jec calibrator that only runs nominal correction
jec_nominal = jec.derive("jec_nominal", cls_dict={"uncertainty_sources": []})


@calibrator(
    uses={
        "Jet.pt", "Jet.mass", "Jet.btagDeepFlavB", "Jet.bRegCorr",
    },
    produces={"Jet.pt", "Jet.mass"},
    btag_wp=None,
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
        Jet.pt > 20 and Jet-btagDeepFlavB > btag_wp is always required
    """

    # apply regression only for jet pt > 20 (docu: https://twiki.cern.ch/twiki/bin/view/Main/BJetRegression)

    default_jet_mask = (events.Jet.pt > 20)
    if self.btag_wp:
        btag_wp = self.btag_wp
        if not test_float(self.btag_wp):
            btag_wp = self.config_inst.x.btag_working_points.deepjet[self.btag_wp]
        default_jet_mask = default_jet_mask & (events.Jet.btagDeepFlavB > btag_wp)

    if jet_mask:
        jet_mask = jet_mask & default_jet_mask
    else:
        jet_mask = default_jet_mask

    jet_pt = ak.where(jet_mask, events.Jet.pt * events.Jet.bRegCorr, events.Jet.pt)
    jet_mass = ak.where(jet_mask, events.Jet.mass * events.Jet.bRegCorr, events.Jet.mass)
    events = set_ak_column_f32(events, "Jet.pt", jet_pt)
    events = set_ak_column_f32(events, "Jet.mass", jet_mass)

    return events
