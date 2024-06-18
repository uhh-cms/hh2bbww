# coding: utf-8

"""
Custom jet energy calibration methods that disable data uncertainties (for searches).
"""

from __future__ import annotations

import functools

from columnflow.util import maybe_import, try_float
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
jec_nominal = jec.derive("jec_nominal", cls_dict={"uncertainty_sources": ["Total"]})


@calibrator(
    uses={
        "Jet.pt", "Jet.mass",
    },
    produces={"Jet.pt", "Jet.mass"},
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

    The *btag_wp* for jet masking is using the "medium" WP per default, but can be customized via config.
    The *b_tagger* is set to "deepjet" for run 2 and "particlenet" for run 3.

    Documentation: https://twiki.cern.ch/twiki/bin/view/Main/BJetRegression

    :param events: Awkward array containing events to process
    :param jet_mask: Optional awkward array containing a mask on which to apply the bjet regression.
        Jet.pt > 20 and Jet b-score > btag_wp is always required
    """

    # apply regression only for jet pt > 20 (docu: https://twiki.cern.ch/twiki/bin/view/Main/BJetRegression)
    # NOTE: we need to apply bjet regression for each jet variation separately.
    # the implementation is not super clean since variations are not included in the uses and produces
    jet_variations = [f for f in events.Jet.fields if f.startswith("pt") and "raw" not in f]

    for variation in jet_variations:
        pt = variation
        mass = variation.replace("pt", "mass")

        default_jet_mask = (events.Jet[pt] > 20)
        if self.btag_wp:
            btag_wp = self.btag_wp
            if not try_float(self.btag_wp):
                btag_wp = self.config_inst.x.btag_working_points[self.b_tagger][self.btag_wp]
            default_jet_mask = default_jet_mask & (events.Jet[self.b_score_column] > btag_wp)

        if jet_mask:
            combined_jet_mask = jet_mask & default_jet_mask
        else:
            combined_jet_mask = default_jet_mask

        jet_pt = ak.where(combined_jet_mask, events.Jet[pt] * events.Jet[self.b_reg_column], events.Jet[pt])
        jet_mass = ak.where(combined_jet_mask, events.Jet[mass] * events.Jet[self.b_reg_column], events.Jet[mass])
        events = set_ak_column_f32(events, f"Jet.{pt}", jet_pt)
        events = set_ak_column_f32(events, f"Jet.{mass}", jet_mass)

    return events


@bjet_regression.init
def bjet_regression_init(self: Calibrator):
    # setup only required by CalibrateEvents task itself
    if self.task and self.task.task_family == "cf.CalibrateEvents":
        self.b_tagger = {
            2: "deepjet",
            3: "particlenet",
        }[self.config_inst.x.run]

        self.b_score_column = {
            "particlenet": "btagPNetB",
            "deepjet": "btagDeepFlavB",
        }[self.b_tagger]

        self.b_reg_column = {
            "particlenet": "PNetRegPtRawCorr",
            "deepjet": "bRegCorr",
        }[self.b_tagger]

        self.btag_wp = self.config_inst.x("btag_wp", "medium")

        self.uses.add(f"Jet.{self.b_score_column}")
        self.uses.add(f"Jet.{self.b_reg_column}")
