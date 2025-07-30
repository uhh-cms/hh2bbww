# coding: utf-8

"""
Production modules related to electrons.
"""

from __future__ import annotations

from functools import partial


import law
from columnflow.util import maybe_import

from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")


set_ak_bool = partial(set_ak_column, value_type=np.bool_)


logger = law.logger.get_logger(__name__)


@producer(
    uses={"run", "PuppiMET.{pt,phi}", "Jet.{pt,eta,phi,mass,neEmEF,chEmEF}"},
    produces={"patchedEcalBadCalibFilter"},
    data_only=True,
)
def ECALBadCalibrationFilter(
    self: Producer, events: ak.Array, **kwargs
) -> ak.Array:
    """
    Producer that applies the ECAL bad calibration filter.

    At the time of writing this, the filter only needs to be applied to 2022 Era F+G
    since these datasets are still Prompt data.

    Resources:
    - https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2?rev=167#ECal_BadCalibration_Filter_Flag
    - https://cms-talk.web.cern.ch/t/noise-met-filters-in-run-3/63346/5

    """
    set(events.run)
    logger.debug(f"{events.run}")
    jet_mask = (
        (events.Jet.pt > 50) &
        (events.Jet.eta > -0.5) &
        (events.Jet.eta < 0.1) &
        (events.Jet.phi > -2.1) &
        (events.Jet.phi < -1.8) &
        (events.Jet.neEmEF > 0.9) &
        (events.Jet.chEmEF > 0.9) &
        (events.Jet.delta_phi(events.PuppiMET) > 2.9)
    )
    reject = (
        (events.run >= 362433) &
        (events.run <= 367144) &
        (events.PuppiMET.pt > 100) &
        (ak.any(jet_mask, axis=1))
    )
    ecal_bad_calibration_mask = ~reject
    logger.info(f"{ak.sum(reject)} / {len(events)} events affected by ECAL bad calibration filter")

    events = set_ak_bool(events, "patchedEcalBadCalibFilter", ecal_bad_calibration_mask)

    return events


@ECALBadCalibrationFilter.skip
def ECALBadCalibrationFilter_skip(self: Producer) -> None:
    if (
        self.config_inst.campaign.x.year == 2022 and
        self.dataset_inst.is_data and
        self.dataset_inst.x.era in "FG"
    ):
        # only run on prompt data
        return False
    return True
