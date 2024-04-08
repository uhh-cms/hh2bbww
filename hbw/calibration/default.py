# coding: utf-8

"""
Calibration methods.
"""

import law

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.jets import jec, jer
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import

from hbw.calibration.jet import jec_nominal, bjet_regression

ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)


@calibrator(
    uses={deterministic_seeds},
    produces={deterministic_seeds},
    skip_jecunc=True,
    bjet_regression=True,
)
def base(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    events = self[deterministic_seeds](events, **kwargs)

    logger.info(f"Running calibrators '{[calib.cls_name for calib in self.calibrators]}' (in that order)")
    for calibrator_inst in self.calibrators:
        events = self[calibrator_inst](events, **kwargs)

    return events


@base.init
def base_init(self: Calibrator) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    # list of calibrators to apply (in that order)
    self.calibrators = []

    if self.dataset_inst.is_data or self.skip_jecunc:
        self.calibrators.append(jec_nominal)
    else:
        self.calibrators.append(jec)

    if self.bjet_regression:
        self.calibrators.append(bjet_regression)

    # run JER only on MC
    # and not for 2022 (TODO: update as soon as JER is done for Summer22)
    if self.dataset_inst.is_mc and not self.config_inst.campaign.x.year == 2022:
        self.calibrators.append(jer)

    self.uses |= set(self.calibrators)
    self.produces |= set(self.calibrators)


default = base.derive("default", cls_dict=dict(skip_jecunc=False, bjet_regression=False))
skip_jecunc = base.derive("skip_jecunc", cls_dict=dict(skip_jecunc=True, bjet_regression=False))
with_b_reg = base.derive("with_b_reg", cls_dict=dict(skip_jecunc=True, bjet_regression=True))
full = base.derive("full", cls_dict=dict(skip_jecunc=False, bjet_regression=True))
