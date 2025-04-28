# coding: utf-8

"""
Calibration methods.
"""

import law

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.met import met_phi
from columnflow.calibration.cms.jets import jec, jer
from columnflow.production.cms.jet import msoftdrop
from columnflow.calibration.cms.egamma import electrons
from columnflow.production.cms.seeds import (
    deterministic_seeds, deterministic_electron_seeds, deterministic_event_seeds,
)
from columnflow.production.cms.electron import electron_sceta
from columnflow.util import maybe_import, try_float
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT

from hbw.util import MET_COLUMN

from hbw.calibration.jet import bjet_regression


ak = maybe_import("awkward")
np = maybe_import("numpy")


logger = law.logger.get_logger(__name__)


# customized electron calibrator (also needs deterministic event seeds...)
electrons.deterministic_seed_index = 0


@calibrator(
    version=2,
    uses={electron_sceta, deterministic_event_seeds, deterministic_electron_seeds},
    produces={deterministic_event_seeds, deterministic_electron_seeds},
)
def ele(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Electron calibrator, combining scale and resolution.
    """
    # obtain the electron super cluster eta needed for the calibration
    events = self[electron_sceta](events, **kwargs)

    events = self[deterministic_event_seeds](events, **kwargs)
    events = self[deterministic_electron_seeds](events, **kwargs)

    # apply the electron calibration
    events = self[self.electron_calib_cls](events, **kwargs)
    return events


@ele.init
def ele_init(self: Calibrator) -> None:
    self.electron_calib_cls = electrons

    self.uses |= {self.electron_calib_cls}
    self.produces |= {self.electron_calib_cls}


@calibrator(
    version=2,
    # add dummy produces such that this calibrator will always be run when requested
    # (temporary workaround until init's are only run as often as necessary)
    # TODO: deterministic FatJet seeds
    uses={msoftdrop},
    produces={msoftdrop, "FatJet.pt"},
)
def fatjet(self: Calibrator, events: ak.Array, task, **kwargs) -> ak.Array:
    """
    FatJet calibrator, combining JEC and JER.
    Uses as JER uncertainty either only "Total" for MC or no uncertainty for data.
    """
    if task.local_shift != "nominal":
        raise Exception("FatJet Calibrator should not be run for shifts other than nominal")

    # apply the fatjet JEC and JER
    events = self[self.fatjet_jec_cls](events, **kwargs)
    if self.dataset_inst.is_mc:
        events = self[self.fatjet_jer_cls](events, **kwargs)

    # recalculate the softdrop mass
    events = self[msoftdrop](events, **kwargs)

    return events


@fatjet.init
def fatjet_init(self: Calibrator) -> None:
    # derive calibrators to add settings once
    flag = f"custom_fatjet_calibs_registered_{self.cls_name}"
    if not self.config_inst.x(flag, False):
        fatjet_jec_cls_dict = {
            "jet_name": "FatJet",
            "gen_jet_name": "GenJetAK8",
            # MET propagation is performed in AK4 jet calibrator; fatjet should never use any MET columns
            "propagate_met": False,
            "met_name": "DO_NOT_USE",
            "raw_met_name": "DO_NOT_USE",
        }
        fatjet_jer_cls_dict = fatjet_jec_cls_dict.copy()
        # NOTE: deterministic FatJet seeds are not yet possible to produce
        # fatjet_jer_cls_dict["deterministic_seed_index"] = 0

        # fatjet_jec = jec.derive("fatjet_jec", cls_dict={
        #     **fatjet_jec_cls_dict,
        # })
        self.config_inst.x.fatjet_jec_data_cls = jec.derive("fatjet_jec_data", cls_dict={
            **fatjet_jec_cls_dict,
            "data_only": True,
            "nominal_only": True,
            "uncertainty_sources": [],
        })
        self.config_inst.x.fatjet_jec_total_cls = jec.derive("fatjet_jec_total", cls_dict={
            **fatjet_jec_cls_dict,
            "mc_only": True,
            "nominal_only": True,
            "uncertainty_sources": ["Total"],
        })
        self.config_inst.x.fatjet_jer_cls = jer.derive("deterministic_fatjet_jer", cls_dict=fatjet_jer_cls_dict)

        # change the flag
        self.config_inst.set_aux(flag, True)

    if not getattr(self, "dataset_inst", None):
        return

    # chose the JEC and JER calibrators based on dataset instance
    self.fatjet_jec_cls = (
        self.config_inst.x.fatjet_jec_total_cls if self.dataset_inst.is_mc
        else self.config_inst.x.fatjet_jec_data_cls
    )
    self.fatjet_jer_cls = self.config_inst.x.fatjet_jer_cls

    self.uses |= {self.fatjet_jec_cls, self.fatjet_jer_cls}
    self.produces |= {self.fatjet_jec_cls, self.fatjet_jer_cls}


fatjet_test = fatjet.derive("fatjet_test")


@calibrator(
    uses={deterministic_seeds, MET_COLUMN("{pt,phi}")},
    produces={deterministic_seeds},
    # jec uncertainty_sources: set to None to use config default
    jec_sources=["Total"],
    bjet_regression=True,
    skip_jer=False,
    version=1,
)
def jet_base(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    events = self[deterministic_seeds](events, **kwargs)

    # keep a copy of non-propagated MET to replace infinite values
    pre_calib_met = events[self.config_inst.x.met_name]

    logger.info(f"Running calibrators '{[calib.cls_name for calib in self.calibrators]}' (in that order)")
    for calibrator_inst in self.calibrators:
        events = self[calibrator_inst](events, **kwargs)

    # workaround for infinite values in MET pt/phi
    for route in self.produced_columns:
        col = route.string_column
        m = ~np.isfinite(route.apply(events))
        if ak.any(m):
            # replace infinite values
            replace_value = EMPTY_FLOAT
            if self.config_inst.x.met_name in col:
                # use pre-calibrated MET to replace infinite values of MET pt/phi
                replace_value = pre_calib_met[col.split(".")[-1].split("_")[0]]
            logger.info(
                f"Found infinite values in {col}; Values will be replaced with "
                f"{replace_value if try_float(replace_value) else replace_value[m]}",
            )
            events = set_ak_column(events, col, ak.where(m, replace_value, route.apply(events)))

    return events


@jet_base.init
def jet_base_init(self: Calibrator) -> None:

    # derive calibrators to add settings once
    flag = f"custom_jet_calibs_registered_{self.cls_name}"
    if not self.config_inst.x(flag, False):
        met_name = self.config_inst.x.met_name
        raw_met_name = self.config_inst.x.raw_met_name

        jec_cls_kwargs = {
            "nominal_only": True,
            "met_name": met_name,
            "raw_met_name": raw_met_name,
        }

        # jec calibrators
        self.config_inst.x.calib_jec_full_cls = jec.derive("jec_full", cls_dict={
            **jec_cls_kwargs,
            "mc_only": True,
            "uncertainty_sources": self.jec_sources,
        })
        self.config_inst.x.calib_jec_data_cls = jec.derive("jec_data", cls_dict={
            **jec_cls_kwargs,
            "data_only": True,
            "uncertainty_sources": [],
        })
        # version of jer that uses the first random number from deterministic_seeds
        self.config_inst.x.calib_deterministic_jer_cls = jer.derive("deterministic_jer", cls_dict={
            "deterministic_seed_index": 0,
            "met_name": met_name,
        })
        # derive met_phi calibrator (currently only used in run 2)
        self.config_inst.x.calib_met_phi_cls = met_phi.derive("met_phi", cls_dict={
            "met_name": met_name,
        })
        # change the flag
        self.config_inst.set_aux(flag, True)

    if not getattr(self, "dataset_inst", None):
        return

    # list of calibrators to apply (in that order)
    self.calibrators = []

    # JEC
    jec_cls = (
        self.config_inst.x.calib_jec_full_cls if self.dataset_inst.is_mc
        else self.config_inst.x.calib_jec_data_cls
    )
    self.calibrators.append(jec_cls)

    # BJet regression
    if self.bjet_regression:
        self.calibrators.append(bjet_regression)

    # JER (only for MC)
    jer_cls = self.config_inst.x.calib_deterministic_jer_cls
    if self.dataset_inst.is_mc and not self.skip_jer:
        self.calibrators.append(jer_cls)

    # MET phi (only in run 2)
    if self.config_inst.x.run == 2:
        met_phi_cls = self.config_inst.x.calib_met_phi_cls
        self.calibrators.append(met_phi_cls)

    self.uses |= set(self.calibrators)
    self.produces |= set(self.calibrators)


jec_only = jet_base.derive("jec_only", cls_dict=dict(bjet_regression=False, skip_jer=True))
skip_jer = jet_base.derive("skip_jer", cls_dict=dict(bjet_regression=True, skip_jer=True))
no_breg = jet_base.derive("no_breg", cls_dict=dict(bjet_regression=False))
with_b_reg = jet_base.derive("with_b_reg", cls_dict=dict(bjet_regression=True))
with_b_reg_test = jet_base.derive("with_b_reg_test", cls_dict=dict(bjet_regression=True))
