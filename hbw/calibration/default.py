# coding: utf-8

"""
Calibration methods.
"""

import law

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.met import met_phi
from columnflow.calibration.cms.jets import jec, jer, jer_horn_handling
from columnflow.production.cms.jet import msoftdrop
from columnflow.calibration.cms.egamma import electrons
from columnflow.production.cms.seeds import (
    deterministic_object_seeds, deterministic_jet_seeds, deterministic_electron_seeds,
    deterministic_event_seeds,
)
from columnflow.production.cms.electron import electron_sceta
from columnflow.util import maybe_import, try_float
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT

from hbw.util import MET_COLUMN

from hbw.calibration.jet import bjet_regression


ak = maybe_import("awkward")
np = maybe_import("numpy")


logger = law.logger.get_logger(__name__)


# customized electron calibrator (also needs deterministic event seeds)
electrons.deterministic_seed_index = 0

# custom fatjet seeds
deterministic_fatjet_seeds = deterministic_object_seeds.derive(
    "deterministic_fatjet_seeds",
    cls_dict={
        "object_field": "FatJet",
        "prime_offset": 80,
    },
)


@calibrator(
    uses={
        deterministic_event_seeds,
        deterministic_jet_seeds, deterministic_fatjet_seeds, deterministic_electron_seeds,
    },
    produces={
        deterministic_event_seeds,
        deterministic_jet_seeds, deterministic_fatjet_seeds, deterministic_electron_seeds,
    },
)
def deterministic_seeds_calibrator(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Calibrator that produces deterministic seeds for events and objects (Jet, Electron, and FatJet).
    This is defined as a Calibrator such that this can run as separate task before the actual calibration.
    """
    # create the event seeds
    events = self[deterministic_event_seeds](events, **kwargs)

    # create the jet seeds
    events = self[deterministic_jet_seeds](events, **kwargs)

    # create the electron seeds
    events = self[deterministic_electron_seeds](events, **kwargs)

    # create the fatjet seeds
    events = self[deterministic_fatjet_seeds](events, **kwargs)

    return events


@calibrator
def seeds_user_base(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    raise Exception("This is a base class for user-defined calibrators that use the seeds.")


@seeds_user_base.requires
def seeds_user_base_requires(self: Calibrator, task: law.Task, reqs: dict) -> None:
    if "seeds" in reqs:
        return

    # initialize the deterministic seeds calibrator by hand
    calibrator_inst = task.build_calibrator_inst("deterministic_seeds_calibrator", params={
        "dataset": task.dataset,
        "dataset_inst": task.dataset_inst,
        "config": task.config,
        "config_inst": task.config_inst,
        "analysis": task.analysis,
        "analysis_inst": task.analysis_inst,
    })

    from columnflow.tasks.calibration import CalibrateEvents
    reqs["seeds"] = CalibrateEvents.req(
        task,
        calibrator=deterministic_seeds_calibrator.cls_name,
        calibrator_inst=calibrator_inst,
    )


@seeds_user_base.setup
def seeds_user_base_setup(
    self: Calibrator, task: law.Task, reqs: dict, inputs: dict, reader_targets: law.util.InsertableDict,
) -> None:
    reader_targets["seeds"] = inputs["seeds"]["columns"]


@seeds_user_base.calibrator(
    version=3,
    uses={electron_sceta, deterministic_seeds_calibrator.PRODUCES},
    produces={"Electron.pt"},  # dummy produces to ensure this calibrator is run
)
def ele(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Electron calibrator, combining scale and resolution.
    """
    # obtain the electron super cluster eta needed for the calibration
    events = self[electron_sceta](events, **kwargs)

    # apply the electron calibration
    events = self[self.electron_calib_cls](events, **kwargs)
    return events


@ele.init
def ele_init(self: Calibrator) -> None:
    self.electron_calib_cls = electrons

    self.uses |= {self.electron_calib_cls}
    self.produces |= {self.electron_calib_cls}


@seeds_user_base.calibrator(
    version=3,
    uses={msoftdrop, deterministic_seeds_calibrator.PRODUCES},
    produces={msoftdrop, "FatJet.pt"},  # never leave this empty, otherwise the calibrator will not run
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
        fatjet_jer_cls_dict["deterministic_seed_index"] = 0

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


@seeds_user_base.calibrator(
    uses={deterministic_seeds_calibrator.PRODUCES, MET_COLUMN("{pt,phi}")},
    # We produce event seeds here again to be able to keep them after ReduceEvents (required for NN training).
    produces={deterministic_event_seeds.PRODUCES},
    # jec uncertainty_sources: set to None to use config default
    jec_sources=["Total"],
    bjet_regression=True,
    skip_jer=False,
    jer_horn_handling=False,
    version=3,
)
def jet_base(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
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
        base_jer_cls = jer_horn_handling if self.jer_horn_handling else jer
        self.config_inst.x.calib_deterministic_jer_cls = base_jer_cls.derive("deterministic_jer", cls_dict={
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

ak4 = jet_base.derive("ak4", cls_dict=dict(
    bjet_regression=False,
    skip_jer=False,
    jer_horn_handling=True,
))


@seeds_user_base.calibrator(
    uses={ak4, fatjet, ele},
    produces={ak4, fatjet, ele},
    version=0,
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Default calibration method that applies the jet, fatjet and electron calibrators.
    """
    # apply the jet calibrator
    events = self[ak4](events, **kwargs)

    # apply the fatjet calibrator
    events = self[fatjet](events, **kwargs)

    # apply the electron calibrator
    events = self[ele](events, **kwargs)

    return events
