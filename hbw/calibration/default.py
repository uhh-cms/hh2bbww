# coding: utf-8

"""
Calibration methods.
"""

import law

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.met import met_phi
from columnflow.calibration.cms.jets import jec, jer
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import, try_float
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT

from hbw.util import MET_COLUMN

from hbw.calibration.jet import bjet_regression

ak = maybe_import("awkward")
np = maybe_import("numpy")


logger = law.logger.get_logger(__name__)


@calibrator(
    # jec uncertainty_sources: set to None to use config default
    jec_sources=["Total"],
    version=1,
    # add dummy produces such that this calibrator will always be run when requested
    # (temporary workaround until init's are only run as often as necessary)
    produces={"FatJet.pt"},
)
def fatjet(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    FatJet calibrator, combining JEC and JER.
    """
    if self.task.local_shift != "nominal":
        raise Exception("FatJet Calibrator should not be run for shifts other than nominal")

    # apply the fatjet JEC and JER
    events = self[self.fatjet_jec_cls](events, **kwargs)
    if self.dataset_inst.is_mc:
        events = self[self.fatjet_jer_cls](events, **kwargs)

    return events


@fatjet.init
def fatjet_init(self: Calibrator) -> None:
    if not self.task or self.task.task_family != "cf.CalibrateEvents":
        # init only required for task itself
        return

    if not getattr(self, "dataset_inst", None):
        return

    # list of calibrators to apply (in that order)
    self.calibrators = []

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

    uncertainty_sources = [] if self.dataset_inst.is_data else self.jec_sources
    jec_cls_name = f"fatjet_jec{'_nominal' if uncertainty_sources == [] else ''}"
    self.fatjet_jec_cls = jec.derive(jec_cls_name, cls_dict={
        **fatjet_jec_cls_dict,
        "uncertainty_sources": uncertainty_sources,
    })
    self.fatjet_jer_cls = jer.derive("deterministic_fatjet_jer", cls_dict=fatjet_jer_cls_dict)

    self.uses |= {self.fatjet_jec_cls, self.fatjet_jer_cls}
    self.produces |= {self.fatjet_jec_cls, self.fatjet_jer_cls}


@calibrator(
    uses={deterministic_seeds, MET_COLUMN("{pt,phi}")},
    produces={deterministic_seeds},
    # jec uncertainty_sources: set to None to use config default
    jec_sources=["Total"],
    bjet_regression=True,
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
    if not self.task or self.task.task_family != "cf.CalibrateEvents":
        # init only required for task itself
        return

    if not getattr(self, "dataset_inst", None):
        return

    met_name = self.config_inst.x.met_name
    raw_met_name = self.config_inst.x.raw_met_name

    # list of calibrators to apply (in that order)
    self.calibrators = []

    uncertainty_sources = [] if self.dataset_inst.is_data else self.jec_sources
    jec_cls_name = f"ak4_jec{'_nominal' if uncertainty_sources == [] else ''}"

    jec_cls = jec.derive(
        jec_cls_name,
        cls_dict={
            "uncertainty_sources": uncertainty_sources,
            "met_name": met_name,
            "raw_met_name": raw_met_name,
        },
    )
    self.calibrators.append(jec_cls)

    if self.bjet_regression:
        self.calibrators.append(bjet_regression)

    # run JER only on MC
    if self.dataset_inst.is_mc:
        # version of jer that uses the first random number from deterministic_seeds
        deterministic_jer_cls = jer.derive(
            "deterministic_jer",
            cls_dict={
                "deterministic_seed_index": 0,
                "met_name": met_name,
            },
        )
        self.calibrators.append(deterministic_jer_cls)

    if self.config_inst.x.run == 2:
        # derive met_phi calibrator (currently only for run 2)
        met_phi_cls = met_phi.derive("met_phi", cls_dict={"met_name": met_name})
        self.calibrators.append(met_phi_cls)

    self.uses |= set(self.calibrators)
    self.produces |= set(self.calibrators)


skip_jecunc = jet_base.derive("skip_jecunc", cls_dict=dict(bjet_regression=False))
with_b_reg = jet_base.derive("with_b_reg", cls_dict=dict(bjet_regression=True))
