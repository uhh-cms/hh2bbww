# coding: utf-8

"""
Exemplary reduction methods that can run on-top of columnflow's default reduction.
"""

import law

from columnflow.reduction import Reducer, reducer
from columnflow.reduction.default import cf_default
from columnflow.util import maybe_import

from hbw.util import IF_TOP, IF_VJETS, IF_DY
from columnflow.production.cms.top_pt_weight import gen_parton_top
from hbw.production.gen_v import gen_v_boson
from hbw.production.jets import njet_for_recoil
from columnflow.production.cms.dy import gen_dilepton, recoil_corrected_met
from hbw.production.gen_hbv_decay import gen_hbv_decay
from columnflow.production.util import attach_coffea_behavior


ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


recoil_corrected_met.njet_column = "njet_for_recoil"


@reducer(
    uses={
        attach_coffea_behavior,
        cf_default,
        IF_TOP(gen_parton_top),
        IF_VJETS(gen_v_boson),
        IF_DY(gen_dilepton, recoil_corrected_met, njet_for_recoil),
        gen_hbv_decay,
    },
    produces={
        attach_coffea_behavior,
        cf_default,
        IF_TOP(gen_parton_top),
        IF_VJETS(gen_v_boson),
        IF_DY(gen_dilepton, recoil_corrected_met, njet_for_recoil),
        gen_hbv_decay,
    },
)
def default(self: Reducer, events: ak.Array, selection: ak.Array, task: law.Task, **kwargs) -> ak.Array:
    # run cf's default reduction which handles event selection and collection creation
    events = self[cf_default](events, selection, task, **kwargs)

    if selector_config := self.config_inst.x("selector_config"):
        requested_jet_pt = selector_config.get("jet_pt", 25.)
        min_jet_pt = ak.min(events.Jet.pt)
        if min_jet_pt is not None and min_jet_pt < requested_jet_pt:
            raise ValueError(
                f"Minimum jet pt found {ak.min(events.Jet.pt)} is smaller than requested {requested_jet_pt}"
                f"for {self.dataset_inst.name} with config {self.config_inst.name}, shift {task.shift}."
                "Please check that your selection is configured correctly.",
            )
    # compute and store additional columns after the default reduction
    # (so only on a subset of the events and objects which might be computationally lighter)

    # compute gen information that will later be needed for top pt reweighting
    if self.has_dep(gen_parton_top):
        events = self[gen_parton_top](events, **kwargs)

    # compute gen information that will later be needed for vector boson pt reweighting
    if self.has_dep(gen_v_boson):
        events = self[gen_v_boson](events, **kwargs)
    if self.has_dep(gen_dilepton):
        events = self[gen_dilepton](events, **kwargs)
    if self.has_dep(recoil_corrected_met):
        events = self[njet_for_recoil](events, **kwargs)
        events = self[recoil_corrected_met](events, **kwargs)

    if self.has_dep(gen_hbv_decay):
        events = self[attach_coffea_behavior](events, **kwargs)
        events = self[gen_hbv_decay](events, **kwargs)

    return events


@default.init
def default_init(self: Reducer) -> None:
    """
    Initialize the default reducer.
    """
    # Add shift dependencies
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag(("jec", "jer"))
    }
