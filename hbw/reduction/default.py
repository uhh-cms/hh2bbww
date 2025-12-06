# coding: utf-8

"""
Exemplary reduction methods that can run on-top of columnflow's default reduction.
"""

import law

from columnflow.reduction import Reducer, reducer
from columnflow.reduction.default import cf_default
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT

from hbw.util import IF_TOP, IF_VJETS, IF_DY
from hbw.production.top_pt_theory import gen_parton_top
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
        IF_DY(gen_dilepton),
        # IF_DY(gen_dilepton, recoil_corrected_met, njet_for_recoil),
        gen_hbv_decay,
    },
    produces={
        attach_coffea_behavior,
        cf_default,
        IF_TOP(gen_parton_top),
        IF_VJETS(gen_v_boson),
        IF_DY(gen_dilepton),
        # IF_DY(gen_dilepton, recoil_corrected_met, njet_for_recoil),
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


test_default = default.derive("test_default")
triggersf = default.derive("triggersf")


@triggersf.init
def triggersf_init(self: Reducer) -> None:
    cfg = self.config_inst

    # prevent multiple initializations
    flag = f"reducer_init_done_{self.cls_name}"
    if cfg.has_tag(flag):
        return
    cfg.add_tag(flag)

    # add config entries needed already during the reduction
    # TODO: At some point this should probably time dependent as well
    cfg.x.dl_orthogonal_trigger = "PFMETNoMu120_PFMHTNoMu120_IDTight"
    cfg.x.dl_orthogonal_trigger2 = "PFMET120_PFMHT120_IDTight"
    cfg.x.hlt_L1_seeds = {
        "PFMETNoMu120_PFMHTNoMu120_IDTight": [
            "ETMHF90",
            "ETMHF100",
            "ETMHF110",
            "ETMHF120",
            "ETMHF130",
            "ETMHF140",
            "ETMHF150",
            "ETM150",
            "ETMHF90_SingleJet60er2p5_dPhi_Min2p1",
            "ETMHF90_SingleJet60er2p5_dPhi_Min2p6",
        ],
        "PFMET120_PFMHT120_IDTight": [
            "ETMHF90",
            "ETMHF100",
            "ETMHF110",
            "ETMHF120",
            "ETMHF130",
            "ETMHF140",
            "ETMHF150",
            "ETM150",
            "ETMHF90_SingleJet60er2p5_dPhi_Min2p1",
            "ETMHF90_SingleJet60er2p5_dPhi_Min2p6",
        ],
    }

    # set default hist producer
    self.config_inst.x.default_hist_producer = "default"

    # add combined process with ttbar and drell-yan
    cfg.add_process(cfg.x.procs.n.sf_bkg_reduced)

    # Change variables
    cfg.add_variable(
        name="trg_npvs",
        expression=lambda events: events.PV.npvs * 1.0,
        aux={
            "inputs": {"PV.npvs"},
        },
        binning=(81, 0, 81),
        x_title=r"$\text{N}_{\text{PV}}$",
        # discrete_x=True,
    )
    cfg.add_variable(
        name="trg_n_jet",
        expression=lambda events: ak.num(events.Jet["pt"], axis=1),
        aux={"inputs": {"Jet.pt"}},
        binning=(9, -0.5, 8.5),
        x_title="Number of jets",
        discrete_x=False,
    )
    # change lepton pt binning
    cfg.add_variable(
        name="trg_lepton0_pt",
        expression=lambda events: events.Lepton[:, 0].pt,
        aux=dict(
            inputs={"{Electron,Muon}.{pt,eta,phi,mass}"},
        ),
        binning=(400, 0., 400.),
        unit="GeV",
        null_value=EMPTY_FLOAT,
        x_title=r"Leading lepton $p_{{T}}$",
    )
    cfg.add_variable(
        name="trg_lepton1_pt",
        expression=lambda events: events.Lepton[:, 1].pt,
        aux=dict(
            inputs={"{Electron,Muon}.{pt,eta,phi,mass}"},
        ),
        binning=(400, 0., 400.),
        unit="GeV",
        null_value=EMPTY_FLOAT,
        x_title=r"Subleading lepton $p_{{T}}$",
    )
    cfg.add_variable(
        name="sf_lepton0_pt",
        expression=lambda events: events.Lepton[:, 0].pt,
        aux=dict(
            inputs={"{Electron,Muon}.{pt,eta,phi,mass}"},
        ),
        binning=[0., 15.] + [i for i in range(16, 76)] + [80., 90., 100., 110., 120., 150., 175., 200., 240., 400.],
        unit="GeV",
        null_value=EMPTY_FLOAT,
        x_title=r"Leading lepton $p_{{T}}$",
    )
    cfg.add_variable(
        name="sf_lepton1_pt",
        expression=lambda events: events.Lepton[:, 1].pt,
        aux=dict(
            inputs={"{Electron,Muon}.{pt,eta,phi,mass}"},
        ),
        binning=[0., 15.] + [i for i in range(16, 66)] + [100., 110., 120., 150., 175., 200., 240., 400.],
        unit="GeV",
        null_value=EMPTY_FLOAT,
        x_title=r"Subleading lepton $p_{{T}}$",
    )
    cfg.add_variable(
        name="sf_npvs",
        expression=lambda events: events.PV.npvs * 1.0,
        aux={
            "inputs": {"PV.npvs"},
        },
        binning=[0., 30.] + [i for i in range(31, 41)] + [50., 81.],
        x_title=r"$\text{N}_{\text{PV}}$",
    )
    # add trigger ids as variables
    cfg.add_variable(
        name="trigger_ids",  # these are the trigger IDs saved during the selection
        aux={"axis_type": "intcat"},
        x_title="Trigger IDs",
    )
    # trigger ids für scale factors
    cfg.add_variable(
        name="trig_ids",  # these are produced in the trigger_prod producer when building different combinations
        aux={"axis_type": "strcat"},
        x_title="Trigger IDs for scale factors",
    )


@triggersf.post_init
def triggersf_post_init(self: Reducer, task: law.Task, **kwargs) -> None:
    if task.selector_steps:
        raise Exception("Selector steps are not supported in triggersf reducer")

    # the updates to selector_steps and used columns are only necessary if the task invokes the reducer
    if not task.invokes_reducer:
        return

    task.selector_steps = ("all_but_trigger",)

    triggersf_required_columns = {
        f"HLT.{self.config_inst.x.dl_orthogonal_trigger}",
        *{
            f"L1.{seed}"
            for seed in self.config_inst.x.hlt_L1_seeds[self.config_inst.x.dl_orthogonal_trigger]
        },
        f"HLT.{self.config_inst.x.dl_orthogonal_trigger2}",
        *{
            f"L1.{seed}"
            for seed in self.config_inst.x.hlt_L1_seeds[self.config_inst.x.dl_orthogonal_trigger2]
        },
    }
    self.uses.update(triggersf_required_columns)
    self.produces.update(triggersf_required_columns)


test_triggersf = triggersf.derive("test_triggersf")
