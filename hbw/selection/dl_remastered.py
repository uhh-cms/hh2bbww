# coding: utf-8

"""
Selection modules for HH -> bbWW(lnulnu).
"""

from importlib import import_module
from functools import partial

from collections import defaultdict

import law

from cmsdb.constants import m_z

from columnflow.util import maybe_import, DotDict
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.selection import Selector, SelectionResult, selector

from hbw.selection.common import (
    masked_sorted_indices, pre_selection, get_weights_and_no_sel_mask, post_selection, configure_selector,
)
from hbw.selection.lepton import lepton_definition
from hbw.selection.jet import jet_selection, dl_boosted_jet_selection, vbf_jet_selection
from hbw.selection.trigger import hbw_trigger_selection

np = maybe_import("numpy")
ak = maybe_import("awkward")

# helper functions
set_ak_bool = partial(set_ak_column, value_type=np.bool_)
logger = law.logger.get_logger(__name__)


@selector(
    uses={lepton_definition, "Electron.charge", "Muon.charge"},
    produces={lepton_definition},
)
def dl_lepton_selection(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Lepton Selector for the DL channel. Produces:

    Steps:
    - TripleLeptonVeto
    - Lep_mm, Lep_ee, Lep_emu, Lep_mue
    - Dilepton (logical or of the previous 4)

    Objects:
    - Electron (fakeable Electron + pt > 15)
    - Muon (fakeable Muon + pt > 15)
    """
    # load default lepton definition

    events, lepton_results = self[lepton_definition](events, stats, **kwargs)

    # number of electrons
    lepton_results.steps["nRecoElectron2"] = ak.num(events.Electron) >= 2
    lepton_results.steps["nLooseElectron2"] = events.cutflow.n_loose_electron >= 2
    lepton_results.steps["nFakeableElectron2"] = events.cutflow.n_fakeable_electron >= 2
    lepton_results.steps["nTightElectron2"] = events.cutflow.n_tight_electron >= 2

    # number of muons
    lepton_results.steps["nRecoMuon2"] = ak.num(events.Muon) >= 2
    lepton_results.steps["nLooseMuon2"] = events.cutflow.n_loose_muon >= 2
    lepton_results.steps["nFakeableMuon2"] = events.cutflow.n_fakeable_muon >= 2
    lepton_results.steps["nTightMuon2"] = events.cutflow.n_tight_muon >= 2

    lepton_results.steps["TripleLooseLeptonVeto"] = (
        events.cutflow.n_loose_electron + events.cutflow.n_loose_muon
    ) <= 2
    lepton_results.steps["TripleFakeableLeptonVeto"] = (
        events.cutflow.n_fakeable_electron + events.cutflow.n_fakeable_muon
    ) <= 2
    lepton_results.steps["TripleTightLeptonVeto"] = (
        events.cutflow.n_tight_electron + events.cutflow.n_tight_muon
    ) <= 2

    # select events
    mu_mask_fakeable = lepton_results.x.mu_mask_fakeable
    e_mask_fakeable = lepton_results.x.e_mask_fakeable

    # NOTE: leading lepton pt could be reduced to trigger threshold + 1
    leading_mu_mask = (mu_mask_fakeable) & (events.Muon.pt > self.mu_pt)
    leading_e_mask = (e_mask_fakeable) & (events.Electron.pt > self.ele_pt)

    # NOTE: we might need pt > 15 for lepton SFs. Needs to be checked in Run 3.
    subleading_mu_mask = (mu_mask_fakeable) & (events.Muon.pt > self.mu2_pt)
    subleading_e_mask = (e_mask_fakeable) & (events.Electron.pt > self.ele2_pt)

    # For further analysis after Reduction, we consider all tight leptons with pt > 15 GeV
    lepton_results.objects["Electron"]["Electron"] = masked_sorted_indices(subleading_e_mask, events.Electron.pt)
    lepton_results.objects["Muon"]["Muon"] = masked_sorted_indices(subleading_mu_mask, events.Muon.pt)

    electron = events.Electron[subleading_e_mask]
    muon = events.Muon[subleading_mu_mask]

    # Create a temporary lepton collection
    lepton = ak.concatenate(
        [
            electron * 1,
            muon * 1,
        ],
        axis=1,
    )
    lepton = lepton_results.aux["lepton"] = lepton[ak.argsort(lepton.pt, axis=-1, ascending=False)]

    lepton_results.steps["TripleLeptonVeto"] = ak.num(lepton, axis=1) <= 2
    lepton_results.steps["Charge"] = ak.sum(electron.charge, axis=1) + ak.sum(muon.charge, axis=1) == 0

    dilepton = ak.pad_none(lepton, 2)
    dilepton = dilepton[:, 0] + dilepton[:, 1]
    events = set_ak_column(
        events,
        "mll",
        ak.fill_none(ak.nan_to_none(dilepton.mass), EMPTY_FLOAT),
        value_type=np.float32,
    )
    lepton_results.steps["DiLeptonMass81"] = ak.fill_none(dilepton.mass <= m_z.nominal - 10, False)
    # lepton channel masks
    lepton_results.steps["Lep_mm"] = mm_mask = (
        lepton_results.steps.ll_lowmass_veto &
        (ak.sum(leading_mu_mask, axis=1) >= 1) &
        (ak.sum(subleading_mu_mask, axis=1) >= 2)
    )
    lepton_results.steps["Lep_ee"] = ee_mask = (
        lepton_results.steps.ll_lowmass_veto &
        (ak.sum(leading_e_mask, axis=1) >= 1) &
        (ak.sum(subleading_e_mask, axis=1) >= 2)
    )
    lepton_results.steps["Lep_emu"] = emu_mask = (
        lepton_results.steps.ll_lowmass_veto &
        (ak.sum(leading_e_mask, axis=1) >= 1) &
        (ak.sum(subleading_mu_mask, axis=1) >= 1)
    )
    lepton_results.steps["Lep_mue"] = mue_mask = (
        lepton_results.steps.ll_lowmass_veto &
        (ak.sum(leading_mu_mask, axis=1) >= 1) &
        (ak.sum(subleading_e_mask, axis=1) >= 1)
    )
    lepton_results.steps["Lep_mixed"] = (emu_mask | mue_mask)

    lepton_results.steps["Dilepton"] = (mm_mask | ee_mask | emu_mask | mue_mask)

    # define (but not apply) steps on how to separate between Fake Region and Signal Region
    lepton_results.steps["Fake"] = (
        lepton_results.steps.Dilepton &
        (ak.sum(electron.is_tight, axis=1) + ak.sum(muon.is_tight, axis=1) <= 1)
    )
    lepton_results.steps["SR"] = (
        lepton_results.steps.Dilepton &
        (ak.sum(electron.is_tight, axis=1) + ak.sum(muon.is_tight, axis=1) == 2)
    )

    return events, lepton_results


@dl_lepton_selection.init
def dl_lepton_selection_init(self: Selector) -> None:
    # configuration of defaults
    self.mu_pt = self.config_inst.x("mu_pt", 25)
    self.ele_pt = self.config_inst.x("ele_pt", 25)
    self.mu2_pt = self.config_inst.x("mu2_pt", 15)
    self.ele2_pt = self.config_inst.x("ele2_pt", 15)

    # update selector steps labels
    self.config_inst.x.selector_step_labels = self.config_inst.x("selector_step_labels", {})
    self.config_inst.x.selector_step_labels.update({
        "TripleLooseLeptonVeto": r"$N_{lepton}^{loose} \leq 2$",
        "TripleFakeableLeptonVeto": r"$N_{lepton}^{fakeable} \leq 2$",
        "TripleTightLeptonVeto": r"$N_{lepton}^{tight} \leq 2$",
        "Charge": r"$\sum q_{\ell} = 0$",
        "Dilepton": r"$N_{lepton} = 2$",
        "DiLeptonMass81": r"$m_{\ell\ell} < 81$",
        "Lep_mm": r"$N_{\mu} = 2$ and $N_{e} = 0$",
        "Lep_ee": r"$N_{\mu} = 0$ and $N_{e} = 2$",
        "Lep_emu": r"Leading e, subleading $\mu$",
        "Lep_mue": r"Leading $\mu$, subleading e",
        "Fake": r"$N_{lepton}^{tight} \leq 1$",
        "SR": r"$N_{lepton}^{tight} = 2$",
        "TriggerAndLep": "Trigger+Lep",
    })

    self.produces.add("mll")

    return


from hbw.util import timeit


@selector(
    produces={"steps.*"},
    exposed=True,
    # configurable attributes
    # object selection requirements
    mu_pt=None,
    ele_pt=None,
    mu2_pt=None,
    ele2_pt=None,
    jet_pt=None,
    # function to configure the self.config_inst.x.triggers aux data
    trigger_config_func=lambda self: getattr(import_module("hbw.config.trigger"), "add_triggers")(self.config_inst),
    # jet selection requirements
    n_jet=None,
    n_btag=None,
    version=law.config.get_expanded("analysis", "dl1_version", 0),
)
@timeit
def dl1(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    hists: DotDict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # prepare events
    events, results = self[pre_selection](events, stats, **kwargs)

    # lepton selection
    events, lepton_results = self[dl_lepton_selection](events, stats, **kwargs)
    results += lepton_results

    # trigger selection
    events, trigger_results = self[hbw_trigger_selection](events, lepton_results, stats, **kwargs)
    results += trigger_results

    # jet selection
    events, jet_results = self[jet_selection](events, lepton_results, stats, **kwargs)
    results += jet_results

    # derive event weights and add base mask of all events that are not considered bad to "cleanup" step
    events, results = self[get_weights_and_no_sel_mask](events, results, **kwargs)
    results.steps["cleanup"] = results.steps.cleanup & results.steps["no_sel_mask"]

    # boosted selection
    events, boosted_results = self[dl_boosted_jet_selection](events, lepton_results, jet_results, stats, **kwargs)
    results += boosted_results

    # vbf_jet selection
    events, vbf_jet_results = self[vbf_jet_selection](events, results, stats, **kwargs)
    results += vbf_jet_results

    # NOTE: the bjet step can be customized only for the resolved selection as of now
    jet_step = results.steps[f"nJet{self.n_jet}"] if self.n_jet != 0 else True
    bjet_step = results.steps[f"nBjet{self.n_btag}"] if self.n_btag != 0 else True

    results.steps["Resolved"] = (jet_step & bjet_step)
    results.steps["ResolvedOrBoosted"] = (
        ((jet_step & bjet_step) | results.steps.HbbJet)
    )

    results.steps["all_but_trigger_and_bjet"] = (
        results.steps.cleanup &
        jet_step &
        results.steps.ll_lowmass_veto &
        results.steps.TripleLeptonVeto &
        results.steps.Charge &
        results.steps.Dilepton &
        results.steps.SR  # exactly 2 tight leptons
    )
    # combined event selection after all steps except b-jet selection
    results.steps["all_but_bjet"] = (
        results.steps.all_but_trigger_and_bjet &
        results.steps.data_double_counting &
        results.steps.Trigger &
        results.steps.TriggerAndLep
    )
    # combined event selection after all steps except trigger
    results.steps["all_but_trigger"] = (
        results.steps.all_but_trigger_and_bjet &
        bjet_step
    )

    # combined event selection after all steps
    results.steps["all"] = results.event = (
        results.steps.all_but_bjet &
        bjet_step
    )
    results.steps["all_or_boosted"] = (
        results.steps.all_but_bjet &
        ((jet_step & bjet_step) | results.steps.HbbJet)
    )
    results.steps["all_SR"] = results.event & results.steps.SR
    results.steps["all_Fake"] = results.event & results.steps.Fake

    # build categories
    events, results = self[post_selection](events, results, stats, hists, **kwargs)

    # keep various steps for last-minute selection changes for data/MC debugging
    keep_steps = (
        "all", "all_but_trigger", "all_but_bjet", "all_but_trigger_and_bjet",
        "Trigger", "TriggerAndLep", "data_double_counting",
        "TripleLooseLeptonVeto", "TripleTightLeptonVeto",
        "VetoTau",
    )
    for step in keep_steps:
        events = set_ak_bool(events, f"steps.{step}", results.steps[step])

    return events, results


@dl1.init
def dl1_init(self: Selector) -> None:
    # defaults
    if self.n_jet is None:
        self.n_jet = 1
    if self.n_btag is None:
        self.n_btag = 1

    # configuration of selection parameters
    # apparently, this init only runs after the used selectors, but we can run this init first
    # by only adding the used selectors in the init
    logger.debug("adding selector tag")
    self.config_inst.add_tag("selector_init")
    configure_selector(self)

    # NOTE: since we add these uses so late, init's of these Producers will not run
    # e.g. during Plotting tasks
    self.uses = {
        pre_selection,
        vbf_jet_selection, dl_boosted_jet_selection,
        jet_selection, dl_lepton_selection,
        hbw_trigger_selection,
        get_weights_and_no_sel_mask,
        post_selection,
    }
    self.produces |= {
        pre_selection,
        vbf_jet_selection, dl_boosted_jet_selection,
        jet_selection, dl_lepton_selection,
        hbw_trigger_selection,
        get_weights_and_no_sel_mask,
        post_selection,
    }

    # define mapping from selector step to labels used in cutflow plots
    self.config_inst.x.selector_step_labels = self.config_inst.x("selector_step_labels", {})
    self.config_inst.x.selector_step_labels.update({
        "Resolved": r"$N_{jets}^{AK4} \geq 1$ and $N_{jets}^{BTag} \geq 1$",
        "ResolvedOrBoosted": (
            r"($N_{jets}^{AK4} \geq 3$ and $N_{jets}^{BTag} \geq 1$) "
            r"or $N_{H \rightarrow bb}^{AK8} \geq 1$"
        ),
    })


dl1_no_btag = dl1.derive("dl1_no_btag", cls_dict={"n_btag": 0})
test_dl = dl1.derive("test_dl")


def single_lepton_trigger_config_func(self: Selector) -> ak.Array:
    sl_triggers = [
        "HLT_IsoMu24",
        "HLT_Ele30_WPTight_Gsf",
    ]
    triggers = self.config_inst.x.triggers
    for trig_name in list(triggers.names()):
        if trig_name not in sl_triggers:
            triggers.remove(trig_name)
    # check that all sl_triggers are still there
    for trig in sl_triggers:
        if trig not in triggers:
            raise ValueError(f"Trigger {trig} not found in {self.config_inst.x.triggers}")


dl1_sl_triggers = dl1.derive(
    "dl1_sl_triggers",
    cls_dict={
        "trigger_config_func": single_lepton_trigger_config_func,
        "mu_pt": 25.,
        "ele_pt": 31.,
        "mu2_pt": 15.,
        "ele2_pt": 15.,
    },
)


dl1_mu18 = dl1.derive(
    "dl1_mu18",
    cls_dict={
        "mu_pt": 18.,
        "ele_pt": 25.,
        "mu2_pt": 15.,
        "ele2_pt": 15.,
    },
)

# selectors version for scale factor testing and calculation
dl1_test = dl1.derive("dl1_test", cls_dict={"version": 3})
dl1_no_trigger = dl1.derive("dl1_no_trigger", cls_dict={"version": 1})
dl1_low_lep = dl1.derive("dl1_low_lep", cls_dict={
    "version": 1,
    "mu_pt": 15.,
    "ele_pt": 15.,
    "mu2_pt": 15.,
    "ele2_pt": 15.,
})
