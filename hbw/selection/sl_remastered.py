# coding: utf-8

"""
Selection modules for HH -> bbWW(qqlnu).
"""

from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import
from columnflow.selection import Selector, SelectionResult, selector

from hbw.selection.common import masked_sorted_indices, pre_selection, post_selection, configure_selector
from hbw.selection.lepton import lepton_definition
from hbw.selection.jet import jet_selection, sl_boosted_jet_selection, vbf_jet_selection
from hbw.production.weights import event_weights_to_normalize
from hbw.util import ak_any

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={lepton_definition, "Electron.charge", "Muon.charge"},
    produces={lepton_definition},
)
def sl_lepton_selection(
        self: Selector,
        events: ak.Array,
        stats: defaultdict,
        **kwargs,
) -> Tuple[ak.Array, SelectionResult]:

    events, lepton_results = self[lepton_definition](events, stats, **kwargs)

    # tau veto
    lepton_results.steps["VetoTau"] = events.cutflow.n_veto_tau == 0

    # number of electrons
    lepton_results.steps["nRecoElectron1"] = ak.num(events.Electron) >= 1
    lepton_results.steps["nLooseElectron1"] = events.cutflow.n_loose_electron >= 1
    lepton_results.steps["nFakeableElectron1"] = events.cutflow.n_fakeable_electron >= 1
    lepton_results.steps["nTightElectron1"] = events.cutflow.n_tight_electron >= 1

    # number of muons
    lepton_results.steps["nRecoMuon1"] = ak.num(events.Muon) >= 1
    lepton_results.steps["nLooseMuon1"] = events.cutflow.n_loose_muon >= 1
    lepton_results.steps["nFakeableMuon1"] = events.cutflow.n_fakeable_muon >= 1
    lepton_results.steps["nTightMuon1"] = events.cutflow.n_tight_muon >= 1

    lepton_results.steps["DoubleLooseLeptonVeto"] = (
        events.cutflow.n_loose_electron + events.cutflow.n_loose_muon
    ) <= 1
    lepton_results.steps["DoubleFakeableLeptonVeto"] = (
        events.cutflow.n_fakeable_electron + events.cutflow.n_fakeable_muon
    ) <= 1
    lepton_results.steps["DoubleTightLeptonVeto"] = (
        events.cutflow.n_tight_electron + events.cutflow.n_tight_muon
    ) <= 1

    # select events
    mu_mask_fakeable = lepton_results.x.mu_mask_fakeable
    e_mask_fakeable = lepton_results.x.e_mask_fakeable

    # NOTE: leading lepton pt could be reduced to trigger threshold + 1
    leading_mu_mask = (mu_mask_fakeable) & (events.Muon.pt > self.config_inst.x.mu_pt)
    leading_e_mask = (e_mask_fakeable) & (events.Electron.pt > self.config_inst.x.ele_pt)

    # NOTE: we might need pt > 15 for lepton SFs. Needs to be checked in Run 3.
    veto_mu_mask = (mu_mask_fakeable) & (events.Muon.pt > self.config_inst.x.mu2_pt)
    veto_e_mask = (e_mask_fakeable) & (events.Electron.pt > self.config_inst.x.ele2_pt)

    # For further analysis after Reduction, we consider all tight leptons with pt > 15 GeV
    lepton_results.objects["Electron"]["Electron"] = masked_sorted_indices(veto_e_mask, events.Electron.pt)
    lepton_results.objects["Muon"]["Muon"] = masked_sorted_indices(veto_mu_mask, events.Muon.pt)
    electron = events.Electron[veto_e_mask]
    muon = events.Muon[veto_mu_mask]

    # Create a temporary lepton collection
    lepton = ak.concatenate(
        [
            electron * 1,
            muon * 1,
        ],
        axis=1,
    )
    lepton = lepton_results.aux["lepton"] = lepton[ak.argsort(lepton.pt, axis=-1, ascending=False)]

    # reject all events with > 1 lepton: this step ensures orthogonality to the DL channel.
    # NOTE: we could tighten this cut (veto loose lepton or reduce pt threshold)
    lepton_results.steps["DileptonVeto"] = ak.num(lepton, axis=1) <= 1

    lepton_results.steps["Lep_e"] = e_mask = (
        (ak.sum(leading_e_mask, axis=1) == 1) &
        (ak.sum(veto_mu_mask, axis=1) == 0)
    )
    lepton_results.steps["Lep_mu"] = mu_mask = (
        (ak.sum(leading_mu_mask, axis=1) == 1) &
        (ak.sum(veto_e_mask, axis=1) == 0)
    )

    lepton_results.steps["Lepton"] = (e_mask | mu_mask)

    lepton_results.steps["Fake"] = (
        lepton_results.steps.Lepton &
        (ak.sum(electron.is_tight, axis=1) + ak.sum(muon.is_tight, axis=1) == 0)
    )
    lepton_results.steps["SR"] = (
        lepton_results.steps.Lepton &
        (ak.sum(electron.is_tight, axis=1) + ak.sum(muon.is_tight, axis=1) == 1)
    )

    for channel, trigger_columns in self.config_inst.x.trigger.items():
        # apply the "or" of all triggers of this channel
        trigger_mask = ak_any([events.HLT[trigger_column] for trigger_column in trigger_columns], axis=0)
        lepton_results.steps[f"Trigger_{channel}"] = trigger_mask

        # ensure that Lepton channel is in agreement with trigger
        lepton_results.steps[f"TriggerAndLep_{channel}"] = (
            lepton_results.steps[f"Trigger_{channel}"] & lepton_results.steps[f"Lep_{channel}"]
        )

    # combine results of each individual channel
    lepton_results.steps["Trigger"] = ak_any([
        lepton_results.steps[f"Trigger_{channel}"]
        for channel in self.config_inst.x.trigger.keys()
    ], axis=0)

    lepton_results.steps["TriggerAndLep"] = ak_any([
        lepton_results.steps[f"TriggerAndLep_{channel}"]
        for channel in self.config_inst.x.trigger.keys()
    ], axis=0)

    return events, lepton_results


@sl_lepton_selection.init
# @call_once_on_instance()
def sl_lepton_selection_init(self: Selector) -> None:
    # update selector steps labels
    self.config_inst.x.selector_step_labels = self.config_inst.x("selector_step_labels", {})
    self.config_inst.x.selector_step_labels.update({
        "DoubleLooseLeptonVeto": r"$N_{lepton}^{loose} \leq 1$",
        "DoubleFakeableLeptonVeto": r"$N_{lepton}^{fakeable} \leq 1$",
        "DoubleTightLeptonVeto": r"$N_{lepton}^{tight} \leq 1$",
        "DileptonVeto": r"$N_{lepton} \leq 1$",
        "Lepton": r"$N_{lepton} = 1$",
        "Lep_e": r"$N_{e} = 1$ and $N_{\mu} = 0$",
        "Lep_mu": r"$N_{\mu} = 1$ and $N_{e} = 0$",
        "Fake": r"$N_{lepton}^{tight} = 0$",
        "SR": r"$N_{lepton}^{tight} = 1$",
        "TriggerAndLep": "Trigger matches Lepton Channel",
        "VetoTau": r"$N_{\tau}^{veto} = 0$",
    })

    year = self.config_inst.campaign.x.year

    #
    # if not already done, setup lepton pt and trigger requirements in the config
    #

    # for vetoing additional leptons; should be in sync with DL channel
    self.config_inst.x.mu2_pt = self.config_inst.x("mu2_pt", 15)
    self.config_inst.x.ele2_pt = self.config_inst.x("ele2_pt", 15)

    if year == 2016:
        self.config_inst.x.mu_pt = self.config_inst.x("mu_pt", 25)
        self.config_inst.x.ele_pt = self.config_inst.x("ele_pt", 27)
        self.config_inst.x.trigger = self.config_inst.x("trigger", {
            "e": ["Ele27_WPTight_Gsf"],
            "mu": ["IsoMu24"],
        })
    elif year == 2017:
        self.config_inst.x.mu_pt = self.config_inst.x("mu_pt", 28)
        self.config_inst.x.ele_pt = self.config_inst.x("ele_pt", 36)
        self.config_inst.x.trigger = self.config_inst.x("trigger", {
            "e": ["Ele35_WPTight_Gsf"],
            "mu": ["IsoMu27"],
        })
    elif year == 2018:
        self.config_inst.x.mu_pt = self.config_inst.x("mu_pt", 25)
        self.config_inst.x.ele_pt = self.config_inst.x("ele_pt", 33)
        self.config_inst.x.trigger = self.config_inst.x("trigger", {
            "e": ["Ele32_WPTight_Gsf"],
            "mu": ["IsoMu24"],
        })
    elif year == 2022:
        self.config_inst.x.mu_pt = self.config_inst.x("mu_pt", 25)
        self.config_inst.x.ele_pt = self.config_inst.x("ele_pt", 31)
        self.config_inst.x.trigger = self.config_inst.x("trigger", {
            "e": ["Ele30_WPTight_Gsf"],
            "mu": ["IsoMu24"],
        })
    else:
        raise Exception(f"Single lepton trigger not implemented for year {year}")

    # add all required trigger to the uses
    for trigger_columns in self.config_inst.x.trigger.values():
        for column in trigger_columns:
            self.uses.add(f"HLT.{column}")


@selector(
    exposed=True,
    # configurable attributes
    mu_pt=None,
    ele_pt=None,
    mu2_pt=None,
    ele2_pt=None,
    trigger=None,
    jet_pt=None,
    n_jet=None,
    b_tagger=None,
    btag_wp=None,
    n_btag=None,
    version=1,
)
def sl1(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    hists: dict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # prepare events
    events, results = self[pre_selection](events, stats, **kwargs)

    # lepton selection
    events, lepton_results = self[sl_lepton_selection](events, stats, **kwargs)
    results += lepton_results

    # jet selection
    events, jet_results = self[jet_selection](events, lepton_results, stats, **kwargs)
    results += jet_results

    # boosted selection
    events, boosted_results = self[sl_boosted_jet_selection](events, lepton_results, jet_results, stats, **kwargs)
    results += boosted_results

    # vbf-jet selection
    events, vbf_jet_results = self[vbf_jet_selection](events, results, stats, **kwargs)
    results += vbf_jet_results

    # NOTE: the bjet step can be customized only for the resolved selection as of now
    jet_step = results.steps[f"nJet{self.n_jet}"] if self.n_jet != 0 else True
    bjet_step = results.steps[f"nBjet{self.n_btag}"] if self.n_btag != 0 else True

    results.steps["Resolved"] = (jet_step & bjet_step)

    results.steps["ResolvedOrBoosted"] = (
        (jet_step & bjet_step) | results.steps.HbbJet
    )

    # combined event selection after all steps except b-jet selection
    results.steps["all_but_bjet"] = (
        results.steps.cleanup &
        (jet_step | results.steps.HbbJet_no_bjet) &
        results.steps.ll_lowmass_veto &
        results.steps.ll_zmass_veto &
        results.steps.DileptonVeto &
        results.steps.Lepton &
        results.steps.VetoTau &
        results.steps.Trigger &
        results.steps.TriggerAndLep
    )

    # combined event selection after all steps
    # NOTE: we only apply the b-tagging step when no AK8 Jet is present; if some event with AK8 jet
    #       gets categorized into the resolved category, we might need to cut again on the number of b-jets
    results.steps["all"] = results.event = (
        results.steps.all_but_bjet &
        ((jet_step & bjet_step) | results.steps.HbbJet)
    )
    results.steps["all_SR"] = results.event & results.steps.SR
    results.steps["all_Fake"] = results.event & results.steps.Fake

    # build categories
    events, results = self[post_selection](events, results, stats, hists, **kwargs)

    return events, results


@sl1.init
def sl1_init(self: Selector) -> None:
    # defaults
    if self.n_jet is None:
        self.n_jet = 3
    if self.n_btag is None:
        self.n_btag = 1

    # configuration of selection parameters
    # apparently, this init only runs after the used selectors, but we can run this init first
    # by only adding the used selectors in the init
    configure_selector(self)

    self.uses = {
        pre_selection,
        vbf_jet_selection, sl_boosted_jet_selection,
        jet_selection, sl_lepton_selection,
        post_selection,
    }
    self.produces = {
        pre_selection,
        vbf_jet_selection, sl_boosted_jet_selection,
        jet_selection, sl_lepton_selection,
        post_selection,
    }

    # define mapping from selector step to labels used in cutflow plots
    self.config_inst.x.selector_step_labels = self.config_inst.x("selector_step_labels", {})
    self.config_inst.x.selector_step_labels.update({
        "Resolved": r"$N_{jets}^{AK4} \geq 3$ and $N_{jets}^{BTag} \geq 1$",
        "ResolvedOrBoosted": (
            r"($N_{jets}^{AK4} \geq 3$ and $N_{jets}^{BTag} \geq 1$) "
            r"or $N_{H \rightarrow bb}^{AK8} \geq 1$"
        ),
    })

    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return

    self.uses.add(event_weights_to_normalize)
    self.produces.add(event_weights_to_normalize)


sl1_no_btag = sl1.derive("sl1_no_btag", cls_dict={"n_btag": 0, "b_tagger": "deepjet"})
