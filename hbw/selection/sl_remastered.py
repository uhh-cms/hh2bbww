# coding: utf-8

"""
Selection modules for HH -> bbWW(qqlnu).
"""

from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import
from columnflow.selection import Selector, SelectionResult, selector

from hbw.selection.common import (
    jet_selection, lepton_definition,
    masked_sorted_indices, sl_boosted_jet_selection, vbf_jet_selection,
    pre_selection, post_selection,
)
from hbw.production.weights import event_weights_to_normalize
from hbw.selection.cutflow_features import cutflow_features

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={lepton_definition, "Electron.charge", "Muon.charge"},
    produces={lepton_definition},
    e_pt=None, mu_pt=None, trigger=None,
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

    # select events
    mu_mask_fakeable = lepton_results.x.mu_mask_fakeable
    e_mask_fakeable = lepton_results.x.e_mask_fakeable

    # NOTE: leading lepton pt could be reduced to trigger threshold + 1
    leading_mu_mask = (mu_mask_fakeable) & (events.Muon.cone_pt > self.config_inst.x.mu_pt)
    leading_e_mask = (e_mask_fakeable) & (events.Electron.cone_pt > self.config_inst.x.e_pt)

    # NOTE: we might need pt > 15 for lepton SFs. Needs to be checked in Run 3.
    subleading_mu_mask = (mu_mask_fakeable) & (events.Muon.cone_pt > 15)
    subleading_e_mask = (e_mask_fakeable) & (events.Electron.cone_pt > 15)

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

    lepton_results.steps["DileptonVeto"] = ak.num(lepton, axis=1) <= 1

    lepton_results.steps["Lep_e"] = e_mask = (
        (ak.sum(leading_e_mask, axis=1) == 1) &
        (ak.sum(subleading_mu_mask, axis=1) == 0)
    )
    lepton_results.steps["Lep_mu"] = mu_mask = (
        (ak.sum(leading_mu_mask, axis=1) == 1) &
        (ak.sum(subleading_e_mask, axis=1) == 0)
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
        trigger_mask = ak.any([events.HLT[trigger_column] for trigger_column in trigger_columns], axis=0)
        lepton_results.steps[f"Trigger_{channel}"] = trigger_mask

        # ensure that Lepton channel is in agreement with trigger
        lepton_results.steps[f"TriggerAndLep_{channel}"] = (
            lepton_results.steps[f"Trigger_{channel}"] & lepton_results.steps[f"Lep_{channel}"]
        )

    # combine results of each individual channel
    lepton_results.steps["Trigger"] = ak.any([
        lepton_results.steps[f"Trigger_{channel}"]
        for channel in self.config_inst.x.trigger.keys()
    ], axis=0)

    lepton_results.steps["TriggerAndLep"] = ak.any([
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

    # setup lepton pt and trigger requirements in the config
    # when the lepton selector does not define the values, resort to defaults
    # NOTE: this is not doing what I was intending: this allows me to share the selector info
    # with other tasks, but I want other selectors to be able to change these attributes...
    if year == 2016:
        self.config_inst.x.mu_pt = self.mu_pt or 25
        self.config_inst.x.e_pt = self.e_pt or 27
        self.config_inst.x.trigger = self.trigger or {
            "e": ["Ele27_WPTight_Gsf"],
            "mu": ["IsoMu24"],
        }
    elif year == 2017:
        self.config_inst.x.mu_pt = self.mu_pt or 28
        self.config_inst.x.e_pt = self.e_pt or 36
        self.config_inst.x.trigger = self.trigger or {
            "e": ["Ele35_WPTight_Gsf"],
            "mu": ["IsoMu27"],
        }
    elif year == 2018:
        self.config_inst.x.mu_pt = self.mu_pt or 25
        self.config_inst.x.e_pt = self.e_pt or 33
        self.config_inst.x.trigger = self.trigger or {
            "e": ["Ele32_WPTight_Gsf"],
            "mu": ["IsoMu24"],
        }
    elif year == 2022:
        self.config_inst.x.mu_pt = self.mu_pt or 25
        self.config_inst.x.e_pt = self.e_pt or 31
        self.config_inst.x.trigger = self.trigger or {
            "e": ["Ele30_WPTight_Gsf"],
            "mu": ["IsoMu24"],
        }
    else:
        raise Exception(f"Single lepton trigger not implemented for year {year}")

    # add all required trigger to the uses
    for trigger_columns in self.config_inst.x.trigger.values():
        for column in trigger_columns:
            self.uses.add(f"HLT.{column}")


@selector(
    uses={
        pre_selection, post_selection,
        vbf_jet_selection, sl_boosted_jet_selection,
        jet_selection, sl_lepton_selection,
    },
    produces={
        pre_selection, post_selection,
        vbf_jet_selection, sl_boosted_jet_selection,
        jet_selection, sl_lepton_selection,
    },
    exposed=True,
)
def sl1(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
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

    results.steps["Resolved"] = (results.steps.nJet3 & results.steps.nBjet1)

    results.steps["ResolvedOrBoosted"] = (
        (results.steps.nJet3 & results.steps.nBjet1) | results.steps.HbbJet
    )

    # combined event selection after all steps except b-jet selection
    results.steps["all_but_bjet"] = (
        results.steps.cleanup &
        (results.steps.nJet3 | results.steps.HbbJet_no_bjet) &
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
    results.event = (
        results.steps.all_but_bjet &
        ((results.steps.nJet3 & results.steps.nBjet1) | results.steps.HbbJet)
    )
    results.steps["all"] = results.event

    # build categories
    events, results = self[post_selection](events, results, stats, **kwargs)

    return events, results


@sl1.init
def sl1_init(self: Selector) -> None:
    # define mapping from selector step to labels used in cutflow plots
    self.config_inst.x.selector_step_labels = self.config_inst.x("selector_step_labels", {})
    self.config_inst.x.selector_step_labels.update({
        "Resolved": r"$N_{jets}^{AK4} \geq 3$ and $N_{jets}^{BTag} \geq 1$",
        "ResolvedOrBoosted": (
            r"($N_{jets}^{AK4} \geq 3$ and $N_{jets}^{BTag} \geq 1$) "
            r"or $N_{H \rightarrow bb}^{AK8} \geq 1$"
        ),
    })

    if self.config_inst.x("do_cutflow_features", False):
        self.uses.add(cutflow_features)
        self.produces.add(cutflow_features)

    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return

    self.uses.add(event_weights_to_normalize)
    self.produces.add(event_weights_to_normalize)
