# coding: utf-8

"""
Selection modules for HH -> bbWW(lnulnu).
"""

from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT, optional_column as optional
from columnflow.selection import Selector, SelectionResult, selector

from hbw.selection.common import masked_sorted_indices, pre_selection, post_selection, configure_selector
from hbw.selection.jet import sl_boosted_jet_selection, vbf_jet_selection
from hbw.production.weights import event_weights_to_normalize
from hbw.selection.cutflow_features import cutflow_features

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


def TetraVec(arr: ak.Array) -> ak.Array:
    TetraVec = ak.zip({"pt": arr.pt, "eta": arr.eta, "phi": arr.phi, "mass": arr.mass},
    with_name="PtEtaPhiMLorentzVector",
    behavior=coffea.nanoevents.methods.vector.behavior)
    return TetraVec


def invariant_mass(events: ak.Array):
    empty_events = ak.zeros_like(events, dtype=np.uint16)[:, 0:0]
    where = ak.num(events, axis=1) == 2
    events_with_2 = ak.where(where, events, empty_events)
    mass = ak.fill_none(ak.firsts((TetraVec(events_with_2[:, :1]) + TetraVec(events_with_2[:, 1:2])).mass), 0)
    mass = ak.nan_to_num(mass, nan=EMPTY_FLOAT)
    return mass


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB", "Jet.jetId"},
    produces={"cutflow.n_jet", "cutflow.n_deepjet_med"},
    exposed=True,
)
def dl_jet_selection(
    self: Selector,
    events: ak.Array,
    lepton_results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # HH -> bbWW(qqlnu) jet selection
    # - require at least 2 jet with pt>20, eta<2.4, b-score>0.3040 (Medium WP)

    # assign local index to all Jets
    events = set_ak_column(events, "local_index", ak.local_index(events.Jet))

    # jets
    jet_mask_loose = (events.Jet.pt > 5) & abs(events.Jet.eta < 2.4)
    jet_mask = (
        (events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4) & (events.Jet.jetId == 6) &
        ak.all(events.Jet.metric_table(lepton_results.x.lepton) > 0.3, axis=2)
    )
    events = set_ak_column(events, "cutflow.n_jet", ak.sum(jet_mask, axis=1))
    jet_sel = events.cutflow.n_jet >= 1
    jet_indices = masked_sorted_indices(jet_mask, events.Jet.pt)

    # b-tagged jets, medium working point
    wp_med = self.config_inst.x.btag_working_points.deepjet.medium
    btag_mask = (jet_mask) & (events.Jet.btagDeepFlavB >= wp_med)
    events = set_ak_column(events, "cutflow.n_deepjet_med", ak.sum(btag_mask, axis=1))
    btag_sel = events.cutflow.n_deepjet_med >= 1

    # define b-jets as the two b-score leading jets, b-score sorted
    bjet_indices = masked_sorted_indices(jet_mask, events.Jet.btagDeepFlavB)[:, :2]

    # define lightjets as all non b-jets, pt-sorted
    b_idx = ak.fill_none(ak.pad_none(bjet_indices, 2), -1)
    lightjet_indices = jet_indices[(jet_indices != b_idx[:, 0]) & (jet_indices != b_idx[:, 1])]

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={"Jet": jet_sel, "Bjet": btag_sel, "Resolved": (jet_sel & btag_sel)},
        objects={
            "Jet": {
                "LooseJet": masked_sorted_indices(jet_mask_loose, events.Jet.pt),
                "Jet": jet_indices,
                "Bjet": bjet_indices,
                "Lightjet": lightjet_indices,
            },
        },
        aux={
            "jet_mask": jet_mask,
            "n_central_jets": ak.num(jet_indices),
        },
    )


@selector(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Electron.charge", "Electron.pdgId",
        "Electron.cutBased",
        optional("Electron.mvaFall17V2Iso_WP80"), optional("Electron.mvaIso_WP90"),
        "Muon.tightId", "Muon.looseId", "Muon.pfRelIso04_all",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "Muon.charge", "Muon.pdgId",
        "Tau.pt", "Tau.eta",
    },
    produces={"m_ll", "channel_id", "Muon.is_tight", "Electron.is_tight"},
    e_pt=None, mu_pt=None,
    ee_trigger=None, mm_trigger=None, emu_trigger=None, mue_trigger=None,
)
def dl_lepton_selection(
        self: Selector,
        events: ak.Array,
        stats: defaultdict,
        **kwargs,
) -> Tuple[ak.Array, SelectionResult]:

    # dummy values to make selection work technically
    events = set_ak_column(events, "Muon.is_tight", ak.ones_like(events.Muon.pt), value_type=bool)
    events = set_ak_column(events, "Electron.is_tight", ak.ones_like(events.Electron.pt), value_type=bool)

    electron = (events.Electron)
    muon = (events.Muon)

    mvaIso_column = "mvaFall17V2Iso_WP80" if "mvaFall17V2Iso_WP80" in electron.fields else "mvaIso_WP90"

    # loose electron mask
    e_mask_loose = (
        (abs(electron.eta) < 2.5) &
        (electron.pt > 15) &
        (electron.cutBased == 4) &
        (electron[mvaIso_column] == 1)
    )
    # loose mask muons
    mu_mask_loose = (
        (abs(muon.eta) < 2.4) &
        (muon.pt > 15) &  # TODO: A lower cut gives problems with muon SFs
        (muon.tightId) &
        (muon.pfRelIso04_all < 0.15)
    )
    # not needed, saves flavour forlepton object
    electron["e_idx"] = ak.local_index(electron, axis=1)
    muon["mu_idx"] = ak.local_index(muon, axis=1)
    electron, muon = ak.broadcast_fields(electron, muon)
    electron = ak.without_parameters(electron)
    muon = ak.without_parameters(muon)
    # create new object: leptons
    leptons = ak.concatenate([muon[mu_mask_loose], electron[e_mask_loose]], axis=-1)
    leptons = leptons[ak.argsort(leptons.pt, axis=-1, ascending=False)]
    fill_with = {
        "pt": -999, "eta": -999, "phi": -999, "charge": -999,
        "pdgId": -999, "mass": -999, "e_idx": -999, "mu_idx": -999,
    }
    leptons = ak.fill_none(ak.pad_none(leptons, 2, axis=-1), fill_with)

    # mumu channel
    mm_mask = (
        (ak.num(leptons.pdgId, axis=-1) == 2) &  # this requires <= 2 leptons (padding: all events have >= 2 leptons)
        (abs(leptons.pdgId[:, 0]) == 13) &
        (abs(leptons.pdgId[:, 1]) == 13) &  # this requires <= 2 leptons
        (leptons.pt[:, 0] > 20) &
        (invariant_mass(leptons) > 12) &
        (invariant_mass(leptons) < 76) &
        (ak.sum(leptons.charge, axis=-1) == 0)  # this vetos same-sign leptons + events with 1 or 3 leptons
    )

    # ee channel
    ee_mask = (
        (ak.num(leptons.pdgId, axis=-1) == 2) &
        (abs(leptons.pdgId[:, 0]) == 11) &
        (abs(leptons.pdgId[:, 1]) == 11) &
        (leptons.pt[:, 0] > 25) &
        (invariant_mass(leptons) > 12) &
        (invariant_mass(leptons) < 76) &
        (ak.sum(leptons.charge, axis=-1) == 0)
    )

    # mixed flavour channel
    em_mask = (
        (ak.num(leptons.pdgId, axis=-1) == 2) &
        (
            ((abs(leptons.pdgId[:, 0]) == 11) & (abs(leptons.pdgId[:, 1]) == 13)) |
            ((abs(leptons.pdgId[:, 0]) == 13) & (abs(leptons.pdgId[:, 1]) == 11))
        ) &
        (leptons.pt[:, 0] > 25) &
        (invariant_mass(leptons) > 12) &
        (invariant_mass(leptons) < 76) &
        (ak.sum(leptons.charge, axis=-1) == 0)
    )

    # channel Id and m_ll as new variable
    empty_events = ak.zeros_like(1 * events.event, dtype=np.uint16)
    channel_id = ak.zeros_like(empty_events)
    channel_id = ak.where(mm_mask, ak.full_like(events.event, 1), channel_id)
    channel_id = ak.where(ee_mask, ak.full_like(events.event, 2), channel_id)
    channel_id = ak.where(em_mask, ak.full_like(events.event, 3), channel_id)
    leptons["channel_id"] = channel_id
    events = set_ak_column(events, "channel_id", channel_id)
    leptons["m_ll"] = channel_id
    events = set_ak_column(events, "m_ll", invariant_mass(leptons))

    # combined lepton mask from all channels
    ll_mask = mm_mask | ee_mask | em_mask

    # Electron and Muon indices corresponding to lepton selection
    empty_indices = empty_events[..., None][..., :0]
    e_indices = ak.where(ll_mask, leptons.e_idx, empty_indices)
    mu_indices = ak.where(ll_mask, leptons.mu_idx, empty_indices)
    e_indices_l = ak.drop_none(e_indices)
    mu_indices_l = ak.drop_none(mu_indices)
    lep_sel = (ak.num(e_indices_l, axis=-1) + ak.num(mu_indices_l, axis=-1)) == 2

    # Implementation of dileptonic triggers
    # dummy mask for triggers
    ones = ak.ones_like(lep_sel)
    false_trigger = empty_events != 0
    mixed_trigger_sel = false_trigger
    mm_trigger_sel = false_trigger
    ee_trigger_sel = false_trigger

    # 2e channel trigger
    if not self.ee_trigger:
        ee_trigger_sel = ones
    else:
        for dl_trigger in self.ee_trigger:
            ee_trigger_sel = ee_trigger_sel | events.HLT[dl_trigger]

    # 2mu channel triggers
    mm_trigger_sel = false_trigger
    if not self.mm_trigger:
        mm_trigger_sel = ones
    else:
        for dl_trigger in self.mm_trigger:
            mm_trigger_sel = mm_trigger_sel | events.HLT[dl_trigger]

    # mixed channel triggers
    if not self.mue_trigger:
        mixed_trigger_sel = ones
    else:
        for dl_trigger in self.mue_trigger:
            mixed_trigger_sel = mixed_trigger_sel | events.HLT[dl_trigger]
    if not self.emu_trigger:
        mixed_trigger_sel = mixed_trigger_sel
    else:
        for dl_trigger in self.emu_trigger:
            mixed_trigger_sel = mixed_trigger_sel | events.HLT[dl_trigger]

    trigger_sel = mm_trigger_sel | ee_trigger_sel | mixed_trigger_sel
    trigger_lep_crosscheck = (trigger_sel & ll_mask)

    # loose indices on electron and muon
    e_indices = masked_sorted_indices(e_mask_loose, electron.pt)
    mu_indices = masked_sorted_indices(mu_mask_loose, muon.pt)

    return events, SelectionResult(
        steps={
            "Lepton": ll_mask,
            "Lep_mm": mm_mask,
            "Lep_ee": ee_mask,
            "Lep_mixed": em_mask,
            "Trigger": trigger_sel, "TriggerAndLep": trigger_lep_crosscheck,
        },
        objects={
            "Electron": {
                "ElectronLoose": e_indices,
                "Electron": e_indices_l,
            },
            "Muon": {
                "MuonLoose": mu_indices,
                "Muon": mu_indices_l,
            },
        },
        aux={
            # save the selected lepton for the duration of the selection
            # multiplication of a coffea particle with 1 yields the lorentz vector
            "lepton": leptons,
        },
    )


@dl_lepton_selection.init
def dl_lepton_selection_init(self: Selector) -> None:
    configure_selector(self)

    year = self.config_inst.campaign.x.year

    # NOTE: the none will not be overwritten later when doing this...
    # self.mu_trigger = self.e_trigger = None

    # Lepton pt thresholds (if not set manually) based on year (1 pt above trigger threshold)
    # When lepton pt thresholds are set manually, don't use any trigger
    # triggers 2e channel
    if not self.ee_trigger:
        self.ee_trigger = {
            2017: ["Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
            2022: [
                "Ele30_WPTight_Gsf",
                "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
            ],
        }[year]
        for trigger in self.ee_trigger:
            self.uses.add(f"HLT.{trigger}")

    # triggers 2mu channel
    if not self.mm_trigger:
        self.mm_trigger = {
            2017: ["Mu17_TrkIsoVVL_Mu8_TrkIsoVVL", "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
                  "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8", "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8"],
            2022: [
                "IsoMu24",
                "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            ],
        }[year]
        for trigger in self.mm_trigger:
            self.uses.add(f"HLT.{trigger}")

    # triggers e mu channel
    if not self.emu_trigger:
        self.emu_trigger = {
            2017: ["Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
                  "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ"],
            2022: [
                "IsoMu24",
                "Ele30_WPTight_Gsf",
                "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",  # TODO: recommentations (unprescaled?)
            ],
        }[year]
        for trigger in self.emu_trigger:
            self.uses.add(f"HLT.{trigger}")

    # triggers mu e channel
    if not self.mue_trigger:
        self.mue_trigger = {
            2017: ["Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
                  "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL"],
            2022: [
                "IsoMu24",
                "Ele30_WPTight_Gsf",
                "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",  # TODO: recommentations (unprescaled?)
            ],
        }[year]
        for trigger in self.mue_trigger:
            self.uses.add(f"HLT.{trigger}")


@selector(
    uses={
        pre_selection, post_selection,
        vbf_jet_selection, sl_boosted_jet_selection,
        dl_jet_selection, dl_lepton_selection,
    },
    produces={
        pre_selection, post_selection,
        vbf_jet_selection, sl_boosted_jet_selection,
        dl_jet_selection, dl_lepton_selection,
    },
    exposed=True,
)
def dl(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # prepare events
    events, results = self[pre_selection](events, stats, **kwargs)

    # lepton selection
    events, lepton_results = self[dl_lepton_selection](events, stats, **kwargs)
    results += lepton_results

    # jet selection
    events, jet_results = self[dl_jet_selection](events, lepton_results, stats, **kwargs)
    results += jet_results

    # boosted selection
    events, boosted_results = self[sl_boosted_jet_selection](events, lepton_results, jet_results, stats, **kwargs)
    results += boosted_results

    # vbf-jet selection
    # events, vbf_jet_results = self[vbf_jet_selection](events, results, stats, **kwargs)
    # results += vbf_jet_results

    results.steps["ResolvedOrBoosted"] = (
        (results.steps.Jet & results.steps.Bjet | results.steps.HbbJet)
    )

    # combined event selection after all steps except b-jet selection
    results.steps["all_but_bjet"] = (
        # NOTE: the boosted selection actually includes a b-jet selection...
        (results.steps.Jet | results.steps.HbbJet_no_bjet) &
        results.steps.Lepton &
        results.steps.Trigger &
        results.steps.TriggerAndLep
    )

    # combined event selection after all steps
    # NOTE: we only apply the b-tagging step when no AK8 Jet is present; if some event with AK8 jet
    #       gets categorized into the resolved category, we might need to cut again on the number of b-jets
    results.steps["all"] = results.event = (
        results.steps.all_but_bjet &
        ((results.steps.Jet & results.steps.Bjet) | results.steps.HbbJet)
    )

    # dummy values to make selection work technically
    results.steps.SR = results.event
    results.steps.Fake = ~results.event

    # build categories
    events, results = self[post_selection](events, results, stats, **kwargs)

    return events, results


@dl.init
def dl_init(self: Selector) -> None:
    # define mapping from selector step to labels used in cutflow plots
    self.config_inst.x.selector_step_labels = self.config_inst.x("selector_step_labels", {})
    self.config_inst.x.selector_step_labels.update({
        # NOTE: many of these labels are too long for the cf.PlotCutflow task
        "Jet": r"$N_{jets}^{AK4} \neq 3$",
        "Bjet": r"$N_{jets}^{BTag} =  2$",
        "Lepton": r"$N_{lepton} =  2$",
        "TriggerAndLep": "Trigger matches Lepton Channel",
        # "Resolved": r"$N_{jets}^{AK4} \geq 3$ and $N_{jets}^{BTag} \geq 1$",
        # "HbbJet": r"$N_{H \rightarrow bb}^{AK8} \geq 1$",
        "VBFJetPair": r"$N_{VBFJetPair}^{AK4} \geq 1$",
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
