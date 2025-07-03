# coding: utf-8

"""
Selection methods defining categories based on selection step results.
"""

from __future__ import annotations

import law

from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer
from columnflow.selection import SelectionResult
from columnflow.columnar_util import has_ak_column, optional_column

from hbw.util import MET_COLUMN, BTAG_COLUMN

np = maybe_import("numpy")
ak = maybe_import("awkward")


@categorizer(uses={"event"})
def catid_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = ak.ones_like(events.event) > 0
    return events, mask


@categorizer(uses={catid_incl})
def catid_never(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events, mask = self[catid_incl](events, **kwargs)
    return events, ~mask


#
# Categorizers based on gen info
#


@categorizer(
    uses=optional_column("HardGenPart.pdgId", "GenPart.pdgId", "GenPart.statusFlags"),
    gp_dict={},  # dict with (tuple of) pdgId + number of required prompt particles with this pdgId
    ignore_charge=True,
    _operator="eq",
)
def catid_n_gen_particles(
    self: Categorizer, events: ak.Array, results: SelectionResult | None = None, **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """ Categorizer to select events with a certain number of prompt gen particles """

    # possible options to compare number of particles with required number of particles: ==, >=, >, <=, <
    assert self._operator in ("eq", "ge", "gt", "le", "lt")

    # start with true mask
    mask = np.ones(len(events), dtype=bool)
    if self.dataset_inst.is_data:
        # for data, always return true mask
        return events, mask

    if has_ak_column(events, "HardGenPart.pdgId"):
        gp_id = events.HardGenPart.pdgId
    else:
        try:
            # try to get gp_id column via SelectionResult
            gp_id = events.GenPart.pdgId[results.objects.GenPart.HardGenPart]
        except AttributeError:
            # try to select hard gen particles via status flags
            gp_id = events.GenPart.pdgId[events.GenPart.hasFlags("isHardProcess")]

    if self.ignore_charge:
        gp_id = abs(gp_id)

    for pdgIds, required_n_particles in self.gp_dict.items():
        # make sure that 'pdgIds' is a tuple
        pdgIds = law.util.make_tuple(pdgIds)

        # get number of gen particles with requested pdgIds for each event
        n_particles = sum([ak.sum(gp_id == pdgId, axis=1) for pdgId in pdgIds])

        # compare number of gen particles with required number of particles with requested operator
        this_mask = getattr(n_particles, f"__{self._operator}__")(required_n_particles)
        mask = mask & this_mask

    return events, mask


catid_gen_0lep = catid_n_gen_particles.derive("catid_gen_0lep", cls_dict={"gp_dict": {11: 0, 13: 0, 15: 0}})
catid_gen_1e = catid_n_gen_particles.derive("catid_gen_1e", cls_dict={"gp_dict": {11: 1, 13: 0, 15: 0}})
catid_gen_1mu = catid_n_gen_particles.derive("catid_gen_1mu", cls_dict={"gp_dict": {11: 0, 13: 1, 15: 0}})
catid_gen_1tau = catid_n_gen_particles.derive("catid_gen_1tau", cls_dict={"gp_dict": {11: 0, 13: 0, 15: 1}})
# catid_gen_2e = catid_n_gen_particles.derive("catid_gen_2e", cls_dict={"gp_dict": {11: 2, 13: 0, 15: 0}})
# catid_gen_2mu = catid_n_gen_particles.derive("catid_gen_2mu", cls_dict={"gp_dict": {11: 0, 13: 2, 15: 0}})
# catid_gen_2tau = catid_n_gen_particles.derive("catid_gen_2tau", cls_dict={"gp_dict": {11: 0, 13: 0, 15: 2}})
# catid_gen_emu = catid_n_gen_particles.derive("catid_gen_emu", cls_dict={"gp_dict": {11: 1, 13: 1, 15: 0}})
# catid_gen_etau = catid_n_gen_particles.derive("catid_gen_etau", cls_dict={"gp_dict": {11: 1, 13: 0, 15: 1}})
# catid_gen_mutau = catid_n_gen_particles.derive("catid_gen_mutau", cls_dict={"gp_dict": {11: 0, 13: 1, 15: 1}})
catid_geq_2_gen_leptons = catid_n_gen_particles.derive(
    "catid_geq_2_gen_leptons", cls_dict={"gp_dict": {(11, 13, 15): 2}, "_operator": "ge"},
)


#
# Categorizers based on lepton multiplicity
#

@categorizer(
    uses={"{Muon,Electron}.pt"},
    n_muon=0,
    n_electron=0,
)
def catid_lep(
    self: Categorizer, events: ak.Array, results: SelectionResult | None = None, **kwargs,
) -> tuple[ak.Array, ak.Array]:
    if results:
        electron = events.Electron[results.objects.Electron.Electron]
        muon = events.Muon[results.objects.Muon.Muon]
    else:
        electron = events.Electron
        muon = events.Muon

    mask = (
        (ak.sum(electron["pt"] > 0, axis=-1) == self.n_electron) &
        (ak.sum(muon["pt"] > 0, axis=-1) == self.n_muon)
    )
    return events, mask


catid_1e = catid_lep.derive("catid_1e", cls_dict={"n_electron": 1, "n_muon": 0})
catid_1mu = catid_lep.derive("catid_1mu", cls_dict={"n_electron": 0, "n_muon": 1})
catid_2e = catid_lep.derive("catid_2e", cls_dict={"n_electron": 2, "n_muon": 0})
catid_2mu = catid_lep.derive("catid_2mu", cls_dict={"n_electron": 0, "n_muon": 2})
catid_emu = catid_lep.derive("catid_emu", cls_dict={"n_electron": 1, "n_muon": 1})


@categorizer(
    uses={"{Muon,Electron}.pt"},
)
def catid_ge3lep(
    self: Categorizer, events: ak.Array, results: SelectionResult | None = None, **kwargs,
) -> tuple[ak.Array, ak.Array]:
    if results:
        electron = events.Electron[results.objects.Electron.Electron]
        muon = events.Muon[results.objects.Muon.Muon]
    else:
        electron = events.Electron
        muon = events.Muon

    mask = ak.sum(electron["pt"] > 0, axis=-1) + ak.sum(muon["pt"] > 0, axis=-1) >= 3
    return events, mask


#
# Categorizer for ABCD (either during cf.SelectEvents or cf.ProduceColumns)
#


@categorizer(uses={"Electron.is_tight", "Muon.is_tight"})
def catid_sr(
    self: Categorizer,
    events: ak.Array,
    results: SelectionResult | None = None,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    if results:
        return events, results.steps.SR

    n_lep_tight = ak.sum(events.Electron.is_tight, axis=1) + ak.sum(events.Muon.is_tight, axis=1)
    n_lep_req = 1 if self.config_inst.has_tag("is_sl") else 2
    mask = (n_lep_tight == n_lep_req)
    return events, mask


@categorizer(uses={"Electron.is_tight", "Muon.is_tight"})
def catid_fake(
    self: Categorizer,
    events: ak.Array,
    results: SelectionResult | None = None,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    if results:
        return events, results.steps.Fake

    n_lep_tight = ak.sum(events.Electron.is_tight, axis=1) + ak.sum(events.Muon.is_tight, axis=1)
    n_lep_req = 1 if self.config_inst.has_tag("is_sl") else 2
    mask = (n_lep_tight < n_lep_req)
    return events, mask


@categorizer(uses={MET_COLUMN("pt")})
def catid_highmet(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = events[self.config_inst.x.met_name]["pt"] >= 20
    return events, mask


@categorizer(uses={MET_COLUMN("pt")})
def catid_lowmet(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = events[self.config_inst.x.met_name]["pt"] < 20
    return events, mask

#
# Categorizer for mll categories
#


@categorizer(uses={"mll"})
def catid_mll_low(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.mll < 81)
    return events, mask


@categorizer(uses={"mll"})
def catid_cr(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.mll >= 81)
    return events, mask


@categorizer(uses={"mll"})
def catid_mll_z(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.mll >= 81) & (events.mll < 101)
    return events, mask


@categorizer(uses={"mll"})
def catid_mll_high(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (events.mll >= 101)
    return events, mask


#
# Jet categorization
#

@categorizer(uses={"Jet.pt", "FatJet.particleNet_XbbVsQCD"})
def catid_boosted(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Categorization of events in the boosted category: presence of at least 1 AK8 jet candidate
    fulfilling medium WP of PNetHbb
    """
    xbb_btag_wp_score = self.config_inst.x.xbb_btag_wp_score
    mask = (ak.sum(events.FatJet.particleNet_XbbVsQCD > xbb_btag_wp_score, axis=-1) >= 1)
    return events, mask


@categorizer(uses={"Jet.pt", "HbbJet.pt"})
def catid_resolved(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Categorization of events in the resolved category: presence of no AK8 jet candidate
    for the H->bb decay
    """
    xbb_btag_wp_score = self.config_inst.x.xbb_btag_wp_score
    mask = (ak.sum(events.FatJet.particleNet_XbbVsQCD > xbb_btag_wp_score, axis=-1) == 0)
    return events, mask


@categorizer(
    uses={"Jet.pt"},
    n_jet=2,
)
def catid_njet2(
    self: Categorizer, events: ak.Array, results: SelectionResult | None = None, **kwargs,
) -> tuple[ak.Array, ak.Array]:
    if results:
        return events, results.steps.nJet1
    mask = ak.num(events.Jet["pt"], axis=-1) >= self.n_jet
    return events, mask


@categorizer(uses={BTAG_COLUMN("Jet")})
def catid_le1b_loose(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    b_tagger = self.config_inst.x.b_tagger
    btag_column = self.config_inst.x.btag_column
    btag_wp_score_loose = self.config_inst.x.btag_working_points[b_tagger]["loose"]
    n_deepjet_loose = ak.sum(events.Jet[btag_column] >= btag_wp_score_loose, axis=-1)
    mask = (n_deepjet_loose <= 1)
    return events, mask


@categorizer(uses={BTAG_COLUMN("Jet")})
def catid_ge2b_loose(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    b_tagger = self.config_inst.x.b_tagger
    btag_column = self.config_inst.x.btag_column
    btag_wp_score_loose = self.config_inst.x.btag_working_points[b_tagger]["loose"]
    n_deepjet_loose = ak.sum(events.Jet[btag_column] >= btag_wp_score_loose, axis=-1)
    mask = (n_deepjet_loose >= 2)
    return events, mask


@categorizer(uses={BTAG_COLUMN("Jet")})
def catid_1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    btag_column = self.config_inst.x.btag_column
    btag_wp_score = self.config_inst.x.btag_wp_score
    n_deepjet = ak.sum(events.Jet[btag_column] >= btag_wp_score, axis=-1)
    mask = (n_deepjet <= 1)
    return events, mask




@categorizer(uses={BTAG_COLUMN("Jet")})
def catid_2b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    btag_column = self.config_inst.x.btag_column
    btag_wp_score = self.config_inst.x.btag_wp_score
    n_deepjet = ak.sum(events.Jet[btag_column] >= btag_wp_score, axis=-1)
    mask = (n_deepjet >= 2)
    return events, mask


#
# DNN categorizer
#

# TODO: not hard-coded -> use config?
ml_processes = [
    "signal_ggf", "signal_ggf2", "signal_vbf", "signal_vbf2",
    "signal_ggf4", "signal_ggf5", "signal_vbf4", "signal_vbf5",
    "hh_ggf_hbb_hvv_kl1_kt1", "hh_vbf_hbb_hvv_kv1_k2v1_kl1",
    "hh_ggf_hbb_hvvqqlnu_kl1_kt1", "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1",
    "hh_ggf_hbb_hvv2l2nu_kl1_kt1", "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1",
    "tt", "st", "w_lnu", "dy", "v_lep", "h", "qcd",
    "dy_m50toinf", "tt_dl", "st_tchannel_t",
    "bkg_binary", "sig_ggf_binary", "sig_vbf_binary",
    "sig_ggf", "sig_vbf",
]
for proc in ml_processes:
    @categorizer(
        uses=set(f"mlscore.{proc1}" for proc1 in ml_processes),
        cls_name=f"catid_ml_{proc}",
        proc_col_name=f"{proc}",
        # skip check because we don't know which ML processes were used from the MLModel
        check_used_columns=False,
    )
    def dnn_mask(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
        """
        dynamically built Categorizer that categorizes events based on dnn scores
        """
        # start with true mask
        outp_mask = np.ones(len(events), dtype=bool)
        for col_name in events.mlscore.fields:
            # check for each mlscore if *this* score is larger and combine all masks
            mask = events.mlscore[self.proc_col_name] >= events.mlscore[col_name]
            outp_mask = outp_mask & mask

        return events, outp_mask


@categorizer(uses={"{Electron,Muon}.{pt,eta,phi,mass}", "mll"})
def mask_fn_highpt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Categorizer that selects events in the phase space that we understand.
    Needs to be used in combination with a Producer that defines the leptons.
    """
    mask = (events.Lepton[:, 0].pt > 70) & (events.Lepton[:, 1].pt > 50) & (events.mll > 20)
    return events, mask


@categorizer(uses={"gen_hbw_decay.*.*"})
def mask_fn_gen_barrel(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Categorizer that selects events generated only in the barrel region
    """
    mask = (abs(events.gen_hbw_decay["sec1"]["eta"]) < 2.4) & (abs(events.gen_hbw_decay["sec2"]["eta"]) < 2.4)
    return events, mask


@categorizer(uses={"mll"}, mll=20)
def mask_fn_mll20(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, (events.mll > self.mll)


mask_fn_mll15 = mask_fn_mll20.derive("mask_fn_mll15", cls_dict={"mll": 15})
