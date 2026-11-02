# coding: utf-8

"""
Tests of the object selection on small synthetic events with known results.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.selection import SelectionResult

from hbw.selection.jet import jet_selection
from hbw.selection.lepton import lepton_definition
from hbw.selection.dl_remastered import dl_lepton_selection

from tests.helpers import toy_config, make_events, run_taf

np = maybe_import("numpy")
ak = maybe_import("awkward")


def obj(pt, eta=0.0, phi=0.0, mass=0.0, **fields):
    return {"pt": pt, "eta": eta, "phi": phi, "mass": mass, **fields}


#
# jet selection (toy config)
#

def jet(pt, eta=0.0, phi=0.0, jet_id=6, btag=0.0):
    return obj(pt, eta, phi, 10.0, jetId=jet_id, btagUParTAK4B=btag)


@pytest.fixture(scope="module")
def jet_results():
    config_inst = toy_config(btag_column="btagUParTAK4B", btag_wp_score=0.5, b_tagger="upart", btag_wp="medium")
    events = make_events({
        "Jet": [
            [
                jet(50.0, btag=0.9),
                jet(30.0, eta=1.0, phi=1.5, btag=0.2),  # overlaps with the electron
                jet(20.0, phi=-2.0),  # below pt threshold
                jet(40.0, eta=3.0, phi=-1.0),  # forward jet
                jet(60.0, eta=0.5, phi=3.0, jet_id=0),  # fails the tight jet id
            ],
            [
                jet(30.0, phi=0.0, btag=0.1),
                jet(80.0, phi=2.0, btag=0.6),
                jet(45.0, phi=-2.0, btag=0.7),
            ],
            [],
        ],
        "Electron": [[obj(30.0, eta=1.0, phi=1.6, charge=-1)], [], []],
        # muon overlapping with a jet, but not selected as lepton -> no overlap removal
        "Muon": [[], [obj(20.0, phi=2.0, charge=1)], []],
    }, behavior=True)
    lepton_results = SelectionResult(objects={
        "Electron": {"Electron": ak.Array([[0], [], []])},
        "Muon": {"Muon": ak.Array([[], [], []])},
    })
    return run_taf(jet_selection, events, config_inst, lepton_results=lepton_results, stats=defaultdict(float))


def test_jet_objects(jet_results):
    _, results = jet_results
    jets = results.objects.Jet
    # central jets are pt-sorted, without leptons in a cone of 0.4, low pt jets or jets failing the id
    assert jets.Jet.to_list() == [[0], [1, 2, 0], []]
    assert jets.ForwardJet.to_list() == [[3], [], []]
    # b-jets are the two jets with the highest b-score, light jets all others
    assert jets.Bjet.to_list() == [[0], [2, 1], []]
    assert jets.Lightjet.to_list() == [[], [0], []]


def test_jet_steps(jet_results):
    events, results = jet_results
    assert events.cutflow.n_jet.to_list() == [1, 3, 0]
    assert events.cutflow.n_btag.to_list() == [1, 2, 0]
    assert results.steps.nJet1.to_list() == [True, True, False]
    assert results.steps.nJet3.to_list() == [False, True, False]
    assert results.steps.nBjet1.to_list() == [True, True, False]
    assert results.steps.nBjet2.to_list() == [False, True, False]
    assert np.allclose(ak.to_numpy(results.x.ht), [50.0, 155.0, 0.0])


#
# lepton selection (full config, since the lepton id configuration is taken from it)
#

def electron(pt, eta=0.0, phi=0.0, charge=-1, cut_based=4, dxy=0.01, dz=0.01, delta_eta_sc=0.0):
    return obj(pt, eta, phi, charge=charge, cutBased=cut_based, dxy=dxy, dz=dz, deltaEtaSC=delta_eta_sc)


def muon(pt, eta=0.0, phi=0.0, charge=1, loose=True, tight=True, iso=4):
    return obj(pt, eta, phi, 0.105, charge=charge, looseId=loose, tightId=tight, pfIsoId=iso, dxy=0.0, dz=0.0)


def tau(pt, eta=0.0, phi=0.0):
    return obj(
        pt, eta, phi, 1.7, charge=1, dz=0.0, decayMode=1,
        idDeepTau2018v2p5VSe=2, idDeepTau2018v2p5VSmu=1, idDeepTau2018v2p5VSjet=5,
    )


LEPTON_EVENTS = {
    "Electron": [
        [electron(40.0, eta=0.1, phi=0.0)],  # tight
        [electron(40.0, eta=1.45, delta_eta_sc=0.05)],  # supercluster in barrel-endcap gap
        [electron(40.0, cut_based=2)],  # loose and fakeable, but not tight
        [electron(40.0, eta=2.0, dxy=0.08, dz=0.15)],  # tight endcap electron
        [electron(20.0, eta=0.1), electron(45.0, eta=-0.1, phi=2.5, charge=1)],  # two tight electrons
    ],
    "Muon": [
        [muon(35.0, eta=-0.1, phi=3.0)],  # tight
        [muon(35.0, iso=2)],  # loose and fakeable, not tight
        [muon(4.0)],  # below pt threshold
        [muon(35.0, eta=2.5)],  # outside acceptance
        [],
    ],
    "Tau": [[], [], [], [tau(20.0, eta=1.0)], []],
}


@pytest.fixture(scope="module")
def lepton_config(get_config):
    return get_config("hbw_dl", "c24v15")


@pytest.mark.config
def test_lepton_definition(lepton_config):
    events = make_events(LEPTON_EVENTS, behavior=True)
    events, results = run_taf(
        lepton_definition, events, lepton_config, lepton_config.x.default_dataset, setup=True,
        stats=defaultdict(float),
    )
    cutflow = events.cutflow
    assert cutflow.n_loose_electron.to_list() == [1, 0, 1, 1, 2]
    assert cutflow.n_fakeable_electron.to_list() == [1, 0, 1, 1, 2]
    assert cutflow.n_tight_electron.to_list() == [1, 0, 0, 1, 2]
    assert cutflow.n_loose_muon.to_list() == [1, 1, 0, 0, 0]
    assert cutflow.n_fakeable_muon.to_list() == [1, 1, 0, 0, 0]
    assert cutflow.n_tight_muon.to_list() == [1, 0, 0, 0, 0]
    assert events.Electron.is_tight.to_list() == [[True], [False], [False], [True], [True, True]]
    assert events.Muon.is_tight.to_list() == [[True], [False], [False], [False], []]

    # objects are pt-sorted
    assert results.objects.Electron.TightElectron.to_list() == [[0], [], [], [0], [1, 0]]
    assert results.objects.Tau.VetoTau.to_list() == [[], [], [], [0], []]
    assert results.steps.VetoTau.to_list() == [True, True, True, False, True]
    # all lepton pairs are outside of the low mass and Z mass windows
    assert all(results.steps.ll_lowmass_veto)
    assert all(results.steps.ll_zmass_veto)


@pytest.mark.config
def test_dl_lepton_selection(lepton_config):
    events = make_events(LEPTON_EVENTS, behavior=True)
    events, results = run_taf(
        dl_lepton_selection, events, lepton_config, lepton_config.x.default_dataset, setup=True,
        stats=defaultdict(float),
    )
    steps = results.steps
    # event 0: e + mu, event 4: two electrons
    assert steps.Dilepton.to_list() == [True, False, False, False, True]
    assert steps.Lep_ee.to_list() == [False, False, False, False, True]
    assert steps.Lep_mixed.to_list() == [True, False, False, False, False]
    assert steps.Lep_mm.to_list() == [False] * 5
    assert steps.Charge.to_list() == [True, True, True, False, True]
    assert steps.SR.to_list() == [True, False, False, False, True]

    # mll from the selected leptons (pt 40/35 and 45/20, see LEPTON_EVENTS)
    mll = events.mll.to_list()
    assert mll[0] == pytest.approx(np.sqrt(2 * 40 * 35 * (np.cosh(0.2) - np.cos(3.0))), rel=1e-3)
    assert mll[4] == pytest.approx(np.sqrt(2 * 20 * 45 * (np.cosh(0.2) - np.cos(2.5))), rel=1e-3)
    assert mll[1] == EMPTY_FLOAT
