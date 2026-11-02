# coding: utf-8

"""
Tests of the categorizers in hbw.categorization.categories on small synthetic events.
"""

from __future__ import annotations

import pytest

from columnflow.util import maybe_import

from hbw.categorization.categories import (
    catid_incl, catid_1e, catid_1mu, catid_2e, catid_2mu, catid_emu, catid_ge3lep,
    catid_sr, catid_fake, catid_highmet, catid_lowmet,
    catid_mll_low_narrow, catid_mll_low_wide, catid_cr, catid_mll_z_narrow, catid_mll_z_wide,
    catid_mll_high, catid_mll_very_high,
    catid_fatjet, catid_njet1, catid_njet2, catid_njet3,
    catid_le1b_loose, catid_ge2b_loose, catid_1b, catid_2b, catid_eq2b, catid_eq3b, catid_geq3b,
)

from tests.helpers import toy_config, make_events, run_taf

ak = maybe_import("awkward")


@pytest.fixture(scope="module")
def config_inst():
    return toy_config(
        tags={"is_dl", "is_hh"},
        met_name="PuppiMET",
        b_tagger="upart",
        btag_column="btagUParTAK4B",
        btag_wp_score=0.5,
        btag_working_points={"upart": {"loose": 0.1, "medium": 0.5, "tight": 0.8}},
    )


def mask_of(categorizer, events, config_inst, **kwargs) -> list:
    _, mask = run_taf(categorizer, events, config_inst, **kwargs)
    return ak.to_list(mask)


def test_incl(config_inst):
    events = make_events({"event": [1, 2, 3]})
    assert mask_of(catid_incl, events, config_inst) == [True, True, True]


def test_lepton_multiplicity(config_inst):
    events = make_events({
        "Electron": {"pt": [[30.0], [], [30.0, 20.0], [30.0], [30.0, 20.0]]},
        "Muon": {"pt": [[], [25.0], [], [25.0], [10.0]]},
    })
    assert mask_of(catid_1e, events, config_inst) == [True, False, False, False, False]
    assert mask_of(catid_1mu, events, config_inst) == [False, True, False, False, False]
    assert mask_of(catid_2e, events, config_inst) == [False, False, True, False, False]
    assert mask_of(catid_2mu, events, config_inst) == [False, False, False, False, False]
    assert mask_of(catid_emu, events, config_inst) == [False, False, False, True, False]
    assert mask_of(catid_ge3lep, events, config_inst) == [False, False, False, False, True]


def test_sr_fake(config_inst):
    events = make_events({
        "Electron": {"is_tight": [[True, True], [True], [], [True, False]]},
        "Muon": {"is_tight": [[], [True], [False], [True]]},
    })
    # DL requires exactly two tight leptons for the signal region
    assert mask_of(catid_sr, events, config_inst) == [True, True, False, True]
    assert mask_of(catid_fake, events, config_inst) == [False, False, True, False]

    sl_config = toy_config(tags={"is_sl"})
    assert mask_of(catid_sr, events, sl_config) == [False, False, False, False]
    assert mask_of(catid_fake, events, sl_config) == [False, False, True, False]


def test_met(config_inst):
    events = make_events({"PuppiMET": {"pt": [10.0, 19.99, 20.0, 50.0]}})
    assert mask_of(catid_highmet, events, config_inst) == [False, False, True, True]
    assert mask_of(catid_lowmet, events, config_inst) == [True, True, False, False]


@pytest.mark.parametrize("categorizer, expected", [
    # mll values:      11,    12,    20,    69.9,  70,    80,    99.9,  100,   109.9, 110
    (catid_mll_low_narrow, [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]),
    (catid_mll_low_wide, [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    (catid_cr, [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
    (catid_mll_z_narrow, [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]),
    (catid_mll_z_wide, [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]),
    (catid_mll_high, [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]),
    (catid_mll_very_high, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
])
def test_mll(config_inst, categorizer, expected):
    mll = [11.0, 12.0, 20.0, 69.9, 70.0, 80.0, 99.9, 100.0, 109.9, 110.0]
    events = make_events({"mll": mll, "Jet": {"pt": [[]] * len(mll)}})
    assert mask_of(categorizer, events, config_inst) == list(map(bool, expected))


def test_jet_multiplicity(config_inst):
    events = make_events({
        "Jet": {"pt": [[], [50.0], [50.0, 40.0], [50.0, 40.0, 30.0]]},
        "FatJet": {"pt": [[], [], [300.0], []]},
    })
    assert mask_of(catid_njet1, events, config_inst) == [False, True, True, True]
    assert mask_of(catid_njet2, events, config_inst) == [False, False, True, True]
    assert mask_of(catid_njet3, events, config_inst) == [False, False, False, True]
    assert mask_of(catid_fatjet, events, config_inst) == [False, False, True, False]


def test_btag_multiplicity(config_inst):
    # btag scores w.r.t. toy working points loose 0.1, medium 0.5
    events = make_events({"Jet": {"btagUParTAK4B": [
        [],
        [0.05, 0.2],  # 1 loose, 0 medium
        [0.2, 0.3],  # 2 loose, 0 medium
        [0.5, 0.05],  # 1 loose (incl. medium), 1 medium (at threshold)
        [0.9, 0.6],  # 2 medium
        [0.9, 0.6, 0.7],  # 3 medium
    ]}})
    assert mask_of(catid_le1b_loose, events, config_inst) == [True, True, False, True, False, False]
    assert mask_of(catid_ge2b_loose, events, config_inst) == [False, False, True, False, True, True]
    # NOTE: "1b" means at most one medium b-tag (0b events are removed by the selection)
    assert mask_of(catid_1b, events, config_inst) == [True, True, True, True, False, False]
    assert mask_of(catid_2b, events, config_inst) == [False, False, False, False, True, True]
    assert mask_of(catid_eq2b, events, config_inst) == [False, False, False, False, True, False]
    assert mask_of(catid_eq3b, events, config_inst) == [False, False, False, False, False, True]
    assert mask_of(catid_geq3b, events, config_inst) == [False, False, False, False, False, True]
