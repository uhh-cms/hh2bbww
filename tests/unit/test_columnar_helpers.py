# coding: utf-8

"""
Tests of small columnar and order helpers used by selectors, producers and the config.
"""

from __future__ import annotations

import pytest

import order as od

from columnflow.util import maybe_import

from hbw.selection.common import masked_sorted_indices
from hbw.production.prepare_objects import combine_collections
from hbw.production.process_ids import get_process_id_from_masks
from hbw.config.processes import create_parent_process, unique_process_sanity_check, set_proc_attr

from tests.helpers import make_events

np = maybe_import("numpy")
ak = maybe_import("awkward")


def test_masked_sorted_indices():
    pt = ak.Array([[20.0, 50.0, 30.0], [], [10.0]])
    mask = ak.Array([[True, False, True], [], [False]])
    assert masked_sorted_indices(mask, pt).to_list() == [[2, 0], [], []]
    assert masked_sorted_indices(mask, pt, ascending=True).to_list() == [[0, 2], [], []]


def lep(pt, eta=0.0):
    return {"pt": pt, "eta": eta, "phi": 0.0, "mass": 0.0}


def test_combine_collections():
    events = make_events({
        "Electron": [[lep(30.0)], [], [lep(15.0)]],
        "Muon": [[lep(40.0), lep(10.0)], [lep(20.0)], []],
    })
    events = combine_collections(events, ["Electron", "Muon"], "Lepton")
    assert events.Lepton.pt.to_list() == [[40.0, 30.0, 10.0], [20.0], [15.0]]

    # existing collections are not overwritten
    events = combine_collections(events, ["Muon"], "Lepton")
    assert ak.num(events.Lepton).to_list() == [3, 1, 1]


def test_combine_collections_requires_four_vectors():
    events = make_events({"Electron": [[{"pt": 30.0}]], "Muon": [[lep(40.0)]]})
    events = combine_collections(events, ["Electron", "Muon"], "Lepton")
    assert "Lepton" not in events.fields


@pytest.fixture
def toy_dataset():
    parent = od.Process("tt", id=100)
    sl = parent.add_process("tt_sl", id=101)
    dl = parent.add_process("tt_dl", id=102)
    return od.Dataset("tt_powheg", id=1, processes=[sl, dl], keys=["/toy"])


def test_get_process_id_from_masks(toy_dataset):
    events = ak.Array({"x": [0, 1, 2]})
    masks = {
        "tt_sl": ak.Array([True, False, True]),
        "tt_dl": ak.Array([False, True, False]),
    }
    assert get_process_id_from_masks(events, masks, toy_dataset).to_list() == [101, 102, 101]


def test_get_process_id_from_masks_errors(toy_dataset):
    events = ak.Array({"x": [0, 1]})
    # overlapping masks
    with pytest.raises(ValueError):
        get_process_id_from_masks(events, {
            "tt_sl": ak.Array([True, True]),
            "tt_dl": ak.Array([False, True]),
        }, toy_dataset)
    # unassigned events
    with pytest.raises(ValueError):
        get_process_id_from_masks(events, {"tt_sl": ak.Array([True, False])}, toy_dataset)
    # process not registered for the dataset
    with pytest.raises(NotImplementedError):
        get_process_id_from_masks(events, {"st": ak.Array([True, True])}, toy_dataset)


def test_create_parent_process():
    p1 = od.Process("p1", id=1, xsecs={13: 1.0, 13.6: 2.0})
    p2 = od.Process("p2", id=2, xsecs={13.6: 3.0})
    parent = create_parent_process([p1, p2], name="parent", id=10)
    assert set(parent.processes.names()) == {"p1", "p2"}
    # xsecs are summed for the ecms that all children have in common
    assert list(parent.xsecs.keys()) == [13.6]
    assert parent.get_xsec(13.6).nominal == 5.0

    with pytest.raises(ValueError):
        create_parent_process([p1], name="no_id")


def test_unique_process_sanity_check():
    config_inst = od.Config(name="toy_config", id=1)
    a = config_inst.add_process("a", id=1)
    b = config_inst.add_process("b", id=2)
    a.add_process("a1", id=11)
    unique_process_sanity_check(config_inst)

    # nested processes with duplicate ids are only caught by the sanity check
    b.add_process("b1", id=11)
    with pytest.raises(ValueError):
        unique_process_sanity_check(config_inst)


def test_set_proc_attr():
    proc = od.Process("p", id=1)
    set_proc_attr(proc, "label", r"t\bar{t}")
    set_proc_attr(proc, "color", "#ff0000")
    set_proc_attr(proc, "ml_id", 3)
    assert proc.label == r"$t\bar{t}$"
    assert proc.color1 == od.Process("q", id=2, color="#ff0000").color1
    assert proc.x.ml_id == 3
    with pytest.raises(ValueError):
        set_proc_attr(proc, "id", 5)
