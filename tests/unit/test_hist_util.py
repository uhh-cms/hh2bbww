# coding: utf-8

"""
Tests of hbw.hist_util.
"""

from __future__ import annotations

import pytest

from columnflow.util import maybe_import

from hbw.hist_util import apply_rebinning_edges

np = maybe_import("numpy")
hist = maybe_import("hist")


def make_hist(values, edges=(0, 1, 2, 3, 4), variances=None, flow=False):
    h = hist.Hist(
        hist.axis.Variable(edges, name="x", underflow=flow, overflow=flow),
        storage=hist.storage.Weight(),
    )
    view = h.view(flow=flow)
    view.value = values
    view.variance = values if variances is None else variances
    return h


def test_rebin_merge_bins():
    h = make_hist([1.0, 2.0, 3.0, 4.0])
    h_new = apply_rebinning_edges(h, "x", [0, 2, 4])
    assert list(h_new.axes["x"].edges) == [0, 2, 4]
    assert list(h_new.values()) == [3.0, 7.0]
    assert list(h_new.variances()) == [3.0, 7.0]


def test_rebin_keeps_flow():
    # values ordered as [underflow, bins..., overflow]
    h = make_hist([10.0, 1.0, 2.0, 3.0, 4.0, 20.0], flow=True)
    h_new = apply_rebinning_edges(h, "x", [0, 1, 4])
    assert list(h_new.values()) == [1.0, 9.0]
    assert list(h_new.values(flow=True)) == [10.0, 1.0, 9.0, 20.0]


def test_rebin_subset_range_moves_content_to_flow():
    h = make_hist([1.0, 2.0, 3.0, 4.0])
    h_new = apply_rebinning_edges(h, "x", [1, 2, 3])
    assert list(h_new.values()) == [2.0, 3.0]
    # bins outside of the new range end up in the under- and overflow
    assert list(h_new.values(flow=True)) == [1.0, 2.0, 3.0, 4.0]
    assert h_new.sum(flow=True).value == h.sum().value


def test_rebin_integer():
    h = make_hist([1.0, 2.0, 3.0, 4.0])
    h_new = apply_rebinning_edges(h, "x", 2)
    assert list(h_new.values()) == [3.0, 7.0]


def test_rebin_incompatible_edges():
    h = make_hist([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        apply_rebinning_edges(h, "x", [0, 1.5, 4])


def test_rebin_preserves_other_axes():
    h = hist.Hist(
        hist.axis.IntCategory([1, 2], name="process"),
        hist.axis.Variable([0, 1, 2, 3, 4], name="x", underflow=False, overflow=False),
        storage=hist.storage.Weight(),
    )
    h.fill(process=[1, 1, 2, 2], x=[0.5, 1.5, 2.5, 3.5], weight=[1, 2, 3, 4])
    h_new = apply_rebinning_edges(h, "x", [0, 2, 4])
    assert np.allclose(h_new.values(), [[3.0, 0.0], [0.0, 7.0]])
