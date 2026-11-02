# coding: utf-8

"""
Tests of the helper functions in hbw.tasks.inference.
"""

from __future__ import annotations

import order as od

from columnflow.util import maybe_import

from hbw.tasks.inference import (
    get_hist_name, get_cat_proc_syst_name_from_key, get_rebin_values, resolve_category_groups,
)

np = maybe_import("numpy")
hist = maybe_import("hist")


def test_hist_name_round_trip():
    assert get_hist_name("sr__1b", "hh_ggf") == "sr__1b/hh_ggf"
    assert get_hist_name("sr__1b", "hh_ggf", "jec_up") == "sr__1b/hh_ggf__jec_up"

    assert get_cat_proc_syst_name_from_key("sr__1b/hh_ggf__jec_up;1") == ("sr__1b", "hh_ggf", "jec_up")
    # NOTE: keys without systematic are returned as "nominal" (not None as the docstring says)
    assert get_cat_proc_syst_name_from_key("sr__1b/hh_ggf;1") == ("sr__1b", "hh_ggf", "nominal")


def weight_hist(values, variances=None, edges=None):
    edges = np.linspace(0, 1, len(values) + 1) if edges is None else edges
    h = hist.Hist(hist.axis.Variable(edges, name="mlscore"), storage=hist.storage.Weight())
    view = h.view()
    view.value = values
    view.variance = values if variances is None else variances
    return h


def test_rebin_values_flat():
    # 100 bins with one event each, background with small uncertainties
    n = 100
    signal = weight_hist(np.zeros(n))
    background = weight_hist(np.ones(n), np.full(n, 0.01))
    edges = get_rebin_values(background, signal, background, N_bins_final=4, min_bkg_events=1)

    assert len(edges) == 5
    assert np.isclose(edges[0], 0.0) and np.isclose(edges[-1], 1.0)
    assert np.all(np.diff(edges) > 0)
    # flat background -> equally sized bins
    assert np.allclose(np.diff(edges), 0.25, atol=0.02)


def test_rebin_values_respects_bkg_uncertainty():
    # background bins at high score have large relative uncertainties -> bins are merged further
    n = 100
    signal = weight_hist(np.zeros(n))
    bkg_variance = np.full(n, 0.01)
    bkg_variance[80:] = 10.0
    background = weight_hist(np.ones(n), bkg_variance)
    edges = get_rebin_values(background, signal, background, N_bins_final=4, min_bkg_events=10)

    # the last bin needs to be wider than the default 0.25 to compensate the uncertainty
    assert edges[-1] - edges[-2] > 0.25


def test_resolve_category_groups():
    config_inst = od.Config(name="toy_config", id=1)
    for i, name in enumerate(["sr__1b", "sr__2b", "dycr__1b"]):
        config_inst.add_category(name=name, id=i + 1)
    config_inst.x.category_groups = {"sr": ["sr__1b", "sr__2b", "not_existing"]}

    resolved = resolve_category_groups({"sr": 10, "dycr__1b": 5, "unknown": 1}, config_inst)
    assert resolved == {"sr__1b": 10, "sr__2b": 10, "dycr__1b": 5}
