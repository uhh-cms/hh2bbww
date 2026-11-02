# coding: utf-8

"""
Tests of the efficiency helpers in hbw.trigger.trigger_util.
"""

from __future__ import annotations

from columnflow.util import maybe_import

from hbw.trigger.trigger_util import safe_div, binom_int, calc_efficiency_errors, calc_sf_uncertainty

np = maybe_import("numpy")
hist = maybe_import("hist")


def test_safe_div():
    num = np.array([1.0, 0.0, 2.0, 3.0])
    den = np.array([2.0, 0.0, 0.0, 4.0])
    assert np.allclose(safe_div(num, den), [0.5, 1.0, 1.0, 0.75])
    assert np.allclose(safe_div(num, den, default=0.0), [0.5, 0.0, 1.0, 0.75])


def test_binom_int():
    low, high = binom_int(np.array([5.0, 0.0, 10.0]), np.array([10.0, 10.0, 10.0]))
    # central efficiencies 0.5, 0.0, 1.0 must be covered by the interval
    assert np.all(low <= np.array([0.5, 0.0, 1.0]))
    assert np.all(high >= np.array([0.5, 0.0, 1.0]))
    # boundaries: no lower uncertainty for eff = 0, upper bound 1 for eff = 1
    assert low[1] == 0.0
    assert high[2] == 1.0


def weight_hist(values):
    h = hist.Hist(hist.axis.Regular(len(values), 0, 1, name="x"), storage=hist.storage.Weight())
    view = h.view()
    view.value = values
    view.variance = values
    return h


def test_calc_efficiency_errors():
    num = weight_hist(np.array([5.0, 0.0, 8.0]))
    den = weight_hist(np.array([10.0, 10.0, 8.0]))
    errors = calc_efficiency_errors(num, den)

    # shape (2, n_bins): lower and upper errors per bin
    assert errors.shape == (2, 3)
    assert np.all(errors >= 0)
    # zero efficiency -> errors are removed
    assert np.all(errors[:, 1] == 0)
    # efficiency 1 -> no upper error
    assert errors[1, 2] == 0


def test_calc_sf_uncertainty():
    efficiencies = {"data": np.array([0.8, 0.9]), "mc": np.array([0.8, 0.9])}
    errors = {
        "data": np.array([[0.03, 0.01], [0.04, 0.02]]),
        "mc": np.array([[0.0, 0.0], [0.0, 0.0]]),
    }
    unc = calc_sf_uncertainty(efficiencies, errors, alpha=np.array([1.0, 1.0]))
    # only the (symmetrized, i.e. maximum) data uncertainty contributes
    assert np.allclose(unc, np.array([0.04, 0.02]) / np.array([0.8, 0.9]))

    # non-closure (alpha != 1) is added in quadrature
    unc_alpha = calc_sf_uncertainty(efficiencies, errors, alpha=np.array([0.9, 0.9]))
    assert np.all(unc_alpha > unc)
