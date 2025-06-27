# coding: utf-8

"""
Histogram hooks.
"""

from __future__ import annotations

import law
import order as od

from columnflow.util import maybe_import, DotDict
from hbw.hist_util import apply_rebinning_edges

np = maybe_import("numpy")
hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


def rebin(task, hists: hist.Histogram):
    """
    Rebin histograms with edges that are pre-defined for a certain variable and category.
    Lots of hard-coded stuff at the moment.
    """
    # get variable inst assuming we created a 1D histogram
    variable_inst = task.config_inst.get_variable(task.branch_data.variable)

    # edges for 2b channel
    edges = {
        "mlscore.hh_ggf_hbb_hvv2l2nu_kl1_kt1": [0.0, 0.429, 0.509, 0.5720000000000001, 0.629, 0.68, 0.72, 0.757, 0.789, 0.8200000000000001, 1.0],  # noqa
        "mlscore.hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": [0.0, 0.427, 0.529, 0.637, 0.802, 1.0],
        "mlscore.tt": [0.0, 0.533, 0.669, 1.0],
        "mlscore.h": [0.0, 0.494, 0.651, 1.0],
    }

    h_rebinned = DotDict()

    edges = edges[variable_inst.name]
    for proc, h in hists.items():
        old_axis = h.axes[variable_inst.name]

        h_rebin = apply_rebinning_edges(h.copy(), old_axis.name, edges)

        if not np.isclose(h.sum().value, h_rebin.sum().value):
            raise Exception(f"Rebinning changed histogram value: {h.sum().value} -> {h_rebin.sum().value}")
        if not np.isclose(h.sum().variance, h_rebin.sum().variance):
            raise Exception(f"Rebinning changed histogram variance: {h.sum().variance} -> {h_rebin.sum().variance}")
        h_rebinned[proc] = h_rebin

    return h_rebinned


def blind_bins(task, hists: hist.Histogram, blinding_threshold=0.08):
    from columnflow.plotting.plot_util import blind_sensitive_bins

    out_hists = {}
    for config_inst, hists in hists.items():
        # unify histogram shapes
        hist_list = list(hists.values())
        zero_hist = sum([h * 0 for h in hist_list[1:]], hist_list[0] * 0)
        hists = {proc: zero_hist + h for proc, h in hists.items()}

        # apply blinding if s/sqrt(b) > blinding_threshold
        # NOTE: this does not yet work for Multi-dim histograms that include categories as is the case here....
        out_hists[config_inst] = blind_sensitive_bins(
            hists, config_inst, threshold=blinding_threshold, remove_mc=True,
        )
    return out_hists


def add_hist_hooks(config: od.Config) -> None:
    """
    Add histogram hooks to a configuration.
    """
    # add hist hooks to config
    config.x.hist_hooks = {
        "rebin": rebin,
        # "blind": blind_bins,
    }
