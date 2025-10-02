# coding: utf-8

"""
Histogram hooks.
"""

from __future__ import annotations

from functools import partial

import law
import order as od

from columnflow.util import maybe_import, DotDict
from hbw.hist_util import apply_rebinning_edges

np = maybe_import("numpy")
hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


def cumsum(
    task,
    hists: hist.Histogram,
    reverse: bool = False,
    **kwargs,
):
    for config_inst, proc_hists in hists.items():
        for proc_inst, proc_hist in proc_hists.items():
            if reverse:
                proc_hist.values()[...] = np.cumsum(proc_hist.values()[..., ::-1], axis=-1)[..., ::-1]
            else:
                proc_hist.values()[...] = np.cumsum(proc_hist.values(), axis=-1)

    return hists


def rebin(task, hists: hist.Histogram, **kwargs):
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


def blind_bins_above_score(task, hists: hist.Histogram, default_cut=0.8, **kwargs):
    var_name = task.branch_data.variable
    if "logit" in var_name:
        # identify logit transformed scores and convert cut accordingly
        score_cut = np.log(default_cut / (1 - default_cut))
    elif "mlscore.sig_" in var_name:
        # only apply blinding for signal scores
        score_cut = default_cut
    else:
        # do nothing
        return hists

    for config_inst, proc_hists in hists.items():
        for proc_inst, proc_hist in proc_hists.items():
            if proc_inst.is_data:
                # blind data above score cut
                axis = proc_hist.axes[var_name]
                bin_indices = np.where(axis.edges[1:] > score_cut)[0]
                proc_hist.values()[..., bin_indices] = -1
                proc_hist.variances()[..., bin_indices] = 0

    return hists


def blind_bins(task, hists: hist.Histogram, blinding_threshold=0.08, **kwargs):
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


def add_hist_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to an analysis.
    """
    # add hist hooks to analysis instance
    analysis_inst.x.hist_hooks = {
        "cumsum": cumsum,
        "cumsum_reverse": partial(cumsum, reverse=True),
        "rebin": rebin,
        # "blind": blind_bins,
        "blind_bins_above_score": blind_bins_above_score,
    }
