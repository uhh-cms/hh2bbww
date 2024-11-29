# coding: utf-8

"""
Histogram hooks.
"""

from __future__ import annotations

import law
import order as od

from columnflow.util import maybe_import, DotDict

np = maybe_import("numpy")
hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


def rebin_hist(h, axis_name, edges):
    if isinstance(edges, int):
        return h[{axis_name: hist.rebin(edges)}]

    ax = h.axes[axis_name]
    ax_idx = [a.name for a in h.axes].index(axis_name)
    if not all([np.isclose(x, ax.edges).any() for x in edges]):
        raise ValueError(
            f"Cannot rebin histogram due to incompatible edges for axis '{ax.name}'\n"
            f"Edges of histogram are {ax.edges}, requested rebinning to {edges}",
        )

    # If you rebin to a subset of initial range, keep the overflow and underflow
    overflow = ax.traits.overflow or (edges[-1] < ax.edges[-1] and not np.isclose(edges[-1], ax.edges[-1]))
    underflow = ax.traits.underflow or (edges[0] > ax.edges[0] and not np.isclose(edges[0], ax.edges[0]))
    flow = overflow or underflow
    new_ax = hist.axis.Variable(edges, name=ax.name, overflow=overflow, underflow=underflow)
    axes = list(h.axes)
    axes[ax_idx] = new_ax

    hnew = hist.Hist(*axes, name=h.name, storage=h._storage_type())

    # Offset from bin edge to avoid numeric issues
    offset = 0.5 * np.min(ax.edges[1:] - ax.edges[:-1])
    edges_eval = edges + offset
    edge_idx = ax.index(edges_eval)
    # Avoid going outside the range, reduceat will add the last index anyway
    if edge_idx[-1] == ax.size + ax.traits.overflow:
        edge_idx = edge_idx[:-1]

    if underflow:
        # Only if the original axis had an underflow should you offset
        if ax.traits.underflow:
            edge_idx += 1
        edge_idx = np.insert(edge_idx, 0, 0)

    # Take is used because reduceat sums i:len(array) for the last entry, in the case
    # where the final bin isn't the same between the initial and rebinned histogram, you
    # want to drop this value. Add tolerance of 1/2 min bin width to avoid numeric issues
    hnew.values(flow=flow)[...] = np.add.reduceat(h.values(flow=flow), edge_idx,
            axis=ax_idx).take(indices=range(new_ax.size + underflow + overflow), axis=ax_idx)
    if hnew._storage_type() == hist.storage.Weight():
        hnew.variances(flow=flow)[...] = np.add.reduceat(h.variances(flow=flow), edge_idx,
                axis=ax_idx).take(indices=range(new_ax.size + underflow + overflow), axis=ax_idx)
    return hnew


def add_hist_hooks(config: od.Config) -> None:
    """
    Add histogram hooks to a configuration.
    """

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

            h_rebin = rebin_hist(h.copy(), old_axis.name, edges)

            if not np.isclose(h.sum().value, h_rebin.sum().value):
                raise Exception(f"Rebinning changed histogram value: {h.sum().value} -> {h_rebin.sum().value}")
            if not np.isclose(h.sum().variance, h_rebin.sum().variance):
                raise Exception(f"Rebinning changed histogram variance: {h.sum().variance} -> {h_rebin.sum().variance}")
            h_rebinned[proc] = h_rebin

        return h_rebinned

    # add hist hooks to config
    config.x.hist_hooks = {
        "rebin": rebin,
    }
