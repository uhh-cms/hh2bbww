# coding: utf-8

"""
Histogram hooks.
"""

from __future__ import annotations

import math
import law
import order as od

from columnflow.util import maybe_import, DotDict

np = maybe_import("numpy")
hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


def apply_rebinning_edges(h: hist.Histogram, axis_name: str, edges: list):
    """
    Generalized rebinning of a single axis from a hist.Histogram, using predefined edges.

    :param h: histogram to rebin
    :param axis_name: string representing the axis to rebin
    :param edges: list of floats representing the new bin edges. Must be a subset of the original edges.
    :return: rebinned histogram
    """
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


def merge_hists_per_config(
    task,
    hists: dict[str, dict[od.Process, hist.Histogram]],
):
    if len(task.config_insts) != 1:
        process_memory = {}
        merged_hists = {}
        for config, _hists in hists.items():
            for process_inst, h in _hists.items():

                if process_inst.id in merged_hists:
                    merged_hists[process_inst.id] += h
                else:
                    merged_hists[process_inst.id] = h
                    process_memory[process_inst.id] = process_inst

        process_insts = list(process_memory.values())
        hists = {process_memory[process_id]: h for process_id, h in merged_hists.items()}
    else:
        hists = hists[task.config_inst.name]
        process_insts = list(hists.keys())

    return hists, process_insts


def apply_rebin_edges_to_all(
    hists: dict[str, dict[od.Process, hist.Histogram]],
    edges: list[float],
    axis_name: str,
) -> dict[str, dict[od.Process, hist.Histogram]]:
    """
    Apply rebin edges to histograms for all configs and processes.
    """
    h_out = {}
    for config, _hists in hists.items():
        h_rebinned = DotDict()
        for proc, h in _hists.items():
            old_axis = h.axes[axis_name]
            h_rebin = apply_rebinning_edges(h.copy(), old_axis.name, edges)

            if not np.isclose(h.sum().value, h_rebin.sum().value):
                raise Exception(f"Rebinning changed histogram value: {h.sum().value} -> {h_rebin.sum().value}")
            if not np.isclose(h.sum().variance, h_rebin.sum().variance):
                raise Exception(f"Rebinning changed histogram variance: {h.sum().variance} -> {h_rebin.sum().variance}")
            h_rebinned[proc] = h_rebin

        h_out[config] = h_rebinned

    return h_out


def select_category_and_shift(
    task,
    h: hist.Histogram,
):
    # get the shifts to extract and plot
    plot_shifts = law.util.make_list(task.get_plot_shifts())

    category_inst = task.config_inst.get_category(task.branch_data.category)
    leaf_category_insts = category_inst.get_leaf_categories() or [category_inst]

    # selections
    h = h[{
        "category": [
            hist.loc(c.id)
            for c in leaf_category_insts
            if c.id in h.axes["category"]
        ],
        "shift": [
            hist.loc(s.id)
            for s in plot_shifts
            if s.id in h.axes["shift"]
        ],
    }]
    # reductions
    h = h[{"category": sum, "shift": sum}]

    return h


def add_hist_hooks(config: od.Config) -> None:
    """
    Add histogram hooks to a configuration.
    """

    from hbw.util import timeit_multiple
    @timeit_multiple
    def rebin(task, hists: dict[str, dict[od.Process, hist.Histogram]]) -> dict[str, hist.Histogram]:
        """
        Rebin histograms with edges that are pre-defined for a certain variable and category.
        Lots of hard-coded stuff at the moment.
        """
        logger.info("Rebinning histograms")

        # category_inst = task.config_inst.get_category(task.branch_data.category)

        # get variable inst assuming we created a 1D histogram
        variable_inst = task.config_inst.get_variable(task.branch_data.variable)
        variable_inst.x.rebin = None
        rebin_config = variable_inst.x("rebin_config", None)
        if rebin_config is None:
            logger.info("No rebinning configuration found, skipping rebinning")
            return hists

        # merge histograms over all configs
        hists_per_process, hist_process_insts = merge_hists_per_config(task, hists)

        # # get process instances for rebinning (sub procs pls)
        # rebin_process_insts = [task.config_inst.get_process(proc) for proc in rebin_config["processes"]]
        # rebin_sub_process_insts = {
        #     process_inst.name: [
        #         sub
        #         for sub, _, _ in process_inst.walk_processes(include_self=True)
        #         if sub.id in [p.id for p in hist_process_insts]
        #     ]
        #     for process_inst in rebin_process_insts
        # }

        # rebin_process_insts = [
        #     process_inst.name for p in process_insts if p.name in rebin_config["processes"]]

        if missing_procs := set(rebin_config["processes"]) - set([p.name for p in hists_per_process]):
            raise ValueError(
                f"Processes {missing_procs} not found in histograms. For rebinning, the process names "
                "requested in plotting/datacards need to match the processes required for rebinning."
            )

        # get histograms used for rebinning by merging over rebin processes defined by variable inst
        # work on a copy to not modify original hist
        rebin_hist = sum([
            h for proc_inst, h in hists_per_process.items()
            if proc_inst.name in rebin_config["processes"]
        ]).copy()

        # select and reduce category and shift axis
        rebin_hist = select_category_and_shift(task, rebin_hist)

        # the effective number of events should be larger than a certain number
        # error_criterium = lambda value, variance: value ** 2 / variance > rebin.min_entries
        # equal_width_criterium = lambda value, n_bins, integral: value > integral / n_bins

        edges = []

        def get_bin_edges_simple(
            h,
            n_bins,
            reversed_order: bool = False,
        ):
            requested_cumsum = h.sum().value / n_bins
            h_copy = h.copy()

            cumsum_value = np.cumsum(h_copy.values()[::-1])[::-1] if reversed_order else np.cumsum(h_copy.values())
            # cumsum_variance = np.cumsum(h_copy.variances()[::-1])[::-1] if reversed_order else np.cumsum(h_copy.variances())

            current_bin_edge = np.astype(cumsum_value / requested_cumsum, int)

            diffs = np.diff(current_bin_edge)
            indices = np.where(diffs > 0)[0]

            edges = [0.] + list(h_copy.axes[0].edges[indices])
            edges[-1] = 1.0

            return edges

        edges = get_bin_edges_simple(rebin_hist, rebin_config.get("n_bins", 4))
        print(edges)

        h_out = {}
        h_out = apply_rebin_edges_to_all(hists, edges, variable_inst.name)
        return h_out

    # rebin default parameters
    rebin.default_n_bins = 10

    def rebin_example(task, hists: dict[str, dict[od.Process, hist.Histogram]]) -> dict[str, hist.Histogram]:
        """
        Rebin histograms with edges that are pre-defined for a certain variable and category.
        Lots of hard-coded stuff at the moment.
        """
        logger.info("Rebinning histograms")

        # get variable inst assuming we created a 1D histogram
        variable_inst = task.config_inst.get_variable(task.branch_data.variable)
        variable_inst.x.rebin = None

        # edges for 2b channel
        edges = {
            "mlscore.hh_ggf_hbb_hvv2l2nu_kl1_kt1": [0.0, 0.429, 0.509, 0.5720000000000001, 0.629, 0.68, 0.72, 0.757, 0.789, 0.8200000000000001, 1.0],  # noqa
            "mlscore.hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": [0.0, 0.427, 0.529, 0.637, 0.802, 1.0],
            "mlscore.tt": [0.0, 0.533, 0.669, 1.0],
            "mlscore.h": [0.0, 0.494, 0.651, 1.0],
        }

        h_out = {}
        edges = edges[variable_inst.name]
        h_out = apply_rebin_edges_to_all(hists, edges, variable_inst.name)
        return h_out


    # add hist hooks to config
    config.x.hist_hooks = {
        "rebin_example": rebin_example,
        "rebin": rebin,
    }


def rebinning(N=10):
    # pseudo code
    get_final_bin_width = rebin_rest = None
    check_final_bin = True
    while check_final_bin:
        check_final_bin, bin_width = get_final_bin_width(N)
        N = N - 1

    rebin_rest(N)
