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


# def integrate_hist(h, **kwargs):
#     """
#     Given a scikit-hist histogram object return a reduced histogram with specified
#     axes integrated out.

#     For scikit-hist histograms, the integration should be formed in 3 steps:
#     - slicing the histogram to contain only the range of interest
#     - Setting overflow values to 0 (excluding the values from future calculations)
#     - Summing over the axes of interest.

#     The latter 2 steps will only be carried out if the var_slice doesn't uniquely
#     identify a singular bin in the histogram axis
#     """
#     # Reduction in parallel.
#     r = h[kwargs]
#     for var, var_slice in kwargs.items():
#         # In the case that histogram has been reduced to singular value simple return
#         if not isinstance(r, hist.Hist):
#             return r
#         if var in [x.name for x in r.axes]:
#             ax = h.axes[var]
#             get_underflow = var_slice.start == None or var_slice.start == -1
#             get_overflow = var_slice.stop == None or var_slice.stop == len(ax)
#             if not get_underflow and ax.traits.underflow:
#                 r[{var: hist.underflow}] = np.zeros_like(r[{var: hist.underflow}])
#             if not get_overflow and ax.traits.overflow:
#                 r[{var: hist.overflow}] = np.zeros_like(r[{var: hist.overflow}])

#             # Sum over all remaining elements on axis
#             r = r[{var: sum}]
#     return r


# def rebin_hist(h, **kwargs):
#     """
#     Rebinning a scikit-hist histogram. 2 types of values can be accepted as the
#     argument values:
#     - Derivatives of the `hist.rebin` argument. In this case we directly use the
#         UHI facilities to perform the rebinning.
#     - A new axis object where all the bin edges lands on the old bin edges of the
#         given histogram. In this case a custom intergration loop is performed to
#         extract the rebinning. Beware that this methods is very slow, as it requires
#         a loop generation of all possible UHI values after the rebinning, so be sure
#         that rebinning is performed as the final step of the histogram reduction. See
#         `_rebin_single_scikit` for more information regarding this method.
#     """
#     h = h.copy()
#     for var, var_val in kwargs.items():
#         if isinstance(var_val, hist.rebin):
#             h = h[{var: var_val}]
#         else:
#             h = _rebin_single_scikit(h, var, var_val)
#     return h


# def __check_scikit_axis_compat(axis1, axis2):
#     """
#     Checking that axis 2 is rebin-compatible with axis 1. This checks that:
#     1. The two histogram share the same name.
#     2. The edges of the second axis all land on the edges of the first axis.

#     If the two axis are compatible the function will return an array of the bin
#     index of the axis 1 that the bin edges of axis 2 falls on.
#     """
#     assert axis1.name == axis2.name, \
#         'Naming of the axis is required to match'
#     # Getting the new bin edges index for the old bin edges
#     try:
#         return [
#             np.argwhere(axis1.edges == new_edge)[0][0] for new_edge in axis2.edges
#         ]
#     except IndexError:
#         raise ValueError(
#             f"Bin edges of the axis {axis2} is incompatible with {axis1}")


# def _get_all_indices(axis):
#     """
#     Getting all possible (integer) bin index values given a scikit-hep histogram.
#     The special indices of hist.underflow and hist.overflow will be included if the
#     axis in questions has those traits.
#     """
#     idxs = list(range(len(axis)))
#     if axis.traits.underflow:    # Extension to include the under/overflow bins
#         idxs.insert(0, hist.underflow)
#     if axis.traits.overflow:
#         idxs.append(hist.overflow)
#     return idxs


# def _rebin_single_scikit(h, old_axis, new_axis):
#     """
#     Rebinning a single axis of a scikit-hist histogram. This includes the following
#     routines:

#     - Generating a new scikit hep instance that perserves axis ordering with the
#         exception of the rebinned axis (in place) replacement.
#     - Setting up the integration ranges required to calculate the bin values of the
#         new histogram.
#     - Looping over the UHI values of the new histogram and performing the a
#         summation over the specified range on the old histogram to fill in the new
#         values.

#     As here we have variable number of axis each with variable number of bins, this
#     method will require the use of more old fashioned python looping, which can be
#     very slow for large dimensional histograms with many bins for each axis. So be
#     sure to make rebinning be the final step in histogram reduction.
#     """
#     # assert isinstance(h, hist.NamedHist), "Can only process named histograms"
#     # Additional type casing
#     if isinstance(old_axis, str):
#         return _rebin_single_scikit(h, h.axes[old_axis], new_axis)
#     axis_name = old_axis.name

#     # Creating the new histogram instance with identical axis ordering.
#     all_axes = list(h.axes)
#     all_axes[all_axes.index(old_axis)] = new_axis
#     h_rebinned = hist.Hist(*all_axes, storage=h._storage_type())

#     # Getting the all possible bin indices for all axes in the old histogram
#     bin_idx_dict = {ax.name: _get_all_indices(ax) for ax in h.axes}

#     # Getting the new bin edges index for the old bin edges
#     new_bin_edge_idx = __check_scikit_axis_compat(old_axis, new_axis)
#     if new_axis.traits.underflow:    # Adding additional underflow/overflow
#         new_bin_edge_idx.insert(0, bin_idx_dict[axis_name][0])
#     if new_axis.traits.overflow:
#         new_bin_edge_idx.append(bin_idx_dict[axis_name][-1])

#     # Generating a the int range pair. Additional parsing will be required for the
#     # under/overflow bins
#     def make_slice(index):
#         start = new_bin_edge_idx[index]
#         stop = new_bin_edge_idx[index + 1]
#         if start == hist.underflow:
#             start = -1
#         if stop == hist.overflow:
#             stop = len(old_axis)
#         return slice(int(start), int(stop))

#     new_axis_idx = _get_all_indices(new_axis)
#     new_int_slice = [make_slice(i) for i in range(len(new_axis_idx))]
#     assert len(new_axis_idx) == len(new_bin_edge_idx) - 1

#     new_idx_dict = bin_idx_dict.copy()
#     new_idx_dict[axis_name] = new_axis_idx
#     bin_idx_dict[axis_name] = new_int_slice

#     name_list = list(bin_idx_dict.keys())
#     new_idx = [x for x in itertools.product(*[x for x in new_idx_dict.values()])]
#     old_int = [x for x in itertools.product(*[x for x in bin_idx_dict.values()])]

#     print(new_idx)
#     print("Here")
#     print(old_int)
#     for o, n in zip(old_int, new_idx):
#         n_uhi = {name: n[name_idx] for name_idx, name in enumerate(name_list)}
#         o_uhi = {name: o[name_idx] for name_idx, name in enumerate(name_list)}
#         # Single variable histogram, with just the axis of interest
#         h_rebinned[n_uhi] = integrate_hist(h, **o_uhi)

#     return h_rebinned


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
            "mlscore.hh_ggf_hbb_hvv2l2nu_kl1_kt1_manybins": [0.0, 0.429, 0.509, 0.5720000000000001, 0.629, 0.68, 0.72, 0.757, 0.789, 0.8200000000000001, 1.0],  # noqa
            "mlscore.hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1_manybins": [0.0, 0.427, 0.529, 0.637, 0.802, 1.0],
            "mlscore.tt_manybins": [0.0, 0.533, 0.669, 1.0],
            "mlscore.h_manybins": [0.0, 0.494, 0.651, 1.0],
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
