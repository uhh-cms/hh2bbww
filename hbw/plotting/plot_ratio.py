# coding: utf-8

"""
Ratio plots
"""

from __future__ import annotations

__all__ = []

from collections import OrderedDict

import law
import order as od

from columnflow.util import maybe_import
from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import (
    prepare_stack_plot_config,
    prepare_style_config,
    remove_residual_axis,
    apply_variable_settings,
    apply_process_settings,
    apply_process_scaling,
    apply_density,
    blind_sensitive_bins,
    remove_negative_contributions,
)

from columnflow.types import TYPE_CHECKING

np = maybe_import("numpy")
if TYPE_CHECKING:
    plt = maybe_import("matplotlib.pyplot")
    hist = maybe_import("hist")


logger = law.logger.get_logger(__name__)


def plot_ratio(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    shift_insts: list[od.Shift] | None,
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool | None = False,
    yscale: str | None = "",
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    variable_inst = variable_insts[0]

    # process-based settings (styles and attributes)
    hists, process_style_config = apply_process_settings(hists, process_settings)
    # variable-based settings (rebinning, slicing, flow handling)
    hists, variable_style_config = apply_variable_settings(hists, variable_insts, variable_settings)
    # remove data in bins where sensitivity exceeds some threshold
    blinding_threshold = kwargs.get("blinding_threshold", None)
    if blinding_threshold:
        hists = blind_sensitive_bins(hists, config_inst, blinding_threshold)

    # remove negative contributions per process if requested
    if kwargs.get("remove_negative", None):
        hists = remove_negative_contributions(hists)

    # process scaling
    hists = apply_process_scaling(hists)
    # density scaling per bin
    if density:
        hists = apply_density(hists, density)

    if len(shift_insts) == 1:
        # when there is exactly one shift bin, we can remove the shift axis
        hists = remove_residual_axis(hists, "shift", select_value=shift_insts[0].name)
    else:
        # remove shift axis of histograms that are not to be stacked
        unstacked_hists = {
            proc_inst: h
            for proc_inst, h in hists.items()
            if proc_inst.is_mc and getattr(proc_inst, "unstack", False)
        }
        hists |= remove_residual_axis(unstacked_hists, "shift", select_value="nominal")

    # prepare the plot config
    plot_config = prepare_stack_plot_config(
        hists,
        shape_norm=shape_norm,
        shift_insts=shift_insts,
        **kwargs,
    )

    # add ratio between processes
    norm = None
    integral = None
    # per default, use the mc_stack as denominator
    for key, config in plot_config.items():
        if key == "mc_stack":
            norm = np.sum([config["hist"][i].values() for i in range(len(config["hist"]))], axis=0)
            integral = sum(norm)
            break
    for key, config in plot_config.items():
        if norm is None:
            # if there was no mc_stack, just use the first process as denominator
            norm = config["hist"].values()
            integral = sum(norm)
        if "line" not in key:
            continue
        config["ratio_kwargs"] = config["kwargs"].copy()
        if shape_norm:
            config["ratio_kwargs"]["norm"] = norm / integral * sum(config["hist"].values())
        else:
            config["ratio_kwargs"]["norm"] = norm
        plot_config[key] = config

    # prepare and update the style config
    default_style_config = prepare_style_config(
        config_inst,
        category_inst,
        variable_inst,
        density,
        shape_norm,
        yscale,
    )
    # additional, plot function specific changes
    if shape_norm:
        default_style_config["ax_cfg"]["ylabel"] = "Normalized entries"
    style_config = law.util.merge_dicts(
        default_style_config,
        process_style_config,
        variable_style_config[variable_inst],
        style_config,
        deep=True,
    )
    # change cms label to 2018
    style_config["cms_label_cfg"] = {"lumi": "59.8", "com": 13}
    style_config["rax_cfg"]["ylabel"] = "Ratio"

    return plot_all(plot_config, style_config, **kwargs)


def plot_variable_efficiency_slice(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    shift_insts: list[od.Shift] | None,
    style_config: dict | None = None,
    shape_norm: bool = True,
    cumsum_reverse: bool = True,
    slice_list: list[complex] | None = [200j, 400j, 600j, 800j],
    # slice_down: complex | None = None,
    # slice_up: complex | None = None,
    **kwargs,
):
    """
    This plot function allows users to plot the efficiency of a cut on a variable as a function of the cut value.
    Per default, each bin shows the efficiency of requiring value >= bin edge (cumsum_reverse=True).
    Setting cumsum_reverse=False will instead show the efficiency of requiring value <= bin edge.

    Copy-pasted from c/f to include functionality for slicing a 2-d histogram before calculating the efficiency.
    Example task call to calculate our Hbb efficiencies in bins of fatbjet0_pt:

    claw run cf.PlotVariables1D --variables "fatbjet0_pt_for_sf-fatbjet0_pnet_hbb" --processes ddl4 \
        --plot-function hbw.plotting.plot_ratio.plot_variable_efficiency_slice \
        --plot-suffix pt_300to600 --general-settings "slice_down=300j,slice_up=600j" --yscale log \
        --configs $all_configs \
        --categories sr,ttcr \
        --remove-output 0,a,y --workers 4

    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    if len(variable_insts) != 2:
        raise Exception("This plot function requires exactly two variables.")

    slice_variable_inst = variable_insts[0]
    # use axis 1 for slicing
    variable_insts = [variable_insts[1]]
    logger.info(
        f"Using variable '{slice_variable_inst.name}' for slicing and '{variable_insts[0].name}' for plotting.",
    )

    # build one plot per slice range
    figures = []
    axs_combined = []
    for i, slice_up in enumerate(slice_list[1:]):
        slice_down = slice_list[i]
        logger.info(f"Slicing {slice_variable_inst.name} in range [{slice_down}, {slice_up}]")
        inp_hists = hists.copy()
        for proc_inst, proc_hist in inp_hists.items():
            if slice_down is not None and slice_up is not None:
                proc_hist = proc_hist[{slice_variable_inst.name: slice(slice_down, slice_up)}]

            # sum over each individual slice bin (to get rid of over and underflow)
            # NOTE: there has to be a better way to do this with hist (but .project does not allow passing flow=False)
            # and .sum does not allow summing over one axis only
            # proc_hist = proc_hist[{slice_variable_inst.name: sum}]
            out_hist = proc_hist[{slice_variable_inst.name: 0}]
            for bin_idx in range(1, len(proc_hist.axes[slice_variable_inst.name])):
                out_hist += proc_hist[{slice_variable_inst.name: bin_idx}]
            inp_hists[proc_inst] = out_hist

        for proc_inst, proc_hist in inp_hists.items():
            if cumsum_reverse:
                proc_hist.values()[...] = np.cumsum(proc_hist.values()[..., ::-1], axis=-1)[..., ::-1]
                shape_norm_func = kwargs.get(
                    "shape_norm_func",
                    lambda h, shape_norm: h.values()[0] if shape_norm else 1,
                )
            else:
                proc_hist.values()[...] = np.cumsum(proc_hist.values(), axis=-1)
                shape_norm_func = kwargs.get(
                    "shape_norm_func",
                    lambda h, shape_norm: h.values()[-1] if shape_norm else 1,
                )

        # custom style changes
        cat_label = category_inst.label if category_inst else ""
        default_style_config = {
            "ax_cfg": {"ylabel": "Efficiency" if shape_norm else "Cumulative entries"},
        }
        if slice_up and slice_down:
            if isinstance(slice_up, complex) and isinstance(slice_down, complex):
                slice_up = int(slice_up.imag)
                slice_down = int(slice_down.imag)
            default_style_config["annotate_cfg"] = {
                "text": cat_label + "\n" + rf"$p_{{T}} \in [{slice_down}, {slice_up}]$",
            }

        style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

        from columnflow.plotting.plot_functions_1d import plot_variable_stack
        fig, axs = plot_variable_stack(
            inp_hists,
            config_inst,
            category_inst,
            variable_insts,
            shift_insts,
            shape_norm=shape_norm,
            shape_norm_func=shape_norm_func,
            style_config=style_config,
            **kwargs,
        )
        figures.append(fig)
        axs_combined.append(axs)

    # put all figures into a single large canvas
    n_figs = len(figures)
    n_cols = int(np.ceil(np.sqrt(n_figs)))
    n_rows = int(np.ceil(n_figs / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    for i, (fig_i, axs_i) in enumerate(zip(figures, axs_combined)):
        row = int(i / n_cols)
        col = i % n_cols
        if n_rows > 1 and n_cols > 1:
            ax = axs[row, col]
        elif n_rows > 1:
            ax = axs[row]
        elif n_cols > 1:
            ax = axs[col]
        else:
            ax = axs
        for child in fig_i.get_children():
            if isinstance(child, mpl.axes.Axes):
                for item in child.get_children():
                    try:
                        item.remove()
                        item.set_figure(fig)
                        ax.add_artist(item)
                    except Exception:
                        pass

    return fig, axs
