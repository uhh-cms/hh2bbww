# coding: utf-8

"""
Plots overlaying the distributions of several categories or variables on a single canvas.

Initially taken from bbtautau analysis:
https://github.com/uhh-cms/hh2bbtautau/blob/4a0420ce3645568dcd1f3391fd50af09cc15979c/hbt/plotting/multi.py
"""

from __future__ import annotations

import law
import order as od

from law.util import DotDict

from columnflow.plotting import supports_multi_variable, supports_multi_category
from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import (
    prepare_style_config,
    remove_residual_axis,
    apply_variable_settings,
    apply_process_settings,
    apply_process_scaling,
    apply_density,
    remove_negative_contributions,
    get_attr_or_aux,
)
from columnflow.util import maybe_import
from columnflow.types import TYPE_CHECKING, Any

from hbw.plotting.unrolled import get_category_labels

np = maybe_import("numpy")
if TYPE_CHECKING:
    hist = maybe_import("hist")
    plt = maybe_import("matplotlib.pyplot")


logger = law.logger.get_logger(__name__)


# color scheme, see https://cms-analysis.docs.cern.ch/guidelines/plotting/colors
colors = DotDict(
    # cms recommended 10-color scheme
    blue="#3f90da",
    yellow="#ffa90e",
    red="#bd1f01",
    grey="#94a4a2",
    purple="#832db6",
    brown="#a96b59",
    orange="#e76300",
    olive="#b9ac70",
    dark_grey="#717581",
    teal="#92dadd",
    # additional colors
    green="#4aed00",
    dark_blue="#011c87",
    dark_green="#206700",
    dark_red="#491307",
)

color_names = [
    "blue", "yellow", "red", "grey", "purple", "brown", "orange", "olive", "dark_grey", "teal",
    "green", "dark_blue", "dark_green", "dark_red",
]
get_color_from_sequence = lambda i: colors[color_names[i % len(color_names)]]

line_styles = ["solid", "dotted", "dashed", "dashdot"]
get_line_style_from_sequence = lambda i: line_styles[i % len(line_styles)]


@supports_multi_variable
def multi_var_plot(
    hists: dict[od.Process, hist.Hist] | dict[str, dict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable] | dict[str, list[od.Variable]],
    shift_insts: list[od.Shift],
    style_config: dict | None = None,
    variable_settings: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """
    Create a plot overlaying the distributions of several variables, one line per variable and process. Example:
    claw run cf.PlotVariables1D --multi-variable --plot-function hbw.plotting.multi.multi_var_plot \
        --variables mli_b1_pt,mli_b2_pt --categories sr --processes tt,dy --shape-norm True

    The variables are labeled with their *x_title_short* and the category is shown in the top left. The x-axis
    title and log scale are taken from the first variable, the x-range covers the ranges of all variables. The
    ratio panel (each line divided by the same process in the first variable) is only drawn when all variables
    have the same binning (after applying the variable settings). See plot_multi_hists for the supported options and
    known issues.
    """
    # emulate multi-variable input
    if isinstance(list(hists)[0], od.Process):
        assert isinstance(variable_insts, list)
        assert len(variable_insts) == 1
        hists = {variable_insts[0].name: hists}
        variable_insts = {variable_insts[0].name: variable_insts}

    # make sure all variable lists are 1D
    assert list(variable_insts) == list(hists)
    for var_name, var_insts in variable_insts.items():
        if len(var_insts) != 1:
            raise NotImplementedError(f"multi_var_plot only supports 1D histograms, but got variables {var_insts}")
    variable_insts = {var_name: var_insts[0] for var_name, var_insts in variable_insts.items()}

    # x-range covering all variables (a single variable keeps its default, including variable settings)
    if len(variable_insts) > 1:
        x_min = min(var_inst.x("x_min", var_inst.x_min) for var_inst in variable_insts.values())
        x_max = max(var_inst.x("x_max", var_inst.x_max) for var_inst in variable_insts.values())
        style_config = law.util.merge_dicts(
            {"ax_cfg": {"xlim": (x_min, x_max)}},
            (style_config or {}),
            deep=True,
        )

    return plot_multi_hists(
        hists=hists,
        config_inst=config_inst,
        category_inst=category_inst,
        variable_insts=variable_insts,
        group_labels={var_name: var_inst.x_title_short for var_name, var_inst in variable_insts.items()},
        shift_insts=shift_insts,
        style_config=style_config,
        variable_settings=variable_settings,
        **kwargs,
    )


@supports_multi_category
def multi_cat_plot(
    hists: dict[od.Process, hist.Hist] | dict[str, dict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: od.Category | list[od.Category],
    variable_insts: list[od.Variable],
    shift_insts: list[od.Shift],
    style_config: dict | None = None,
    variable_settings: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """
    Create a plot overlaying the distributions of a variable in several categories, one line per category and
    process. Example:
    claw run cf.PlotVariables1D --multi-category \
        --plot-function hbw.plotting.multi.multi_cat_plot \
        --variables mli_mll,mli_n_jet --processes ddl4 \
        --categories "{sr,dycr,ttcr}__{2e,2mu,emu}__{resolved__1b,resolved__2b,boosted}"

    Categories are split into their root categories (``root_cats`` aux, or the name split at "__", see
    hbw.plotting.unrolled.get_category_tokens). Root categories shared by all categories form the category label in
    the top left, the others (their *label_short*) label the categories in the legend. The ratio panel shows each
    line divided by the same process in the first category. See plot_multi_hists for the supported options and
    known issues.
    """
    # emulate multi-category input
    if isinstance(list(hists)[0], od.Process):
        category_inst = law.util.make_list(category_inst)
        assert len(category_inst) == 1
        hists = {category_inst[0].name: hists}

    # prepare objects
    category_insts = law.util.make_list(category_inst)
    assert [cat_inst.name for cat_inst in category_insts] == list(hists)
    if len(variable_insts) != 1:
        raise NotImplementedError(f"multi_cat_plot only supports 1D histograms, but got variables {variable_insts}")
    variable_inst = variable_insts[0]

    # common label shown in the top left, varying labels per category shown in the legend
    common_label, labels = get_category_labels(config_inst, category_insts)

    return plot_multi_hists(
        hists=hists,
        config_inst=config_inst,
        # dummy category describing all categories
        category_inst=od.Category(name="multi", id="+", label=common_label),
        variable_insts={cat_inst.name: variable_inst for cat_inst in category_insts},
        group_labels={
            cat_inst.name: ", ".join(label for label in _labels if label)
            for cat_inst, _labels in zip(category_insts, labels)
        },
        shift_insts=shift_insts,
        style_config=style_config,
        variable_settings=variable_settings,
        **kwargs,
    )


def plot_multi_hists(
    hists: dict[str, dict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: dict[str, od.Variable],
    group_labels: dict[str, str],
    shift_insts: list[od.Shift] | None,
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool | None = False,
    yscale: str | None = "",
    hide_stat_errors: bool | None = None,
    legend_title: str | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    skip_ratio: bool = False,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """
    Shared implementation of multi_var_plot and multi_cat_plot: draw one line per group (category or variable) in
    *hists* (``{group_name: {proc_inst: h}}``) and process. The variable of each group is given by *variable_insts*,
    its legend label by *group_labels*, and *category_inst* is used for the label in the top left.

    Supported options (task parameters or --general-settings), applied in the same order as in columnflow's
    plot_variable_stack:
    - *process_settings* (color1, label, scale, hide_stat_errors, ...) and *variable_settings* (rebin, overflow,
      underflow, slice, ...), the latter per group; numeric process scales are applied to each group, while
      scale=stack is computed per group (the label shows the factor of the first group)
    - *remove_negative*: set negative bin contents to zero
    - *density*: divide by the bin width
    - *shape_norm*: normalize each line to its integral
    - *yscale*: y-axis scale, defaults to the log_y of the first variable
    - *hide_stat_errors*: hide the statistical errors (can be overwritten per process via process settings)
    - *legend_title*: defaults to the process label when only one process is plotted
    - *skip_ratio*: hide the ratio panel, showing each line divided by the same process in the first group (the
      shape ratio when *shape_norm* is set); it is skipped automatically when the binnings of the groups differ

    Lines are colored per group and styled per process; with a single group, they are colored per process (using
    its color1, if set) instead. The style config is built from the config, category and first variable, and the
    passed *style_config* overwrites all defaults.

    Known issues and limitations:
    - Process scaling: columnflow's apply_process_scaling appends the scale factor (e.g. "$\\times$10") to the
      process label unless it contains a "__SCALE__" placeholder. As the process instances are shared between
      groups and the scaling is applied per group, the factor used to be appended once per group, leading to very
      long legend entries; the label after the first scaling is kept for all groups to prevent this.
    - scale=stack: the scale factor is computed per group from the stack of that group, so lines of the same
      process can be scaled differently, while the label only shows the factor of the first group.
    - Legend size: there is one legend entry per group and process, so the legend grows quickly with the number
      of categories/variables and processes; use e.g. legend_cfg {"ncols": 2} via a custom style config.
    - hide_stat_errors: error bars are suppressed via yerr=False, as error_type="none" in columnflow's draw_hist
      drops yerr and mplhep then draws its default errors from the variances.
    - Ratio panel: only drawn when all groups share the same binning (after the variable settings), which for
      multi_var_plot is usually not the case; zero bins of the reference are not drawn.
    """
    group_names = list(hists)
    first_var_inst = variable_insts[group_names[0]]

    # remove the shift axis, selecting the nominal shift if there are several
    select_shift = "nominal" if shift_insts and len(shift_insts) > 1 else None
    hists = {
        group_name: remove_residual_axis(proc_hists, "shift", select_value=select_shift)
        for group_name, proc_hists in hists.items()
    }

    # process-based settings (styles and attributes), process instances are shared between groups
    proc_insts = list(dict.fromkeys(proc_inst for proc_hists in hists.values() for proc_inst in proc_hists))
    _, process_style_config = apply_process_settings({proc_inst: None for proc_inst in proc_insts}, process_settings)

    # per group: variable-based settings (rebinning, slicing, flow handling), negative contributions, scaling, density
    variable_style_config = {}
    scaled_labels = {}
    for group_name, proc_hists in hists.items():
        var_inst = variable_insts[group_name]
        proc_hists, _variable_style_config = apply_variable_settings(dict(proc_hists), [var_inst], variable_settings)
        variable_style_config.setdefault(var_inst, _variable_style_config[var_inst])
        if kwargs.get("remove_negative", None):
            proc_hists = remove_negative_contributions(proc_hists)
        proc_hists = apply_process_scaling(proc_hists)
        # the scaling appends the scale factor to the (shared) process labels on each call, so keep the label of the
        # first scaling only
        for proc_inst in proc_hists:
            proc_inst.label = scaled_labels.setdefault(proc_inst, proc_inst.label)
        if density:
            proc_hists = apply_density(proc_hists, density)
        hists[group_name] = proc_hists

    # the ratio requires the same binning in all groups
    if not skip_ratio:
        all_edges = [h.axes[0].edges for proc_hists in hists.values() for h in proc_hists.values()]
        if not all(len(edges) == len(all_edges[0]) and np.allclose(edges, all_edges[0]) for edges in all_edges):
            logger.warning("skipping the ratio panel of the multi plot, as the binnings of the histograms differ")
            skip_ratio = True

    # norm of each line
    get_norm = lambda h: sum(h.values()) if shape_norm else 1

    # create plot instructions for plot_all
    plot_config = {}
    for i, (group_name, proc_hists) in enumerate(hists.items()):
        for proc_inst, h in proc_hists.items():
            j = proc_insts.index(proc_inst)

            # legend label, the process is shown in the legend title when there is only one
            group_label = group_labels[group_name]
            if len(hists) == 1 or not group_label:
                label = proc_inst.label
            elif len(proc_insts) == 1:
                label = group_label
            else:
                label = f"{proc_inst.label}: {group_label}"

            # color per group and line style per process, or color per process for a single group
            if len(hists) > 1:
                color = get_color_from_sequence(i)
                linestyle = get_line_style_from_sequence(j)
            else:
                color = proc_inst.color1 or get_color_from_sequence(j)
                linestyle = "solid"

            # per-process setting overrides task parameter
            proc_hide_stat_errors = get_attr_or_aux(proc_inst, "hide_stat_errors", hide_stat_errors)
            norm = get_norm(h)
            plot_config[f"multi_plot__{proc_inst.name}__{group_name}"] = plot_cfg = {
                "method": "draw_hist",
                "hist": h,
                "kwargs": dict(
                    norm=norm,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    error_type="variance",
                    linewidth=1.5,
                ),
            }
            # suppress error bars by overriding yerr (error_type "none" would fall back to mplhep's default errors)
            if proc_hide_stat_errors:
                plot_cfg["kwargs"]["yerr"] = False

            # ratio to the same process in the first group containing it
            if not skip_ratio:
                h_ref = next(_proc_hists[proc_inst] for _proc_hists in hists.values() if proc_inst in _proc_hists)
                ref_values = h_ref.values()
                ref_norm = get_norm(h_ref) or 1
                plot_cfg["ratio_kwargs"] = dict(
                    plot_cfg["kwargs"],
                    # zero reference bins are not drawn
                    norm=np.where(ref_values == 0, np.nan, ref_values * norm / ref_norm),
                    label=None,
                )

    # legend title defaults to the process label for a single process
    if not legend_title and len(proc_insts) == 1 and len(hists) > 1:
        legend_title = proc_insts[0].label

    # combine the style config
    default_style_config = prepare_style_config(
        config_inst=config_inst,
        variable_inst=first_var_inst,
        category_inst=category_inst,
        density=density,
        shape_norm=shape_norm,
        yscale=yscale,
    )
    plot_style_config = {
        "annotate_cfg": {"xy": (0.035, 0.95), "xycoords": "axes fraction"},
        "cms_label_cfg": {"loc": 0},
        "legend_cfg": {"loc": "upper right", "fontsize": 18},
        "rax_cfg": {"ylabel": "Ratio", "ylim": (0.5, 1.5)},
    }
    if legend_title:
        plot_style_config["legend_cfg"]["title"] = legend_title
    if shape_norm:
        plot_style_config["ax_cfg"] = {"ylabel": "Normalized entries"}
    style_config = law.util.merge_dicts(
        default_style_config,
        plot_style_config,
        process_style_config,
        # variable styles (e.g. x ticks) only apply when all groups share the variable
        (variable_style_config[first_var_inst] if len(variable_style_config) == 1 else {}),
        (style_config or {}),
        deep=True,
    )

    return plot_all(plot_config, style_config, skip_ratio=skip_ratio, **kwargs)
