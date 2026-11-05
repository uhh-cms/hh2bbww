# coding: utf-8

"""
Initially taken from bbtautau analysis:
https://github.com/uhh-cms/hh2bbtautau/blob/4a0420ce3645568dcd1f3391fd50af09cc15979c/hbt/plotting/multi.py
"""

from __future__ import annotations

import math

import law
import order as od

from law.util import DotDict

from columnflow.plotting import supports_multi_variable, supports_multi_category
from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_functions_1d import plot_variable_stack
from columnflow.plotting.plot_util import (
    prepare_style_config, remove_residual_axis, use_flow_bins, apply_variable_settings,
)
from columnflow.util import maybe_import
from columnflow.types import TYPE_CHECKING, Any, Hashable

np = maybe_import("numpy")
if TYPE_CHECKING:
    hist = maybe_import("hist")
    plt = maybe_import("matplotlib.pyplot")


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
    # emulate multi-variable input
    if isinstance(list(hists)[0], od.Process):
        assert isinstance(variable_insts, list)
        assert len(variable_insts) == 1
        hists = {variable_insts[0].name: hists}
        variable_insts = {variable_insts[0].name: variable_insts}

    # make sure all variable lists are 1D
    for var_name, var_insts in variable_insts.items():
        if len(var_insts) != 1:
            raise NotImplementedError(f"multi_plot only supports 1D histograms, but got variables {var_insts}")
        variable_insts[var_name] = var_insts[0]

    # remove shift axis
    hists = {var_name: remove_residual_axis(proc_hists, "shift") for var_name, proc_hists in hists.items()}

    # unique process instances
    unique_proc_insts = set(sum((list(v.keys()) for v in hists.values()), []))

    # helper to create legend labels
    def create_legend_label(proc_inst: od.Process, var_inst: od.Variable) -> str:
        return var_inst.x_title_short if len(unique_proc_insts) == 1 else f"{proc_inst.label}: {var_inst.x_title_short}"

    # create plot instructions for plot_all
    plot_config = {
        **{
            f"multi_var_plot__{proc_inst.name}__{var_name}": {
                "method": "draw_hist",
                "hist": move_flow(h, variable_insts[var_name]),
                "kwargs": dict(
                    label=create_legend_label(proc_inst, variable_insts[var_name]),
                    color=config_inst.x.get_color_from_sequence(i),
                    linestyle=config_inst.x.get_line_style_from_sequence(j),
                    error_type="variance",
                    linewidth=1.5,
                ),
            }
            for i, (var_name, proc_hists) in enumerate(hists.items())
            for j, (proc_inst, h) in enumerate(proc_hists.items())
        },
    }

    # combine the style config
    style_config = law.util.merge_dicts(
        # inferred from config, category, variable
        prepare_style_config(
            config_inst=config_inst,
            category_inst=category_inst,
            variable_inst=list(variable_insts.values())[0],
        ),
        # passed styles
        style_config,
        (variable_settings or {}),
        # overwrite with hardcoded settings
        {
            "annotate_cfg": {"xy": (0.035, 0.95)},
            "cms_label_cfg": {"loc": 0},
            "legend_cfg": {"loc": "upper right", "fontsize": 18},
        },
        deep=True,
    )

    # update kwargs for plot_all
    kwargs = kwargs | {
        "skip_ratio": True,
        "skip_legend": False,
    }

    return plot_all(plot_config, style_config, **kwargs)


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
    Create a multi-category plot. Example:
    claw run cf.PlotVariables1D --multi-category --plot-function hbw.plotting.multi.multi_cat_plot \
        --variables jet1_pt --categories sr__resolved__1b,sr__resolved__2b,sr__boosted --processes tt
    """
    # emulate multi-category input
    if isinstance(list(hists)[0], od.Process):
        category_inst = law.util.make_list(category_inst)
        assert len(category_inst) == 1
        hists = {category_inst[0].name: hists}

    # prepare objects
    category_insts = {cat_inst.name: cat_inst for cat_inst in law.util.make_list(category_inst)}
    assert all(cat_name in hists for cat_name in category_insts)
    assert len(variable_insts) == 1
    variable_inst = variable_insts[0]

    # remove shift axis
    hists = {cat_name: remove_residual_axis(proc_hists, "shift") for cat_name, proc_hists in hists.items()}

    # unique process instances
    unique_proc_insts = set(sum((list(v.keys()) for v in hists.values()), []))

    # helper to create legend labels
    def create_legend_label(proc_inst: od.Process, cat_inst: od.Category) -> str:
        return cat_inst.label_short if len(unique_proc_insts) == 1 else f"{proc_inst.label}: {cat_inst.label_short}"

    # create plot instructions for plot_all
    plot_config = {
        **{
            f"multi_cat_plot__{proc_inst.name}__{cat_name}": {
                "method": "draw_hist",
                "hist": move_flow(h, variable_inst),
                "kwargs": dict(
                    label=create_legend_label(proc_inst, category_insts[cat_name]),
                    color=get_color_from_sequence(i),
                    linestyle=get_line_style_from_sequence(j),
                    error_type="variance",
                    linewidth=1.5,
                ),
            }
            for i, (cat_name, proc_hists) in enumerate(hists.items())
            for j, (proc_inst, h) in enumerate(proc_hists.items())
        },
    }

    # combine the style config
    style_config = law.util.merge_dicts(
        # inferred from config, category, variable
        prepare_style_config(
            config_inst=config_inst,
            variable_inst=variable_inst,
        ),
        # passed styles
        style_config,
        (variable_settings or {}),
        # overwrite with hardcoded settings
        {
            "cms_label_cfg": {"loc": 0},
            "legend_cfg": {"loc": "upper right", "fontsize": 18},
        },
        deep=True,
    )

    # update kwargs for plot_all
    kwargs = kwargs | {
        "skip_ratio": True,
        "skip_legend": False,
    }

    return plot_all(plot_config, style_config, **kwargs)


@supports_multi_category
def multi_cat_unrolled_plot(
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
    Create a stack plot with the bins of all categories unrolled next to each other on a single x-axis. Example:
    claw run cf.PlotVariables1D --multi-category \
        --plot-function hbw.plotting.multi.multi_cat_unrolled_plot \
        --variables mll --processes ... \
        --categories sr__2e__resolved__1b,sr__2e__resolved__2b,sr__2mu__resolved__1b,sr__2mu__resolved__2b,\
sr__emu__resolved__1b,sr__emu__resolved__2b

    The categories are unrolled in the order in which the task passes them, which is alphabetical since
    PlotVariables1D sorts the --categories (CategoriesMixin.sort_categories). Variable settings (rebinning, flow
    bins, ...) are applied per category before unrolling; everything else (process settings, blinding, ratio, ...)
    is handled by columnflow's plot_variable_stack on the unrolled histograms. Note that density scaling therefore
    refers to the unrolled bins with unit width.

    Categories are split into their root categories (``root_cats`` aux, or the name split at "__"). Root
    categories shared by all categories form the category label in the top left (sr and resolved above). Among
    the remaining ones, the one changing least often between consecutive categories defines groups (the lepton
    channel above), the others label each category block (the b-tag category above). Labels are taken from the
    root categories' *label_short*.

    The labels are placed in the whitespace above the histograms, so their position follows the
    *whitespace_fraction* (default here: 0.45), which can be changed via
    --general-settings whitespace_fraction=0.5
    The legend gets as many columns as needed for at most 4 entries per column, and the figure width defaults to
    10 inches plus 3 inches per additional category (height 10, at most 25 inches wide); both can be overwritten
    via the style config (legend_cfg, subplots_cfg).
    """
    # emulate multi-category input
    if isinstance(list(hists)[0], od.Process):
        category_inst = law.util.make_list(category_inst)
        assert len(category_inst) == 1
        hists = {category_inst[0].name: hists}

    category_insts = law.util.make_list(category_inst)
    assert [cat_inst.name for cat_inst in category_insts] == list(hists)
    if len(variable_insts) != 1:
        raise NotImplementedError(f"multi_cat_unrolled_plot only supports 1D histograms, but got {variable_insts}")
    variable_inst = variable_insts[0]

    # variable settings (rebinning, flow bins, ...) need to be applied per category before unrolling
    flat_hists = {
        (cat_name, proc_inst): h
        for cat_name, proc_hists in hists.items()
        for proc_inst, h in proc_hists.items()
    }
    flat_hists, _ = apply_variable_settings(flat_hists, variable_insts, variable_settings)
    hists = {
        cat_name: {
            proc_inst: flat_hists[(cat_name, proc_inst)]
            for proc_inst in proc_hists
        }
        for cat_name, proc_hists in hists.items()
    }

    # unroll into a single axis with integer edges
    unrolled_name = f"{variable_inst.name}_unrolled"
    hists, n_bins = unroll_hists(hists, variable_inst.name, axis_name=unrolled_name)

    # dummy variable and category describing the unrolled histograms
    unrolled_variable_inst = od.Variable(
        name=unrolled_name,
        binning=list(range(sum(n_bins) + 1)),
        x_title=variable_inst.x_title,
        y_title=variable_inst.y_title,
        log_y=variable_inst.log_y,
        discrete_x=True,
    )
    common_label, blocks = build_unroll_labels(config_inst, category_insts, n_bins)
    unrolled_category_inst = od.Category(name="unrolled", id="+", label=common_label)

    # make room for the category labels and widen the figure with the number of categories (default CMS style: 10x10)
    kwargs["whitespace_fraction"] = float(kwargs.get("whitespace_fraction", 0.45))
    style_config = law.util.merge_dicts(
        {
            "legend_cfg": {"fontsize": 16},
            "subplots_cfg": {"figsize": (min(10.0 + 3.0 * (len(category_insts) - 1), 40.0), 10.0)},
        },
        (style_config or {}),
        deep=True,
    )
    # legend columns are only chosen automatically when not configured otherwise
    auto_legend_cols = not any(key == "ncols" or key.startswith("cf_") for key in style_config["legend_cfg"])
    # keep the absolute left/right margins of plot_all's defaults (0.15, 0.95 at width 10) for any figure width
    width = style_config["subplots_cfg"]["figsize"][0]
    gridspec_cfg = style_config.setdefault("gridspec_cfg", {})
    gridspec_cfg.setdefault("left", 1.5 / width)
    gridspec_cfg.setdefault("right", 1 - 0.5 / width)

    fig, axs = plot_variable_stack(
        hists,
        config_inst,
        unrolled_category_inst,
        [unrolled_variable_inst],
        shift_insts,
        style_config=style_config,
        variable_settings=None,
        **kwargs,
    )

    if auto_legend_cols:
        set_legend_ncols(axs[0], style_config["legend_cfg"], max_entries_per_col=4)
    draw_unroll_labels(axs, blocks, y_line=1 - kwargs["whitespace_fraction"] + 0.02)

    return fig, axs


def set_legend_ncols(ax: plt.Axes, legend_cfg: dict, max_entries_per_col: int = 4) -> None:
    """
    Recreate the legend drawn by plot_all on *ax* with as many columns as needed for at most
    *max_entries_per_col* entries per column, keeping its entries and the other *legend_cfg* settings.
    """
    legend = ax.get_legend()
    if not legend:
        return

    # original handles, with the labels as shown in the legend (placeholders already resolved)
    handles, _ = ax.get_legend_handles_labels()
    labels = [text.get_text() for text in legend.get_texts()]
    if len(handles) != len(labels):
        return

    legend_kwargs = {"loc": "upper right", **legend_cfg}
    legend_kwargs["ncols"] = math.ceil(len(handles) / max_entries_per_col)
    ax.legend(handles, labels, **legend_kwargs)


def unroll_hists(
    hists: dict[str, dict[od.Process, hist.Hist]],
    var_name: str,
    axis_name: str | None = None,
) -> tuple[dict[od.Process, hist.Hist], list[int]]:
    """
    Concatenate the *var_name* axis of the histograms in *hists* (``{cat_name: {proc_inst: h}}``) per process,
    in the order of the categories. All other axes (e.g. "shift") are kept and must be identical between categories.
    Processes missing in a category are filled with zeros there. The unrolled axis has integer edges and no flow
    bins, and is named *axis_name* (default: *var_name*). Returns the unrolled histograms and the number of bins
    per category.
    """
    import hist

    # number of bins per category
    n_bins = []
    for cat_name, proc_hists in hists.items():
        if not proc_hists:
            raise ValueError(f"no histograms to unroll in category '{cat_name}'")
        n_bins.append(len(list(proc_hists.values())[0].axes[var_name]))

    unrolled_axis = hist.axis.Variable(
        list(range(sum(n_bins) + 1)),
        name=axis_name or var_name,
        underflow=False,
        overflow=False,
    )

    # unique processes in order of first appearance
    proc_insts = list(dict.fromkeys(proc_inst for proc_hists in hists.values() for proc_inst in proc_hists))

    unrolled_hists = {}
    for proc_inst in proc_insts:
        h_ref = next(proc_hists[proc_inst] for proc_hists in hists.values() if proc_inst in proc_hists)
        var_index = h_ref.axes.name.index(var_name)

        views = []
        for (cat_name, proc_hists), n in zip(hists.items(), n_bins):
            if proc_inst not in proc_hists:
                shape = list(h_ref.view().shape)
                shape[var_index] = n
                views.append(np.zeros(shape, dtype=h_ref.view().dtype))
                continue

            h = proc_hists[proc_inst]
            if h.axes.name != h_ref.axes.name or any(
                ax != ax_ref
                for ax, ax_ref in zip(h.axes, h_ref.axes)
                if ax.name != var_name
            ):
                raise ValueError(
                    f"cannot unroll histograms of process '{proc_inst.name}': axes in category '{cat_name}' "
                    f"differ from the ones in other categories",
                )
            views.append(h.view())

        h_out = hist.Hist(
            *[unrolled_axis if ax.name == var_name else ax for ax in h_ref.axes],
            storage=h_ref.storage_type(),
        )
        h_out.view()[...] = np.concatenate(views, axis=var_index)
        unrolled_hists[proc_inst] = h_out

    return unrolled_hists, n_bins


def get_category_tokens(category_insts: list[od.Category]) -> list[dict[Hashable, str | None]]:
    """
    Split each category into the names of its root categories, using the ``root_cats`` aux of categories built
    via create_category_combinations and the category name split at "__" otherwise. The keys are aligned over
    all categories; root categories missing in a category are set to None.
    """
    tokens = [
        dict(cat_inst.x("root_cats", None) or enumerate(cat_inst.name.split("__")))
        for cat_inst in category_insts
    ]
    keys = list(dict.fromkeys(key for _tokens in tokens for key in _tokens))
    return [{key: _tokens.get(key) for key in keys} for _tokens in tokens]


def build_unroll_labels(
    config_inst: od.Config,
    category_insts: list[od.Category],
    n_bins: list[int],
) -> tuple[str, list[DotDict]]:
    """
    Build the labels of the unrolled plot from the root categories of *category_insts* (see
    get_category_tokens). Returns the label common to all categories and one block per category with its first
    bin, number of bins, label and group label (None when there are no groups).
    """
    def get_label(token: str | None) -> str:
        if token is None:
            return ""
        cat_inst = config_inst.get_category(token, default=None)
        return (cat_inst.label_short if cat_inst else token).replace("\n", " ")

    tokens = get_category_tokens(category_insts)
    keys = list(tokens[0])
    varying_keys = [key for key in keys if len({_tokens[key] for _tokens in tokens}) > 1]
    common_label = "\n".join(
        get_label(tokens[0][key])
        for key in keys
        if key not in varying_keys and tokens[0][key] is not None
    )

    # the root category changing least often between consecutive categories defines the groups
    group_key = None
    if len(varying_keys) > 1:
        n_changes = lambda key: sum(t1[key] != t2[key] for t1, t2 in zip(tokens[:-1], tokens[1:]))
        group_key = min(varying_keys, key=n_changes)
    label_keys = [key for key in varying_keys if key != group_key]

    blocks = []
    start = 0
    for _tokens, n in zip(tokens, n_bins):
        blocks.append(DotDict(
            start=start,
            n_bins=n,
            # one line per root category (empty if missing), so that lines are aligned between blocks
            label="\n".join(get_label(_tokens[key]) for key in label_keys),
            group=get_label(_tokens[group_key]) if group_key is not None else None,
        ))
        start += n

    return common_label, blocks


def draw_unroll_labels(
    axs: tuple[plt.Axes, ...],
    blocks: list[DotDict],
    y_line: float,
) -> None:
    """
    Draw separators and labels of the unrolled category *blocks* (see build_unroll_labels). Vertical positions
    are given as fractions of the main axis height, so they are independent of the y-scale: block labels are
    placed above a horizontal line at *y_line*, group labels above them. Labels are shrunk (and as a last resort
    rotated) to fit the width of their block or group.
    """
    ax = axs[0]
    trafo = ax.get_xaxis_transform()
    renderer = ax.figure.canvas.get_renderer()
    line_kwargs = {"color": "grey", "linewidth": 2.5}
    has_groups = blocks[0].group is not None

    # helper to draw a label centered between bins *start* and *end* that fits their width, returns its top
    def draw_label(start: float, end: float, y: float, text: str, fontsize: int) -> float:
        label = ax.text(0.5 * (start + end), y, text, transform=trafo, fontsize=fontsize, ha="center", va="bottom")
        width = np.diff(ax.transData.transform([(start, 0), (end, 0)])[:, 0])[0]
        while label.get_window_extent(renderer).width > 0.95 * width and label.get_fontsize() > 10:
            label.set_fontsize(label.get_fontsize() - 1)
        if label.get_window_extent(renderer).width > 0.95 * width:
            label.set_rotation(90)
        return ax.transAxes.inverted().transform((0, label.get_window_extent(renderer).y1))[1]

    # block labels above a horizontal line
    n_total = blocks[-1].start + blocks[-1].n_bins
    ax.plot([0, n_total], [y_line, y_line], transform=trafo, **line_kwargs)
    y_group = max(
        draw_label(block.start, block.start + block.n_bins, y_line + 0.01, block.label, 16)
        for block in blocks
    ) + 0.02

    # group labels centered on consecutive blocks of the same group
    y_group_top = y_group
    if has_groups:
        group_start = 0
        for i, block in enumerate(blocks):
            if i + 1 < len(blocks) and blocks[i + 1].group == block.group:
                continue
            group_end = block.start + block.n_bins
            y_group_top = max(y_group_top, draw_label(group_start, group_end, y_group, block.group, 20))
            group_start = group_end

    # separators, solid between groups and dashed within groups
    for prev_block, block in zip(blocks[:-1], blocks[1:]):
        new_group = has_groups and block.group != prev_block.group
        sep_kwargs = line_kwargs if new_group else {**line_kwargs, "linestyle": "--"}
        ax.axvline(block.start, ymax=y_group_top if new_group else y_line, **sep_kwargs)
        for rax in axs[1:]:
            rax.axvline(block.start, **sep_kwargs)


# helper to move flow bins into first/last ones
def move_flow(h: hist.Hist, variable_inst: od.Variable) -> hist.Hist:
    overflow = variable_inst.x("overflow", False)
    underflow = variable_inst.x("underflow", False)
    if overflow or underflow:
        h = use_flow_bins(h, variable_inst.name, underflow=underflow, overflow=overflow)
    return h
