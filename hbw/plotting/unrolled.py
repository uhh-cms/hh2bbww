# coding: utf-8

"""
Stack plots with the bins of several categories or variables unrolled next to each other on a single x-axis.
"""

from __future__ import annotations

import math

import law
import order as od

from law.util import DotDict

from columnflow.plotting import supports_multi_variable, supports_multi_category
from columnflow.plotting.plot_functions_1d import plot_variable_stack
from columnflow.plotting.plot_util import apply_variable_settings, get_position
from columnflow.util import maybe_import
from columnflow.types import TYPE_CHECKING, Any, Hashable, Sequence

np = maybe_import("numpy")
if TYPE_CHECKING:
    hist = maybe_import("hist")
    plt = maybe_import("matplotlib.pyplot")


@supports_multi_category
def multi_cat_unrolled_plot(
    hists: dict[od.Process, hist.Hist] | dict[str, dict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: od.Category | list[od.Category],
    variable_insts: list[od.Variable],
    shift_insts: list[od.Shift],
    style_config: dict | None = None,
    variable_settings: dict[str, Any] | None = None,
    unroll_category_order: Sequence[str] | str | None = None,
    unroll_level_order: Sequence[Hashable] | Hashable | None = None,
    unroll_x_labels: bool | None = None,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """
    Create a stack plot with the bins of all categories unrolled next to each other on a single x-axis. Example:
    claw run cf.PlotVariables1D --multi-category \
        --plot-function hbw.plotting.unrolled.multi_cat_unrolled_plot \
        --variables mli_mll,mli_n_jet --processes ddl4 \
        --categories "{sr,dycr,ttcr}__{2e,2mu,emu}__{resolved__1b,resolved__2b,boosted}"

    Variable settings (rebinning, flow bins, ...) are applied per category before unrolling; everything else
    (process settings, blinding, ratio, ...) is handled by columnflow's plot_variable_stack on the unrolled
    histograms. Note that density scaling therefore refers to the unrolled bins with unit width.

    Categories are split into their root categories (``root_cats`` aux, or the name split at "__"). Root
    categories shared by all categories form the category label in the top left. The others define levels, from
    the outermost to the innermost one; each level gets a row of labels, merged over consecutive categories, and
    separators that are solid, dashed and dotted from the outermost to the innermost level. Labels are taken from
    the root categories' *label_short*.

    The categories are re-sorted (instead of the alphabetical order of the task) level by level, configured by:
    - *unroll_level_order*: keys of the root categories (e.g. "main", "lep", "jet", "b") from the outermost to
      the innermost level; keys not listed follow in the order of the category names (the default)
    - *unroll_category_order*: preferred order of root category names on each level, e.g.
      (sr, dycr, ttcr, resolved, boosted); names not listed follow in the order of the task, missing root
      categories (e.g. the b-tag category of boosted categories) come last
    Both are taken from the general settings (e.g. --general-settings "unroll_level_order=jet;lep" or a
    general_settings_groups entry), falling back to the config auxiliaries default_unroll_level_order and
    default_unroll_category_order.

    The labels are placed in the whitespace above the histograms, below the legend. The *whitespace_fraction*
    (at least 0.3 by default, can be raised via --general-settings whitespace_fraction=0.5) is increased
    automatically until the labels fit between the histograms and the legend (and other texts at the top of the
    plot), unless the y-range is fixed via the ax_cfg of the style config.
    By default, the x-axis shows the bin numbers of the unrolled histogram. With *unroll_x_labels*
    (--general-settings unroll_x_labels=True, falling back to the config auxiliary default_unroll_x_labels), it
    shows the original binning of each category instead (bin edges, or bin centers for discrete variables), with
    labels thinned out when they would overlap; between two categories, the lower edge of the next one is shown.
    All bins are drawn with the same width, so non-uniform binnings are shown with equal-width bins labeled with
    their true edges (note that density=True therefore normalizes per bin, not per unit of the variable).
    The legend gets as many columns as needed for at most 3 entries per column, and the figure width defaults to
    10 inches plus 3 inches per additional category (height 10, at most 40 inches wide); both can be overwritten
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

    # re-sort the categories level by level (general settings first, then config defaults)
    category_insts, level_keys = order_categories(
        category_insts,
        category_order=unroll_category_order or config_inst.x("default_unroll_category_order", ()),
        level_order=unroll_level_order or config_inst.x("default_unroll_level_order", ()),
    )
    hists = {cat_inst.name: hists[cat_inst.name] for cat_inst in category_insts}

    # variable settings (rebinning, flow bins, ...) need to be applied per category before unrolling
    hists = {
        cat_name: apply_variable_settings(dict(proc_hists), variable_insts, variable_settings)[0]
        for cat_name, proc_hists in hists.items()
    }

    # labels of the categories, common to all of them or per category and level
    n_bins = [len(edges) - 1 for edges in get_block_edges(hists, [variable_inst] * len(hists))]
    common_label, blocks = build_unroll_labels(config_inst, category_insts, n_bins, level_keys=level_keys)

    return plot_unrolled_stack(
        hists,
        config_inst,
        # dummy category describing the unrolled histograms
        od.Category(name="unrolled", id="+", label=common_label),
        [variable_inst] * len(hists),
        blocks,
        shift_insts,
        unrolled_variable_inst=od.Variable(
            name=f"{variable_inst.name}_unrolled",
            x_title=variable_inst.x_title,
            y_title=variable_inst.y_title,
            log_y=variable_inst.log_y,
        ),
        style_config=style_config,
        unroll_x_labels=unroll_x_labels,
        **kwargs,
    )


@supports_multi_variable
def multi_var_unrolled_plot(
    hists: dict[od.Process, hist.Hist] | dict[str, dict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable] | dict[str, list[od.Variable]],
    shift_insts: list[od.Shift],
    style_config: dict | None = None,
    variable_settings: dict[str, Any] | None = None,
    unroll_x_labels: bool | None = None,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """
    Create a stack plot with the bins of all variables unrolled next to each other on a single x-axis. Example:
    claw run cf.PlotVariables1D --multi-variable \
        --plot-function hbw.plotting.unrolled.multi_var_unrolled_plot \
        --processes ddl4 --categories sr \
        --variables \
            "mli_{mbb,mll,mllMET,mbbllMET,n_jet,ht,ptll,b1_pt,b2_pt,b1_discrete_b_score,b2_discrete_b_score,j1_pt,j1_eta,bb_pt,lep_pt,lep2_pt,ll_pt,maxdr_jj,dr_bb,dr_ll,dr_ll_bb,min_dr_llbb,met_pt,vbf_tag,mixed_channel}"

    This is the multi-variable counterpart of multi_cat_unrolled_plot, see there for details on the styling
    (whitespace, legend columns, figure width, *unroll_x_labels*). Variable settings are applied per variable
    before unrolling. The variables are shown in the order of the task, separated by dashed lines and labeled with
    their *x_title_short*; the category is shown as usual in the top left. Unlike in multi_cat_unrolled_plot, the
    x-axis shows the original binning of each variable by default (without an x-axis title, as the variables are
    labeled above the histograms); with --general-settings unroll_x_labels=False, it shows the bin numbers of the
    unrolled histogram with the x-axis title "Bin" instead. The y-axis title and log scale are taken from the first
    variable.
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
            raise NotImplementedError(
                f"multi_var_unrolled_plot only supports 1D histograms, but got variables {var_insts}",
            )
    variable_insts = [var_insts[0] for var_insts in variable_insts.values()]

    # variable settings (rebinning, flow bins, ...) need to be applied per variable before unrolling
    hists = {
        var_inst.name: apply_variable_settings(dict(hists[var_inst.name]), [var_inst], variable_settings)[0]
        for var_inst in variable_insts
    }

    # show the original binning of the variables by default, the bin numbers of the unrolled axis are not meaningful
    if unroll_x_labels is None:
        unroll_x_labels = True

    # one block and level of labels per variable
    n_bins = [len(edges) - 1 for edges in get_block_edges(hists, variable_insts)]
    blocks = make_unroll_blocks(
        paths=[(var_inst.name,) for var_inst in variable_insts],
        labels=[[var_inst.x_title_short] for var_inst in variable_insts],
        n_bins=n_bins,
    )

    return plot_unrolled_stack(
        hists,
        config_inst,
        category_inst,
        variable_insts,
        blocks,
        shift_insts,
        unrolled_variable_inst=od.Variable(
            name="variables_unrolled",
            x_title="" if unroll_x_labels else "Bin",
            y_title=variable_insts[0].y_title,
            log_y=variable_insts[0].log_y,
        ),
        style_config=style_config,
        unroll_x_labels=unroll_x_labels,
        **kwargs,
    )


def multi_cat_var_unrolled_plot(
    hists: dict[tuple[str, str], dict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: list[od.Category],
    variable_insts: dict[str, list[od.Variable]],
    shift_insts: list[od.Shift],
    style_config: dict | None = None,
    variable_settings: dict[str, Any] | None = None,
    unroll_category_order: Sequence[str] | str | None = None,
    unroll_level_order: Sequence[Hashable] | Hashable | None = None,
    unroll_x_labels: bool | None = None,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """
    Placeholder for a stack plot with the bins of several (category, variable) combinations unrolled next to each
    other on a single x-axis, e.g. to show a different variable per category. Not usable yet.

    Planned behavior: *hists* maps (category name, variable name) pairs to process histograms. Each pair becomes
    one block of plot_unrolled_stack, with the variable settings applied per variable, the categories ordered and
    labeled as in multi_cat_unrolled_plot, and the variable added as the innermost label level.

    This requires upstream changes in columnflow (columnflow/tasks/plotting.py): PlotBase currently rejects
    --multi-variable together with --multi-category (_check_multi_flags) and only passes hists nested by either
    category or variable, so a combined mode passing ``{(cat_name, var_name): proc_hists}`` (and a matching
    plot function decorator) is needed first.
    """
    raise NotImplementedError(
        "multi_cat_var_unrolled_plot is not implemented yet, as it requires columnflow to support combined "
        "multi-category and multi-variable plotting",
    )


def plot_unrolled_stack(
    hists: dict[str, dict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    blocks: list[DotDict],
    shift_insts: list[od.Shift],
    unrolled_variable_inst: od.Variable,
    style_config: dict | None = None,
    unroll_x_labels: bool | None = None,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """
    Shared implementation of the unrolled plots (multi_cat_unrolled_plot, multi_var_unrolled_plot): unroll the
    histograms of all blocks in *hists* (``{block_name: {proc_inst: h}}``, variable settings already applied, the
    variable of each block given by *variable_insts*) into a single axis, draw them as a stack plot via columnflow's
    plot_variable_stack, and draw the labels and separators of the *blocks* (see make_unroll_blocks).
    *category_inst* is used for the category label in the top left, and *unrolled_variable_inst* defines the name,
    titles and y-scale of the unrolled variable (its binning is set here). See multi_cat_unrolled_plot for the
    styling options.
    """
    # original bin edges per block, shown as x tick labels of the unrolled axis
    edges = get_block_edges(hists, variable_insts)

    # unroll into a single axis with integer edges
    unrolled_name = unrolled_variable_inst.name
    hists, n_bins = unroll_hists(hists, [var_inst.name for var_inst in variable_insts], axis_name=unrolled_name)
    if n_bins != [block.n_bins for block in blocks]:
        raise ValueError(f"number of bins per block {[block.n_bins for block in blocks]} do not match {n_bins}")
    unrolled_variable_inst.binning = list(range(sum(n_bins) + 1))
    unrolled_variable_inst.discrete_x = True

    # make room for the block labels and widen the figure with the number of blocks (default CMS style: 10x10)
    kwargs["whitespace_fraction"] = float(kwargs.get("whitespace_fraction", 0.3))
    style_config = law.util.merge_dicts(
        {
            "legend_cfg": {"fontsize": 16},
            "subplots_cfg": {"figsize": (min(10.0 + 3.0 * (len(blocks) - 1), 40.0), 10.0)},
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
        category_inst,
        [unrolled_variable_inst],
        shift_insts,
        style_config=style_config,
        variable_settings=None,
        **kwargs,
    )

    if auto_legend_cols:
        set_legend_ncols(axs[0], style_config["legend_cfg"], max_entries_per_col=3)

    # optionally show the original binning on the x-axis, unless ticks are configured in the style config
    if unroll_x_labels is None:
        unroll_x_labels = config_inst.x("default_unroll_x_labels", False)
    tick_keys = {"xmajorticks", "xmajorticklabels"}
    if unroll_x_labels and not tick_keys & (set(style_config.get("ax_cfg", {})) | set(style_config.get("rax_cfg", {}))):
        set_unrolled_xticks(
            axs[-1],
            [block.start for block in blocks],
            edges,
            discrete_x=[is_discrete_x(var_inst) for var_inst in variable_insts],
            x_labels=[var_inst.x_labels for var_inst in variable_insts],
        )

    # draw the labels right above the histograms (the height of the label rows only depends on the x-axis)
    whitespace_fraction = kwargs["whitespace_fraction"]
    header_bottom = get_header_bottom(axs[0])
    children = [set(ax.get_children()) for ax in axs]
    labels_top = draw_unroll_labels(axs, blocks, y_line=1 - whitespace_fraction + 0.02)

    # when the labels reach into the legend, increase the whitespace and draw them again
    required_fraction = whitespace_fraction + labels_top + 0.02 - header_bottom
    if required_fraction > whitespace_fraction and "ylim" not in style_config.get("ax_cfg", {}):
        for ax, ax_children in zip(axs, children):
            for artist in set(ax.get_children()) - ax_children:
                artist.remove()
        set_whitespace_fraction(axs[0], whitespace_fraction, required_fraction)
        draw_unroll_labels(axs, blocks, y_line=1 - required_fraction + 0.02)

    return fig, axs


def get_block_edges(
    hists: dict[str, dict[od.Process, hist.Hist]],
    variable_insts: Sequence[od.Variable],
) -> list[np.ndarray]:
    """
    Return the bin edges of the variable axis of each block in *hists* (``{block_name: {proc_inst: h}}``), with
    the variable of each block given by *variable_insts*.
    """
    return [
        next(iter(proc_hists.values())).axes[var_inst.name].edges
        for proc_hists, var_inst in zip(hists.values(), variable_insts)
    ]


def get_unrolled_xticks(
    block_starts: Sequence[int],
    edges: Sequence[Sequence[float]],
    discrete_x: bool | Sequence[bool] = False,
    x_labels: Sequence[Sequence[str] | None] | None = None,
    step: int = 1,
) -> tuple[list[float], list[str]]:
    """
    Return positions and labels of x ticks on the unrolled axis that show the original binning *edges* of each
    block (category or variable), whose bins start at the unrolled bin *block_starts*. Ticks are placed at all bin
    edges, or at the bin centers for *discrete_x* variables (a single flag or one per block). *x_labels* can give
    the bin labels per block (or None), which are used for discrete variables when given for each bin. Only every
    *step*-th tick of each block is labeled, starting with the first one, and no label is closer than *step* ticks
    to the next block. Between two blocks, the lower edge of the next block is shown, and the upper edge is only
    shown for the last block.
    """
    n_blocks = len(edges)
    discrete_xs = [discrete_x] * n_blocks if isinstance(discrete_x, bool) else list(discrete_x)
    x_labels = [None] * n_blocks if x_labels is None else list(x_labels)

    positions, labels = [], []
    for i, (start, block_edges, discrete_x, block_x_labels) in enumerate(
        zip(block_starts, edges, discrete_xs, x_labels),
    ):
        block_edges = np.asarray(block_edges, dtype=float)
        if discrete_x:
            values = 0.5 * (block_edges[:-1] + block_edges[1:])
            offset = 0.5
        else:
            values = block_edges if i == len(edges) - 1 else block_edges[:-1]
            offset = 0.0
        if discrete_x and block_x_labels is not None and len(block_x_labels) == len(values):
            block_labels = list(block_x_labels)
        else:
            block_labels = [f"{value:g}" for value in values]
        positions.extend(start + offset + j for j in range(len(values)))
        # skip labels closer than *step* ticks to the first (labeled) tick of the next category
        is_last = i == len(edges) - 1
        labels.extend(
            label if j % step == 0 and (is_last or len(values) - j >= step) else ""
            for j, label in enumerate(block_labels)
        )
    return positions, labels


def set_unrolled_xticks(ax: plt.Axes, block_starts: Sequence[int], edges: Sequence[Sequence[float]], **kwargs) -> None:
    """
    Set the x ticks of *ax* to the original binning of the unrolled blocks (see get_unrolled_xticks, which
    receives all *kwargs*). Only every n-th tick is labeled, with n chosen such that the labels do not overlap.
    """
    positions, labels = get_unrolled_xticks(block_starts, edges, **kwargs)
    ax.set_xticks(positions, labels)

    # widest label compared to the distance between two ticks (one bin)
    renderer = ax.figure.canvas.get_renderer()
    max_width = max((label.get_window_extent(renderer).width for label in ax.get_xticklabels()), default=0)
    bin_width = np.diff(ax.transData.transform([(0, 0), (1, 0)])[:, 0])[0]
    step = max(math.ceil(1.2 * max_width / bin_width), 1)
    if step > 1:
        _, labels = get_unrolled_xticks(block_starts, edges, step=step, **kwargs)
        ax.set_xticklabels(labels)


def get_header_bottom(ax: plt.Axes) -> float:
    """
    Return the lower edge, as a fraction of the height of *ax*, of the legend and the texts (e.g. the category
    annotation) in the upper half of *ax*, or 1 if there are none.
    """
    renderer = ax.figure.canvas.get_renderer()
    artists = [text for text in ax.texts if text.get_visible() and text.get_text()]
    if ax.get_legend():
        artists.append(ax.get_legend())
    bottoms = [
        ax.transAxes.inverted().transform((0, artist.get_window_extent(renderer).y0))[1]
        for artist in artists
    ]
    return min([bottom for bottom in bottoms if 0.5 < bottom < 1], default=1.0)


def set_whitespace_fraction(ax: plt.Axes, old_fraction: float, new_fraction: float) -> None:
    """
    Change the upper y-limit of *ax*, set by plot_all for a whitespace fraction of *old_fraction*, to leave a
    fraction of *new_fraction* above the histograms instead. Annotations positioned in data coordinates keep their
    position relative to the axis.
    """
    log_y = ax.get_yscale() == "log"
    y_min, y_max = ax.get_ylim()
    content_max = get_position(y_min, y_max, factor=1 - old_fraction, logscale=log_y)

    # remember the relative positions of annotations given in data coordinates (both the annotated point and the
    # text, which plot_all places at the same position)
    to_axes = lambda y: ax.transAxes.inverted().transform(ax.transData.transform((0, y)))[1]
    from_axes = lambda y: ax.transData.inverted().transform(ax.transAxes.transform((0, y)))[1]
    annotations = [
        (
            text,
            to_axes(text.xy[1]) if text.xycoords == "data" else None,
            to_axes(text.get_position()[1]) if text.anncoords == "data" else None,
        )
        for text in ax.texts
        if hasattr(text, "xycoords")
    ]

    ax.set_ylim(y_min, get_position(y_min, content_max, factor=1 / (1 - new_fraction), logscale=log_y))

    for text, y_xy, y_text in annotations:
        if y_xy is not None:
            text.xy = (text.xy[0], from_axes(y_xy))
        if y_text is not None:
            text.set_y(from_axes(y_text))


def set_legend_ncols(ax: plt.Axes, legend_cfg: dict, max_entries_per_col: int = 3) -> None:
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
    var_name: str | Sequence[str],
    axis_name: str | None = None,
) -> tuple[dict[od.Process, hist.Hist], list[int]]:
    """
    Concatenate the variable axis of the histograms in *hists* (``{block_name: {proc_inst: h}}``, e.g. one block
    per category or per variable) per process, in the order of the blocks. *var_name* is the name of the variable
    axis, either shared by all blocks or given per block. All other axes (e.g. "shift") are kept and must be
    identical between blocks, with the variable axis at the same position. Processes missing in a block are filled
    with zeros there. The unrolled axis has integer edges and no flow bins, and is named *axis_name* (default: the
    first variable name). Returns the unrolled histograms and the number of bins per block.
    """
    import hist

    var_names = [var_name] * len(hists) if isinstance(var_name, str) else list(var_name)
    if len(var_names) != len(hists):
        raise ValueError(f"got {len(var_names)} variable names for {len(hists)} blocks to unroll")

    # number of bins per block
    n_bins = []
    for (block_name, proc_hists), _var_name in zip(hists.items(), var_names):
        if not proc_hists:
            raise ValueError(f"no histograms to unroll in block '{block_name}'")
        n_bins.append(len(list(proc_hists.values())[0].axes[_var_name]))

    unrolled_axis = hist.axis.Variable(
        list(range(sum(n_bins) + 1)),
        name=axis_name or var_names[0],
        underflow=False,
        overflow=False,
    )

    # helper to get the axis names with the variable axis replaced by a placeholder
    get_axis_names = lambda h, _var_name: tuple(None if name == _var_name else name for name in h.axes.name)

    # unique processes in order of first appearance
    proc_insts = list(dict.fromkeys(proc_inst for proc_hists in hists.values() for proc_inst in proc_hists))

    unrolled_hists = {}
    for proc_inst in proc_insts:
        h_ref, var_name_ref = next(
            (proc_hists[proc_inst], _var_name)
            for proc_hists, _var_name in zip(hists.values(), var_names)
            if proc_inst in proc_hists
        )
        var_index = h_ref.axes.name.index(var_name_ref)

        views = []
        for (block_name, proc_hists), _var_name, n in zip(hists.items(), var_names, n_bins):
            if proc_inst not in proc_hists:
                shape = list(h_ref.view().shape)
                shape[var_index] = n
                views.append(np.zeros(shape, dtype=h_ref.view().dtype))
                continue

            h = proc_hists[proc_inst]
            if get_axis_names(h, _var_name) != get_axis_names(h_ref, var_name_ref) or any(
                ax != ax_ref
                for ax, ax_ref in zip(h.axes, h_ref.axes)
                if ax.name != _var_name
            ):
                raise ValueError(
                    f"cannot unroll histograms of process '{proc_inst.name}': axes in block '{block_name}' "
                    f"differ from the ones in other blocks",
                )
            views.append(h.view())

        h_out = hist.Hist(
            *[unrolled_axis if ax.name == var_name_ref else ax for ax in h_ref.axes],
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


def get_varying_keys(tokens: list[dict[Hashable, str | None]]) -> list[Hashable]:
    """
    Return the keys of the root categories in *tokens* (see get_category_tokens) that differ between categories.
    """
    return [key for key in tokens[0] if len({_tokens[key] for _tokens in tokens}) > 1]


def order_categories(
    category_insts: list[od.Category],
    category_order: Sequence[str] | str = (),
    level_order: Sequence[Hashable] | Hashable = (),
) -> tuple[list[od.Category], list[Hashable]]:
    """
    Sort *category_insts* level by level, with the levels being the root category keys that differ between
    categories (see get_category_tokens). *level_order* lists keys from the outermost to the innermost level;
    keys not listed follow in their order in the category names. On each level, root category names are ordered
    as in *category_order*, followed by names not listed in their order of appearance, and categories missing the
    root category last. Returns the sorted categories and the level keys.
    """
    tokens = get_category_tokens(category_insts)
    varying_keys = get_varying_keys(tokens)
    level_order = [key for key in law.util.make_list(level_order) if key in varying_keys]
    level_keys = level_order + [key for key in varying_keys if key not in level_order]

    category_order = list(law.util.make_list(category_order))
    appearance = {
        key: list(dict.fromkeys(_tokens[key] for _tokens in tokens if _tokens[key] is not None))
        for key in level_keys
    }

    def rank(key: Hashable, token: str | None) -> tuple[int, int]:
        if token is None:
            return (2, 0)
        if token in category_order:
            return (0, category_order.index(token))
        return (1, appearance[key].index(token))

    indices = sorted(
        range(len(category_insts)),
        key=lambda i: tuple(rank(key, tokens[i][key]) for key in level_keys),
    )
    return [category_insts[i] for i in indices], level_keys


def build_unroll_labels(
    config_inst: od.Config,
    category_insts: list[od.Category],
    n_bins: list[int],
    level_keys: list[Hashable] | None = None,
) -> tuple[str, list[DotDict]]:
    """
    Build the labels of the unrolled plot from the root categories of *category_insts* (see
    get_category_tokens). Root categories that differ between categories define the levels, ordered by
    *level_keys* (see order_categories) and otherwise as in the category names. Returns the label common to all
    categories and one block per category with its first bin, number of bins, and per level (outermost first) its
    root category name (*path*) and label (*labels*).
    """
    tokens = get_category_tokens(category_insts)
    if level_keys is None:
        level_keys = get_varying_keys(tokens)
    common_label, labels = get_category_labels(config_inst, category_insts, level_keys=level_keys)

    blocks = make_unroll_blocks(
        paths=[tuple(_tokens[key] for key in level_keys) for _tokens in tokens],
        labels=labels,
        n_bins=n_bins,
    )

    return common_label, blocks


def get_category_labels(
    config_inst: od.Config,
    category_insts: list[od.Category],
    level_keys: list[Hashable] | None = None,
) -> tuple[str, list[list[str]]]:
    """
    Split the labels of *category_insts* into their root categories (see get_category_tokens). Returns the label
    of the root categories common to all categories (one line each) and, per category, the labels of the root
    categories given by *level_keys* (defaults to those that differ between categories). Labels are taken from the
    root categories' *label_short* (the name if the root category is unknown), and are empty if a category misses
    the root category.
    """
    def get_label(token: str | None) -> str:
        if token is None:
            return ""
        cat_inst = config_inst.get_category(token, default=None)
        return (cat_inst.label_short if cat_inst else token).replace("\n", " ")

    tokens = get_category_tokens(category_insts)
    varying_keys = get_varying_keys(tokens)
    common_label = "\n".join(
        get_label(tokens[0][key])
        for key in tokens[0]
        if key not in varying_keys and tokens[0][key] is not None
    )
    if level_keys is None:
        level_keys = varying_keys

    labels = [[get_label(_tokens[key]) for key in level_keys] for _tokens in tokens]

    return common_label, labels


def make_unroll_blocks(
    paths: Sequence[tuple[Hashable, ...]],
    labels: Sequence[Sequence[str]],
    n_bins: Sequence[int],
) -> list[DotDict]:
    """
    Create one block per unrolled histogram part (e.g. category or variable) with its first bin (*start*), its
    number of bins (*n_bins*), and per level (outermost first) the name identifying it (*path*) and its label
    (*labels*). Consecutive blocks agreeing in the path up to a level share a label on that level (see
    get_level_spans).
    """
    blocks = []
    start = 0
    for path, _labels, n in zip(paths, labels, n_bins):
        blocks.append(DotDict(start=start, n_bins=n, path=tuple(path), labels=list(_labels)))
        start += n
    return blocks


def get_level_spans(blocks: list[DotDict], level: int) -> list[tuple[int, int, str]]:
    """
    Merge consecutive *blocks* (see build_unroll_labels) that agree in all levels up to and including *level*.
    Returns the first bin, end bin and label of each merged span.
    """
    spans = []
    for block in blocks:
        end = block.start + block.n_bins
        if spans and spans[-1][3] == block.path[:level + 1]:
            spans[-1][1] = end
        else:
            spans.append([block.start, end, block.labels[level], block.path[:level + 1]])
    return [(start, end, label) for start, end, label, _ in spans]


def draw_unroll_labels(
    axs: tuple[plt.Axes, ...],
    blocks: list[DotDict],
    y_line: float,
) -> float:
    """
    Draw separators and labels of the unrolled category *blocks* (see build_unroll_labels). Vertical positions
    are given as fractions of the main axis height, so they are independent of the y-scale: one row of labels per
    level is placed above a horizontal line at *y_line*, with the innermost level at the bottom. Separators are
    solid, dashed and dotted from the outermost to the innermost level and reach up to the row of their level
    (the innermost ones up to *y_line*). Labels are shrunk (and as a last resort rotated) to fit their width.
    Returns the top of the uppermost row of labels as a fraction of the main axis height.
    """
    ax = axs[0]
    trafo = ax.get_xaxis_transform()
    renderer = ax.figure.canvas.get_renderer()
    line_kwargs = {"color": "grey", "linewidth": 2.5}
    n_levels = len(blocks[0].labels)
    linestyles = ["--"] if n_levels == 1 else ["-", "--", ":"]

    # helper to draw a label centered between bins *start* and *end* that fits their width, returns its top
    def draw_label(start: float, end: float, y: float, text: str, fontsize: int) -> float:
        label = ax.text(0.5 * (start + end), y, text, transform=trafo, fontsize=fontsize, ha="center", va="bottom")
        width = np.diff(ax.transData.transform([(start, 0), (end, 0)])[:, 0])[0]
        while label.get_window_extent(renderer).width > 0.95 * width and label.get_fontsize() > 10:
            label.set_fontsize(label.get_fontsize() - 1)
        if label.get_window_extent(renderer).width > 0.95 * width:
            label.set_rotation(90)
        return ax.transAxes.inverted().transform((0, label.get_window_extent(renderer).y1))[1]

    # rows of labels above a horizontal line, from the innermost level up to the outermost one
    n_total = blocks[-1].start + blocks[-1].n_bins
    ax.plot([0, n_total], [y_line, y_line], transform=trafo, **line_kwargs)
    row_tops = [y_line] * n_levels
    y = y_line + 0.01
    for level in reversed(range(n_levels)):
        fontsize = 20 if level == 0 and n_levels > 1 else 16
        row_tops[level] = max(
            [y] + [
                draw_label(start, end, y, label, fontsize)
                for start, end, label in get_level_spans(blocks, level)
                if label
            ],
        )
        y = row_tops[level] + 0.02

    # separators, styled by the outermost level that changes between two blocks
    for prev_block, block in zip(blocks[:-1], blocks[1:]):
        level = next(i for i, (p1, p2) in enumerate(zip(prev_block.path, block.path)) if p1 != p2)
        sep_kwargs = {**line_kwargs, "linestyle": linestyles[min(level, len(linestyles) - 1)]}
        ax.axvline(block.start, ymax=y_line if level == n_levels - 1 else row_tops[level], **sep_kwargs)
        for rax in axs[1:]:
            rax.axvline(block.start, **sep_kwargs)

    return max(row_tops)


def is_discrete_x(variable_inst: od.Variable) -> bool:
    """
    Return whether *variable_inst* has discrete values, i.e. ticks should be placed at the bin centers.
    """
    return bool(variable_inst.discrete_x or "int" in variable_inst.x("axis_type", "variable"))
