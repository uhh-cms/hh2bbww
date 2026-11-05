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
    prepare_style_config, remove_residual_axis, use_flow_bins, apply_variable_settings, get_position,
)
from columnflow.util import maybe_import
from columnflow.types import TYPE_CHECKING, Any, Hashable, Sequence

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
    unroll_category_order: Sequence[str] | str | None = None,
    unroll_level_order: Sequence[Hashable] | Hashable | None = None,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """
    Create a stack plot with the bins of all categories unrolled next to each other on a single x-axis. Example:
    claw run cf.PlotVariables1D --multi-category \
        --plot-function hbw.plotting.multi.multi_cat_unrolled_plot \
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
    (at least 0.45 by default, can be raised via --general-settings whitespace_fraction=0.5) is increased
    automatically until the labels fit between the histograms and the legend (and other texts at the top of the
    plot), unless the y-range is fixed via the ax_cfg of the style config.
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
    common_label, blocks = build_unroll_labels(config_inst, category_insts, n_bins, level_keys=level_keys)
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
        set_legend_ncols(axs[0], style_config["legend_cfg"], max_entries_per_col=3)

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

    blocks = []
    start = 0
    for _tokens, n in zip(tokens, n_bins):
        blocks.append(DotDict(
            start=start,
            n_bins=n,
            path=tuple(_tokens[key] for key in level_keys),
            # empty label if the root category is missing
            labels=[get_label(_tokens[key]) for key in level_keys],
        ))
        start += n

    return common_label, blocks


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


# helper to move flow bins into first/last ones
def move_flow(h: hist.Hist, variable_inst: od.Variable) -> hist.Hist:
    overflow = variable_inst.x("overflow", False)
    underflow = variable_inst.x("underflow", False)
    if overflow or underflow:
        h = use_flow_bins(h, variable_inst.name, underflow=underflow, overflow=overflow)
    return h
