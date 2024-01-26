# coding: utf-8

"""
Examples for custom plot functions.
"""

from __future__ import annotations

from collections import defaultdict, OrderedDict

import law

from columnflow.util import maybe_import
from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import (
    # prepare_plot_config,
    prepare_style_config,
    remove_residual_axis,
    apply_variable_settings,
    apply_process_settings,
    apply_density_to_hists,
)

from hbw.util import round_sig

hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")

logger = law.logger.get_logger(__name__)


def separate_sig_bkg_hists(hists: OrderedDict):
    # separate histograms into signal and background based on process tag 'is_signal'
    h_sig, h_bkg = None, None
    for proc_inst, h in hists.items():
        if proc_inst.has_tag("is_signal"):
            h_sig = h + h_sig if h_sig else h
        else:
            h_bkg = h + h_bkg if h_bkg else h

    if not h_sig:
        raise Exception(
            "No signal processes given. Remember to add the 'is_signal' auxiliary to your "
            "signal processes",
        )
    if not h_bkg:
        raise Exception("No background processes given")

    return h_sig, h_bkg


def plot_s_over_b(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool | None = False,
    yscale: str | None = "",
    hide_errors: bool | None = None,
    sqrt_b: bool | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plotting function to create a single line presenting signal vs background ratio.
    Some plot parameters might be supported but have not been tested.
    Exemplary task call:

    .. code-block:: bash
        law run cf.PlotVariables1D --version prod1 \
            --processes ggHH_kl_1_kt_1_sl_hbbhww,tt_sl --variables jet1_pt \
            --plot-function hbw.plotting.s_over_b.plot_s_over_b \
            --general-settings sqrt_b
    """
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_process_settings(hists, process_settings)
    hists = apply_density_to_hists(hists, density)

    # separate histograms into signal and background based on process tag 'is_signal'
    h_sig, h_bkg = separate_sig_bkg_hists(hists)

    # TODO: include uncertainties
    S_over_B = h_sig[::sum].value / h_bkg[::sum].value
    S_over_sqrtB = h_sig[::sum].value / np.sqrt(h_bkg[::sum].value)
    logger.info(
        f"\n    Integrated S over B:     {round_sig(S_over_B)}" +
        f"\n    Integrated S over sqrtB: {round_sig(S_over_sqrtB)}",
    )

    # NOTE: this does not take into account the variances on the background stack
    if sqrt_b:
        h_out = h_sig / np.sqrt(h_bkg.values())
        ylabel = r"$Signal / \sqrt{Background}$"
    else:
        h_out = h_sig / h_bkg.values()
        ylabel = "Signal / Background"

    # draw lines
    plot_config = {}
    line_norm = sum(h_out.values()) if shape_norm else 1
    plot_config["s_over_b"] = {
        "method": "draw_hist",
        "hist": h_out,
        "kwargs": {
            "norm": line_norm,
        },
    }

    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # disable autmatic setting of ylim
    default_style_config["ax_cfg"]["ylim"] = None
    default_style_config["ax_cfg"]["ylabel"] = ylabel

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)
    if shape_norm:
        style_config["ax_cfg"]["ylabel"] = r"$\Delta N/N$"

    # ratio plot not used here; set `skip_ratio` to True
    kwargs["skip_ratio"] = True

    return plot_all(plot_config, style_config, **kwargs)


def cutflow_s_over_b(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool = False,
    yscale: str | None = None,
    hide_errors: bool | None = None,
    sqrt_b: bool | None = None,
    process_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plotting function to create a single line presenting signal vs background ratio
    after each selection step.
    Also prints a table containing all process yields after each step + S over B ratios
    Some plot parameters might be supported but have not been tested.
    Exemplary task call:

    .. code-block:: bash
        law run cf.PlotCutflow --version prod1 \
            --processes ggHH_kl_1_kt_1_sl_hbbhww,tt_sl \
            --plot-function hbw.plotting.s_over_b.cutflow_s_over_b \
            --general-settings sqrt_b
    """
    from tabulate import tabulate

    remove_residual_axis(hists, "shift")

    hists = apply_process_settings(hists, process_settings)
    hists = apply_density_to_hists(hists, density)

    selector_steps = list(hists[list(hists.keys())[0]].axes["step"])
    selector_step_labels = config_inst.x("selector_step_labels", {})

    #
    # gather yields per process and step
    #
    yields = defaultdict(list)
    for process_inst, h in hists.items():
        yields["Label"].append(process_inst.name)
        for step in selector_steps:
            step_label = step
            # step_label = selector_step_labels.get(step, step)
            yields[step_label].append(round_sig(h[{"step": step}].value, 4))

    # separate histograms into signal and background based on process tag 'is_signal'
    h_sig, h_bkg = separate_sig_bkg_hists(hists)

    # NOTE: this does not take into account the variances on the background stack
    h_s_over_sqrt_b = h_sig / np.sqrt(h_bkg.values())
    h_s_over_b = h_sig / h_bkg.values()
    if sqrt_b:
        h_out = h_s_over_sqrt_b
        ylabel = r"$Signal / \sqrt{Background}$"
    else:
        h_out = h_s_over_b
        ylabel = "Signal / Background"

    yields["Label"].append("S/sqrt(B)")
    yields["Label"].append("S/B")
    for step in selector_steps:
        step_label = step
        # step_label = selector_step_labels.get(step, step)
        yields[step_label].append(round_sig(h_s_over_sqrt_b[{"step": step}].value, 4))
        yields[step_label].append(round_sig(h_s_over_b[{"step": step}].value, 4))

    # create, print and save the yield table
    yield_table = tabulate(yields, headers="keys", tablefmt="fancy_grid")
    print(yield_table)

    #
    # plotting
    #
    plot_config = {}
    plot_config["s_over_b"] = {
        "method": "draw_hist",
        "hist": h_out,
    }

    # update xticklabels based on config
    xticklabels = []

    for step in selector_steps:
        xticklabels.append(selector_step_labels.get(step, step))

    # setup style config
    if not yscale:
        yscale = "linear"

    default_style_config = {
        "ax_cfg": {
            "ylim": None,
            "ylabel": ylabel,
            "xlabel": "Selection step",
            "xticklabels": xticklabels,
            "yscale": yscale,
        },
        "legend_cfg": {
            "loc": "upper right",
        },
        "annotate_cfg": {"text": category_inst.label},
        "cms_label_cfg": {
            "lumi": config_inst.x.luminosity.get("nominal") / 1000,  # pb -> fb
        },
    }
    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    # ratio plot not used here; set `skip_ratio` to True
    kwargs["skip_ratio"] = True

    fig, (ax,) = plot_all(plot_config, style_config, **kwargs)

    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    return fig, (ax,)
