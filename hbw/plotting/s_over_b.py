# coding: utf-8

"""
Examples for custom plot functions.
"""

from __future__ import annotations

from collections import defaultdict, OrderedDict

import law
import order as od

from columnflow.util import maybe_import
from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import (
    # prepare_stack_plot_config,
    prepare_style_config,
    remove_residual_axis,
    apply_variable_settings,
    apply_process_settings,
    apply_density,
)

from hbw.util import round_sig

from columnflow.types import TYPE_CHECKING

np = maybe_import("numpy")
if TYPE_CHECKING:
    plt = maybe_import("matplotlib.pyplot")
    hist = maybe_import("hist")

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
            "No signal processes given. Remember to add the 'is_signal' tag to your "
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
    cumsum: bool = False,
    reversed_cumsum: bool = False,
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
            --processes hh_ggf_hbb_hvvqqlnu_kl1_kt1,tt_sl --variables jet1_pt \
            --plot-function hbw.plotting.s_over_b.plot_s_over_b \
            --general-settings sqrt_b
    """
    if cumsum and reversed_cumsum:
        raise Exception(
            "We can only do the cumulative cumsum in one direction at the time!")

    hists = remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists, variable_style_config = apply_variable_settings(
        hists, variable_insts, variable_settings)
    hists, process_style_config = apply_process_settings(
        hists, process_settings)
    hists = apply_density(hists, density)

    # separate histograms into signal and background based on process tag 'is_signal'
    h_sig, h_bkg = separate_sig_bkg_hists(hists)

    # TODO: include uncertainties
    S_over_B = h_sig[::sum].value / h_bkg[::sum].value
    S_over_sqrtB = h_sig[::sum].value / np.sqrt(h_bkg[::sum].value)
    logger.info(
        f"\n    Integrated S over B:     {round_sig(S_over_B)}" +
        f"\n    Integrated S over sqrtB: {round_sig(S_over_sqrtB)}",
    )

    if cumsum or reversed_cumsum:
        # calculate the cumulative sum of the histograms
        for h in (h_sig, h_bkg):
            h_view = h.view()
            h_view.value = np.cumsum(h_view.value) if cumsum else np.cumsum(
                h_view.value[::-1])[::-1]

            # TODO: implement correct variances; in the meantime, just set them to 0
            h_view.variance = [0] * len(h.view())

            # overwrite view of original histograms
            h[...] = h_view

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

    # default_style_config = prepare_style_config(
    #     config_inst, variable_inst,  # category_inst, density, shape_norm, yscale,
    # )
    default_style_config = prepare_style_config(
        config_inst=config_inst,
        variable_inst=variable_inst,
        category_inst=category_inst,
        density=density,
        shape_norm=shape_norm,
        yscale=yscale,
    )
    # disable autmatic setting of ylim
    default_style_config["ax_cfg"]["ylim"] = None
    default_style_config["ax_cfg"]["ylabel"] = ylabel

    style_config = law.util.merge_dicts(
        default_style_config,
        process_style_config,
        variable_style_config[variable_inst],
        style_config,
        deep=True,
    )
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
            --processes hh_ggf_hbb_hvvqqlnu_kl1_kt1,tt_sl \
            --plot-function hbw.plotting.s_over_b.cutflow_s_over_b \
            --general-settings sqrt_b
    """
    from tabulate import tabulate

    hists = remove_residual_axis(hists, "shift")

    hists = apply_process_settings(hists, process_settings)
    hists = apply_density(hists, density)

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
        yields[step_label].append(
            round_sig(h_s_over_sqrt_b[{"step": step}].value, 4))
        yields[step_label].append(
            round_sig(h_s_over_b[{"step": step}].value, 4))

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
    style_config = law.util.merge_dicts(
        default_style_config, style_config, deep=True)

    # ratio plot not used here; set `skip_ratio` to True
    kwargs["skip_ratio"] = True

    fig, (ax,) = plot_all(plot_config, style_config, **kwargs)

    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    return fig, (ax,)


def plot_data_mc_double_ratio(
    hists: dict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    shape_norm: bool | None = False,
    **kwargs,
):
    """
    Top:    Data/MC for each config
    Bottom: (Data/MC)_config0 / (Data/MC)_config1
    Requires --hist-hooks campaign_split (process names carry a '__<config>' suffix).
    """
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    variable_inst = variable_insts[0]

    # regroup processes by config suffix
    per_cfg = defaultdict(dict)
    for p, h in hists.items():
        if "__" not in p.name:
            raise Exception(
                f"process '{p.name}' has no config suffix; did you run with --hist-hooks campaign_split?",
            )
        per_cfg[p.name.rsplit("__", 1)[1]][p] = h
    cfg_names = list(per_cfg)
    if len(cfg_names) != 2:
        raise Exception(f"need exactly two configs, got {cfg_names}")

    def _sum(hs):
        hs = list(hs)
        if not hs:
            return None
        out = hs[0].copy()
        for h in hs[1:]:
            out = out + h
        return out

    def _prep(h):
        if "shift" in h.axes.name:
            h = h[{"shift": sum}]
        if "category" in h.axes.name:
            h = h[{"category": sum}]
        v = h.values()
        var = h.variances() if h.variances() is not None else v.copy()
        return v, var, h.axes[0].edges

    def _ratio(n, vn, d, vd):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = n / d
            e = np.abs(r) * np.sqrt(vn / n**2 + vd / d**2)
        r[~np.isfinite(r)] = np.nan
        e[~np.isfinite(e)] = np.nan
        return r, e

    ratios, lumis = {}, {}
    for name in cfg_names:
        procs = per_cfg[name]
        h_data = _sum(h for p, h in procs.items() if p.is_data)
        h_mc = _sum(h for p, h in procs.items()
                    if not p.is_data and not p.has_tag("is_signal"))
        if h_data is None or h_mc is None:
            raise Exception(
                f"{name}: need both data and background MC processes")
        d, vd, edges = _prep(h_data)
        m, vm, _ = _prep(h_mc)
        logger.info(f"{name}: integrated data/MC = {d.sum() / m.sum():.4f}")
        if shape_norm:
            sd, sm = d.sum(), m.sum()
            d, vd, m, vm = d / sd, vd / sd**2, m / sm, vm / sm**2
        ratios[name] = _ratio(d, vd, m, vm)
        lumis[name] = next(iter(procs)).x("lumi", None)

    (c1, (r1, e1)), (c2, (r2, e2)) = ratios.items()
    dr, dr_err = _ratio(r1, e1**2, r2, e2**2)

    centers = 0.5 * (edges[1:] + edges[:-1])
    xerr = 0.5 * (edges[1:] - edges[:-1])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 9),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    )
    for (name, (r, e)), color, marker in zip(ratios.items(), ("black", "tab:red"), ("o", "s")):
        ax1.errorbar(centers, r, xerr=xerr, yerr=e,
                     fmt=marker, ms=4, color=color, label=name)
    ax1.axhline(1.0, color="gray", ls="--", lw=1)
    ax1.set_ylabel("Data / MC" + (" (shape)" if shape_norm else ""))
    ax1.set_ylim(0.0, 2.0)
    ax1.legend(loc="upper right", fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.04, 0.94, category_inst.label, transform=ax1.transAxes,
             ha="left", va="top", fontsize=16)

    ax2.errorbar(centers, dr, xerr=xerr, yerr=dr_err,
                 fmt="o", ms=4, color="tab:blue")
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.set_ylabel(f"{c1} / {c2}", fontsize=16)
    ax2.set_ylim(0.7, 2.0)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel(variable_inst.get_full_x_title())
    ax2.set_xlim(edges[0], edges[-1])

    if all(l is not None for l in lumis.values()):  # noqa
        l1, l2 = (lumis[n] / 1000 for n in cfg_names)  # pb -> fb
        rlabel = f"{l1:.1f} + {l2:.1f} fb$^{{-1}}$ (13.6 TeV)"
    else:
        rlabel = "(13.6 TeV)"
    hep.cms.label(ax=ax1, llabel="Private work",
                  data=True, fontsize=20, rlabel=rlabel)

    return fig, (ax1, ax2)
