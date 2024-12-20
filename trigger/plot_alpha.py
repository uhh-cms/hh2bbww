# coding: utf-8

"""
Custom plotting tasks
"""
from __future__ import annotations

from collections import OrderedDict

import law
import order as od


from columnflow.util import maybe_import

from columnflow.plotting.plot_util import (
    remove_residual_axis,
    apply_variable_settings,
    get_position,
)

logger = law.logger.get_logger(__name__)


# imports for plot function

hist = maybe_import("hist")
plt = maybe_import("matplotlib.pyplot")
np = maybe_import("numpy")
mplhep = maybe_import("mplhep")


def safe_div(num, den, default=1.0):
    """
    Safely divide two arrays, setting the result to 1 where the denominator is zero.
    """
    return np.where(
        (num > 0) & (den > 0),
        num / den,
        default,
    )


def plot_alpha(
    hists: dict[str, OrderedDict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool | None = False,
    yscale: str | None = None,
    hide_errors: bool | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:

    variable_inst = variable_insts[0]

    # take processes from the first weight producer (they should always be the same)
    processes = list(list(hists.values())[0].keys())

    # merge over processes
    for weight_producer, w_hists in hists.items():
        hists[weight_producer] = sum(w_hists.values())

    remove_residual_axis(hists, "shift")
    hists = apply_variable_settings(hists, variable_insts, variable_settings)

    efficiencies = []
    for weight_producer, h in hists.items():
        efficiencies.append(safe_div(h[:, hist.loc(kwargs["trigger"])].values(), h[:, 0].values(), default=0.))

    # calculate scale factors and store them as a correctionlib evaluator
    scale_factors = safe_div(efficiencies[0], efficiencies[1])

    sfhist = hist.Hist(*hists[list(hists.keys())[0]][:, 0].axes, data=scale_factors)
    sfhist.name = f"alpha_{kwargs['trigger']}_{variable_inst.name}"
    sfhist.label = "out"

    # plot 1D or 2D scalefactors
    if len(sfhist.axes.size) <= 2:
        # use CMS plotting style
        plt.style.use(mplhep.style.CMS)

        if len(sfhist.axes.size) == 2:
            fig, ax = plt.subplots()
        else:
            fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0), sharex=True)
            (ax, rax) = axs

        cms_label_kwargs = {
            "ax": ax,
            "llabel": "Private work (CMS simulation)",
            "fontsize": 22,
            "data": False,
            "exp": "",
            "com": config_inst.campaign.ecm,
            "lumi": round(0.001 * config_inst.x.luminosity.get("nominal"), 2)
        }
        mplhep.cms.label(**cms_label_kwargs)

        if len(sfhist.axes.size) == 2:
            sfhist.plot2d(ax=ax)
        else:
            ax.errorbar(x=sfhist.axes[0].centers, y=efficiencies[0], fmt="o",
                        label="Truth")
            ax.errorbar(x=sfhist.axes[0].centers, y=efficiencies[1], fmt="o",
                        label="ortho")
            rax.errorbar(x=sfhist.axes[0].centers, y=sfhist.values(), fmt="o")
            rax.axhline(y=1.0, linestyle="dashed", color="gray")
            rax_kwargs = {
                "ylim": (0.85, 1.15),
                "ylabel": r"Ratio ($\alpha$)",
                "xlabel": f"{sfhist.axes[0].label}",
                "yscale": "linear",
            }
            rax.set(**rax_kwargs)
            ax.set_ylabel("Efficiency")
            ax.set_ylim(0, 1.04)

        annotate_kwargs = {
            "text": category_inst.label,
            "xy": (
                get_position(*ax.get_xlim(), factor=0.05),
                get_position(*ax.get_ylim(), factor=0.95),
            ),
            "xycoords": "data",
            "color": "black",
            "fontsize": 22,
            "horizontalalignment": "left",
            "verticalalignment": "top",
        }
        ax.annotate(**annotate_kwargs)

        ax.legend(title=f"{kwargs['trigger']}, {processes[0].label}")
        fig.tight_layout()

    return fig, ax
