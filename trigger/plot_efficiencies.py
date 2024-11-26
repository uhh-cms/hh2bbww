# coding: utf-8

"""
Examples for custom plot functions.
"""

from __future__ import annotations

from collections import OrderedDict

import law

from columnflow.util import maybe_import
from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import (
    prepare_style_config,
    remove_residual_axis,
    apply_variable_settings,
    apply_process_settings,
    apply_density_to_hists,
)

hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")

logger = law.logger.get_logger(__name__)

"""
law run cf.PlotVariables1D --version v1 --config l22post \
--processes tt_dl --variables electron_pt-trig_bits --categories trig_mu \
--selector trigger_sel --producers event_weights,trigger_prod \
--plot-function trigger.plot_efficiencies.plot_efficiencies \
--variable-settings "electron_pt,rebin=5,x_min=0,x_max=100" \
--general-settings "bin_sel=Ele30_WPTight_Gsf;"
"""


def plot_efficiencies(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool = False,
    yscale: str | None = None,
    variable_settings: dict | None = None,
    process_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    TODO.
    """
    remove_residual_axis(hists, "shift")

    hists = apply_process_settings(hists, process_settings)
    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )

    # switch trigger and processes when plotting efficiency of one trigger for multiple processes
    proc_as_label = False
    if len(hists.keys()) > 1:
        if "bin_sel" in kwargs:
            mask_bins = tuple(bin for bin in kwargs["bin_sel"] if bin)
            if len(mask_bins) == 1:
                legend_title = f"{mask_bins[0]}"
                proc_as_label = True

    # loop over processeslabel
    for proc_inst, myhist in hists.items():

        # get normalisation from first histogram (all events)
        # TODO: this could be possibly be a CLI option
        norm_hist = myhist[:, 0]

        # plot config for the background distribution
        plot_config["hist_0"] = {
            "method": "draw_hist_twin",
            "hist": myhist[:, 0],
            "kwargs": {
                "norm": 1,
                "label": None,
                "color": "grey",
                "histtype": "fill",
                "alpha": 0.3,
            },
        }

        # plot config for the individual triggers
        if "bin_sel" in kwargs:
            mask_bins = tuple(bin for bin in kwargs["bin_sel"] if bin)
        else:
            mask_bins = myhist.axes[1]
        for i in mask_bins:
            if i == 0:
                continue

            if proc_as_label:
                label = f"{proc_inst.label}"
            else:
                label = i

            plot_config[f"hist_{proc_inst.label}_{i}"] = {
                "method": "draw_efficiency",
                "hist": myhist[:, hist.loc(i)],
                "kwargs": {
                    "h_norm": norm_hist,
                    "label": f"{label}",
                },
            }

        # set legend title to process name
        if proc_as_label:
            default_style_config["legend_cfg"]["title"] = legend_title
        else:
            if "title" in default_style_config["legend_cfg"]:
                default_style_config["legend_cfg"]["title"] += " & " + proc_inst.label
            else:
                default_style_config["legend_cfg"]["title"] = proc_inst.label

    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = "Efficiency"

    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    # set correct CMS label TODO: this should be implemented correctly in columnflow by default at one point
    # style_config["cms_label_cfg"]["exp"] = ""
    # if "data" in proc_inst.name:
    #     style_config["cms_label_cfg"]["llabel"] = "Private Work (CMS Data)"
    # else:
    #     style_config["cms_label_cfg"]["llabel"] = "Private Work (CMS Simulation)"
    if "xlim" in kwargs:
        style_config["ax_cfg"]["xlim"] = kwargs["xlim"]

    return plot_all(plot_config, style_config, **kwargs)
