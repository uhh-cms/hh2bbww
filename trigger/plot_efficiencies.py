# coding: utf-8

"""
Plotting function for trigger efficiencies.
Example call:

law run cf.PlotVariables1D --config c22post \
--selector dl1_no_trigger --selector-steps no_trigger \
--producers event_weights,trigger_prod,pre_ml_cats,dl_ml_inputs \
--categories sr__2e --variables mli_lep_pt-trig_ids \
--processes hh_ggf_hbb_hww2l2nu_kl1_kt1,tt_dl \
--plot-function trigger.plot_efficiencies.plot_efficiencies \
--general-settings "bin_sel=Ele30_WPTight_Gsf;" \

optional:
--skip-ratio False --cms-label simpw \
for unrolling, plots are unrolled in the second variable bin:
--variables = mli_lep_pt-mli_lep2_pt-trig_ids \
--general-settings "bin_sel=Ele30_WPTight_Gsf;,unrolling=1;2;3" \
when used as plotting function for multiple weight producers (only one process possible):
law run hbw.PlotVariablesMultiWeightProducer
--general-settings "bin_sel=Ele30_WPTight_Gsf;,multi_weight=True" \
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


def safe_div(num, den, default=0):
    """
    Safely divide two arrays, setting the result to a default value where the denominator is zero.
    """
    return np.where(
        (num > 0) & (den > 0),
        num / den,
        default,
    )


def binom_int(num, den, confint=0.68):
    """
    calculates clopper-pearson error
    """
    from scipy.stats import beta
    quant = (1 - confint) / 2.
    low = beta.ppf(quant, num, den - num + 1)
    high = beta.ppf(1 - quant, num + 1, den - num)

    return (np.nan_to_num(low), np.where(np.isnan(high), 1, high))
    # from hist.intervals import clopper_pearson_interval
    # return clopper_pearson_interval(num=num, denom=den, coverage=confint)


def calc_efficiency_errors(num, den, c):
    """
    Calculate the error on an efficiency given the numerator and denominator histograms.
    """
    if c > -10:
        # Use the variance to scale the numerator and denominator to remove an average weight,
        # this reduces the errors for rare processes to more realistic values
        num_scale = np.nan_to_num(num.values() / num.variances(), nan=1)
        den_scale = num_scale  # np.nan_to_num(den.values() / den.variances(), nan=1)
    else:
        num_scale = 1
        den_scale = 1

    efficiency = np.nan_to_num((num.values()*num_scale) / (den.values()*den_scale), nan=0, posinf=1, neginf=0)

    if np.any(efficiency > 1):
        logger.warning(
            "Some efficiencies are greater than 1",
        )
    elif np.any(efficiency < 0):
        logger.warning(
            "Some efficiencies are less than 0",
        )

    band_low, band_high = binom_int(num.values() * num_scale, den.values() * den_scale)

    error_low = np.asarray(efficiency - band_low)
    error_high = np.asarray(band_high - efficiency)

    # remove negative errors
    if np.any(error_low < 0):
        logger.warning("Some lower uncertainties are negative, setting them to zero")
        error_low[error_low < 0] = 0
    if np.any(error_high < 0):
        logger.warning("Some upper uncertainties are negative, setting them to zero")
        error_high[error_high < 0] = 0

    # remove large errors in empty bins
    error_low[efficiency == 0] = 0
    error_high[efficiency == 0] = 0

    # stacking errors
    errors = np.concatenate(
        (error_low.reshape(error_low.shape[0], 1), error_high.reshape(error_high.shape[0], 1)),
        axis=1,
    )
    errors = errors.T

    return errors


def calc_ratio_uncertainty(efficiencies: dict, errors: dict):
    """
    Calculate the error on the scale factors using a gaussian error propagation.
    """
    # symmetrize errors
    sym_errors = {}
    for key, value in errors.items():
        if value.ndim == 2 and value.shape[0] == 2:
            sym_errors[key] = np.maximum(value[0], value[1])
        else:
            sym_errors[key] = np.maximum(value[..., 0, :], value[..., 1, :])

    # combine errors
    uncertainty = np.sqrt(
        (sym_errors[0] / efficiencies[1]) ** 2 + (efficiencies[0] * sym_errors[1] / efficiencies[1] ** 2) ** 2
    )

    return np.nan_to_num(uncertainty, nan=0, posinf=1, neginf=0)


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
    Plotting function for trigger efficiencies.
    """

    # multi_weight allows using the task hbw.PlotVariablesMultiWeightproducer
    # to plot the efficiencies of multiple weight producers,
    # This only works for single processes!
    if kwargs.get("multi_weight", False):
        # take processes from the first weight producer (they should always be the same)
        processes = list(list(hists.values())[0].keys())
        if len(processes) > 1:
            raise ValueError("multi_weight=True only works for single processes")

        # merge over processes
        for weight_producer, w_hists in hists.items():
            hists[weight_producer] = sum(w_hists.values())

    remove_residual_axis(hists, "shift")

    if not kwargs.get("multi_weight", False):
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

    # save efficiencies for ratio calculation
    if not kwargs.get("skip_ratio", True):
        efficiencies = {}
        efficiency_sums = {}
        errors = {}

    count_key = 0

    # for unrolling efficiencies
    subslices = kwargs.get("unroll", [None])
    for subslice in subslices:
        # loop over processeslabel
        for proc_inst, myhist in hists.items():
            # for unrolling efficiencies, unroll takes the number of the bin to plot
            if "unroll" in kwargs:
                myhist = myhist[:, int(subslice), :]

            if not hasattr(proc_inst, 'label'):
                proc_label = proc_inst
            else:
                proc_label = proc_inst.label

            # get normalisation from first histogram (all events)
            # TODO: this could be possibly be a CLI option
            norm_hist = myhist[:, 0]

            # plot config for the background distribution
            if not kwargs.get("skip_background", False):

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
                    label = f"{proc_label}"
                else:
                    label_dict = {
                        "mixed": r"$e\mu$ trigger",  # "mixed + ele&jet",
                        "emu_dilep": "OR dilepton triggers",
                        "ee_dilep": "OR dilepton triggers",
                        "mm_dilep": "OR dilepton triggers",
                        "emu_single": "Single lepton triggers",
                        "ee_single": "Single lepton triggers",
                        "mm_single": "Single lepton triggers",
                        "single": "Single lepton triggers",
                        "electron+jet": "Electron + Jet trigger",
                        "emu_dilep+emu_single": "OR dilepton triggers",
                        "ee_dilep+ee_single": "OR dilepton triggers",
                        "mm_dilep+mm_single": "OR dilepton triggers",
                        "emu_dilep+emu_single+emu_electronjet": "OR Electron + Jet trigger",
                        "ee_dilep+ee_single+ee_electronjet": "OR Electron + Jet trigger",
                        "mm_dilep+mm_single+mm_electronjet": "OR Electron + Jet trigger",
                        "alt_mix": "mixed + ele115",
                        "dilep+single+electronjet+alt_mix": "mixed + ele115 + ele&jet",
                    }
                    if i in label_dict.keys():
                        label = label_dict[i]
                    else:
                        label = i

                if "unroll" in kwargs:
                    label += f" (bin {subslice})"

                # calculate efficiency
                # this scaling here is not really necessary as it cancels out
                if count_key >= -10:
                    # label += ", scaled"
                    num_scale = np.nan_to_num(myhist[:, hist.loc(i)].values() / myhist[:, hist.loc(i)].variances(), nan=1)  # noqa
                    den_scale = num_scale  # np.nan_to_num(norm_hist.values() / norm_hist.variances(), nan=1)
                else:
                    num_scale = 1
                    den_scale = 1
                efficiency = np.nan_to_num((myhist[:, hist.loc(i)].values()*num_scale) / (norm_hist.values()*den_scale),  # noqa
                                           nan=0, posinf=1, neginf=0
                                           )
                efficiency_sum = np.sum(myhist[:, hist.loc(i)].values()) / np.sum(norm_hist.values())
                # calculate uncertainties
                if kwargs.get("skip_errorbars", False):
                    eff_err = None
                else:
                    eff_err = calc_efficiency_errors(myhist[:, hist.loc(i)], norm_hist, count_key)

                plot_config[f"hist_{proc_label}_{label}"] = {
                    "method": "draw_custom_errorbars",
                    "hist": myhist[:, hist.loc(i)],
                    "kwargs": {
                        "y": efficiency,
                        "yerr": eff_err,
                        "label": r"$e\mu$ trigger" if label == "mixed" else f"{label}",
                    },
                }

                # calculate ratio of efficiencies
                if not kwargs.get("skip_ratio", True):
                    efficiencies[count_key] = efficiency
                    efficiency_sums[count_key] = efficiency_sum
                    errors[count_key] = eff_err
                    count_key += 1
                    if len(efficiencies) > 1:
                        plot_config[f"hist_{proc_label}_{label}"]["ratio_kwargs"] = {
                            "y": np.nan_to_num(efficiencies[0] / efficiencies[1], nan=1, posinf=1, neginf=1),
                            "yerr": calc_ratio_uncertainty(efficiencies, errors),
                            "label": f"{label}",
                            # "annotate": f"alpha={efficiency_sums[0]/efficiency_sums[1]:.2f}",
                        }
                    # else:
                    #    plot_config[f"hist_{proc_label}_{label}"]["ratio_kwargs"] = {
                    #        "y": np.ones_like(efficiency),
                    #        "yerr": None,
                    #        "label": f"{label}",
                    #    }

            # set legend title to process name
            if proc_as_label:
                default_style_config["legend_cfg"]["title"] = legend_title
            else:
                if "title" in default_style_config["legend_cfg"]:
                    if proc_label not in default_style_config["legend_cfg"]["title"]:
                        default_style_config["legend_cfg"]["title"] += " & " + proc_label
                else:
                    default_style_config["legend_cfg"]["title"] = proc_label

    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = "Efficiency"

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    if "xlim" in kwargs:
        style_config["ax_cfg"]["xlim"] = kwargs["xlim"]

    style_config["cms_label_cfg"]["fontsize"] = 21
    style_config["ax_cfg"]["ylim"] = kwargs.get("ylim", (0, 1.5))
    style_config["rax_cfg"]["ylabel"] = "Ratio"
    style_config["rax_cfg"]["ylim"] = kwargs.get("rax_ylim", (0.92, 1.08))

    style_config["legend_cfg"]["title_fontsize"] = 23
    style_config["legend_cfg"]["fontsize"] = 23
    style_config["legend_cfg"]["ncols"] = 1
    style_config["legend_cfg"]["reverse"] = False

    # fig, axs = plot_all(plot_config, style_config, **kwargs)

    # axs[1].annotate(
    #     fr"$\mathit{{\alpha}}$={efficiency_sums[0]/efficiency_sums[1]:.2f}",
    #     xy=(0.9, 0.2),
    #     xycoords="axes fraction",
    #     ha="center",
    #     va="center",
    #     fontsize=20,
    # )
    return plot_all(plot_config, style_config, **kwargs)
