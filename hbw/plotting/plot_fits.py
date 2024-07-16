# coding: utf-8

"""
Examples for custom plot functions.
"""

from __future__ import annotations

from collections import OrderedDict

from columnflow.util import maybe_import
from columnflow.plotting.plot_util import (
    remove_residual_axis,
    apply_variable_settings,
    apply_process_settings,
)

hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")

#
# fit functions
#


def gauss(x, A, loc, scale):
    """ Fit function for a Gaussian """
    return A * np.exp(-(x - loc) ** 2 / (2 * scale ** 2))


def scalable_norm(x, A, loc, scale):
    """ should be the same as 'gauss' but using scipy.stats"""
    from scipy.stats import norm
    return A * norm.pdf(x, loc=loc, scale=scale)


def scalable_exponnorm(x, A, loc, scale, K=1):
    """ Fit function for a Gaussian with exponential tail """
    from scipy.stats import exponnorm
    return A * exponnorm.pdf(x, K=K, loc=loc, scale=scale)


# available fit methods mapped to their names
fit_methods = {
    func.__name__: func
    for func in [gauss, scalable_norm, scalable_exponnorm]
}


def plot_fit(
    hists: OrderedDict[od.Process, hist.Hist],
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    yscale: str | None = "",
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    fit_func: str = "gauss",
    **kwargs,
) -> plt.Figure:
    """
    This is a custom plotting function to perform simple fits.

    Exemplary task call:

    .. code-block:: bash
        law run cf.PlotVariables1D --version v1 \
            --processes hh_ggf_kl1_kt1_hbb_hvvqqlnu --variables mli_mbb \
            --plot-function hbw.plotting.plot_fits.plot_fit \
            --general-settings fit_func=scalable_norm
    """
    # imports
    from scipy.optimize import curve_fit
    import scinum
    import inspect

    # we can add arbitrary parameters via the `general_settings` parameter to access them in the
    # plotting function. They are automatically parsed either to a bool, float, or string
    print(f"The fit_func has been set to '{fit_func}'")

    # get the correct fit function
    fit_func = fit_methods[fit_func]
    fit_func_signature = list(inspect.signature(fit_func).parameters.values())

    # call helper function to remove shift axis from histogram
    remove_residual_axis(hists, "shift")

    # call helper functions to apply the variable_settings and process_settings
    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_process_settings(hists, process_settings)

    # use the mplhep CMS stype
    plt.style.use(mplhep.style.CMS)

    # create a figure and fill it with content
    fig, ax = plt.subplots()
    for proc_inst, h in hists.items():
        # plot default histogram
        h.plot1d(
            ax=ax,
            label=proc_inst.label,
            color=proc_inst.color1,
        )

        # read out histogram
        values = h.values()
        bin_edges = h.axes[variable_inst.name].edges
        bin_centers = h.axes[variable_inst.name].centers
        xmin, xmax = bin_edges[0], bin_edges[-1]

        amplitude = np.max(h.values())
        mean = sum(bin_centers * values) / sum(values)
        std = np.sqrt(sum(values * (bin_centers - mean)**2) / sum(values))

        # set some reasonable defaults
        p0 = []
        for fit_param in fit_func_signature[1:]:
            if fit_param.name == "A":
                p0.append(amplitude)
            elif fit_param.name == "loc":
                p0.append(mean)
            elif fit_param.name == "scale":
                p0.append(std)
            else:
                default = fit_param.default if fit_param.default != inspect._empty else 1
                p0.append(default)

        # perform fit
        fit_kwargs = dict(
            f=fit_func,
            xdata=bin_centers,
            ydata=values,
            p0=p0,
            bounds=(xmin, xmax),
            check_finite=True,
            # NOTE: including uncertainties does not work for some reason
            # sigma=np.sqrt(h.variances()),
        )
        popt, pcov = curve_fit(**fit_kwargs)

        fit_label = f"{proc_inst.label} fit \n"
        for i, fit_value in enumerate(popt):
            unc = np.sqrt(pcov[i][i])
            fit_param = fit_func_signature[i + 1]
            number_repr = scinum.Number(fit_value, unc).str(2)
            fit_label += f"    {fit_param.name} = {number_repr} \n"

        # plot fit result
        xr = np.linspace(xmin, xmax, 1000)
        fit_result = fit_func(xr, *popt)
        ax.plot(
            xr,
            fit_result,
            label=fit_label,
            color=proc_inst.color1,
            linestyle="dashed",
        )

    # styling and parameter implementation (e.g. `yscale`)
    ax.set(
        yscale=yscale if yscale else "linear",
        ylabel=variable_inst.get_full_y_title(),
        xlabel=variable_inst.get_full_x_title(),
        xscale="log" if variable_inst.log_x else "linear",
    )
    ax.legend(title=category_inst.label)
    mplhep.cms.label(ax=ax, fontsize=22, llabel="Simulation private work")

    # task expects a figure and a tuple of axes as output
    return fig, (ax,)
