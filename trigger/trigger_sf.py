# coding: utf-8

"""
Tasks to calculate and save trigger scale factors.
"""

import luigi
import law

from columnflow.util import maybe_import, DotDict
from columnflow.tasks.framework.histograms import (
    HistogramsUserSingleShiftBase,
)
from columnflow.tasks.framework.parameters import MultiSettingsParameter

logger = law.logger.get_logger(__name__)

hist = maybe_import("hist")
np = maybe_import("numpy")
scipy = maybe_import("scipy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


def safe_div(num, den):
    """
    Safely divide two arrays, setting the result to 1 where the denominator is zero.
    """
    return np.where(
        (num > 0) & (den > 0),
        num / den,
        1.0,
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


def calculate_efficiencies(
        h: hist.Hist,
        trigger: str,
) -> dict:
    """
    Calculates the efficiencies for the different triggers.
    """
    efficiencies = {}
    # loop over processes
    for proc in range(h.axes[0].size):
        # Extract histogram data
        values = h[proc, ..., hist.loc(trigger)].values()
        norm = h[proc, ..., 0].values()

        efficiency = np.nan_to_num(values / norm)

        if np.any(efficiency > 1):
            logger.warning(
                "Some efficiencies for are greater than 1, errorbars are capped at zero",
            )
        elif np.any(efficiency < 0):
            logger.warning(
                "Some efficiencies for are less than 0, errorbars are capped at zero",
            )
        efficiencies[proc] = efficiency

    return efficiencies


class CalculateTriggerScaleFactors(
    HistogramsUserSingleShiftBase,
):
    """
    Loads histograms containing the trigger informations for data and MC and calculates the trigger scale factors.
    """

    trigger = luigi.Parameter(
        default="allEvents",
        description="The trigger to calculate the scale factors for, must be set; default: allEvents",
    )

    variable_settings = MultiSettingsParameter(
        default=DotDict(),
        significant=False,
        description="parameter for changing different variable settings; format: "
        "'var1,option1=value1,option3=value3:var2,option2=value2'; options implemented: "
        "rebin; can also be the key of a mapping defined in 'variable_settings_groups; "
        "default: value of the 'default_variable_settings' if defined, else empty default",
        brace_expand=True,
    )

    def output(self):
        return {
            "trigger_scale_factors": self.target(f"sf_{self.trigger}_{self.variables[0]}.json"),
            "trigger_sf_plot": self.target(f"{self.trigger}_{self.variables[0]}_sf_plot.pdf"),
            }

    def run(self):
        import correctionlib
        import correctionlib.convert

        outputs = self.output()

        hists = {}

        for dataset in self.datasets:
            for variable in self.variables:
                h_in = self.load_histogram(dataset, variable)
                h_in = self.slice_histogram(h_in, self.processes, self.categories, self.shift)

                # apply variable settings
                for var in self.variable_settings:
                    if var in h_in.axes.name:
                        h_in = h_in[{var: hist.rebin(int(self.variable_settings[var]["rebin"]))}]
                    else:
                        logger.warning(f"Variable {var} not found in histogram, skipping rebinning")

                if variable in hists.keys():
                    hists[variable] += h_in
                else:
                    hists[variable] = h_in

        # calculate efficiencies
        efficiencies = calculate_efficiencies(hists[self.variables[0]][0, :, 0, ..., :], self.trigger)  # category, process, shift, variable, bin # noqa
        # calculate scale factors and store them as a correctionlib evaluator
        scale_factors = safe_div(efficiencies[0], efficiencies[1])

        sfhist = hist.Hist(*hists[self.variables[0]][0, 0, 0, ..., 0].axes, data=scale_factors)
        sfhist.name = f"sf_{self.trigger}_{self.variables[0]}"
        sfhist.label = "out"

        trigger_scale_factors = correctionlib.convert.from_histogram(sfhist)
        trigger_scale_factors.description = f"{self.trigger} scale factors, binned in {self.variables[0]}"

        # set overflow bins behavior (default is to raise an error when out of bounds)
        trigger_scale_factors.data.flow = "clamp"

        # create correction set and store it
        cset = correctionlib.schemav2.CorrectionSet(
            schema_version=2,
            description=f"{self.trigger} scale factors",
            corrections=[trigger_scale_factors],
        )
        cset_json = cset.json(exclude_unset=True)

        outputs["trigger_scale_factors"].dump(
            cset_json,
            formatter="json",
        )

        # plot 1D or 2D scalefactors
        if len(sfhist.axes.size) <= 2:
            # use CMS plotting style
            plt.style.use(mplhep.style.CMS)
            fig, ax = plt.subplots()
            cms_label_kwargs = {
                "ax": ax,
                "llabel": "Private work (CMS data/simulation)",
                "fontsize": 22,
                "data": False,
                "exp": "",
                "com": self.config_inst.campaign.ecm,
                "lumi": round(0.001 * self.config_inst.x.luminosity.get("nominal"), 2)
            }
            mplhep.cms.label(**cms_label_kwargs)

            if len(sfhist.axes.size) == 2:
                sfhist.plot2d(ax=ax)
            else:
                sfhist.plot1d(ax=ax, yerr=False, label=f"{self.trigger} scale factors")

            plt.legend()
            plt.tight_layout()

            outputs["trigger_sf_plot"].dump(
                fig,
                formatter="mpl",
            )


####################################################################################################
#
# Calcululate alpha factors
#
####################################################################################################


class CalculateAlphaFactors(
    HistogramsUserSingleShiftBase,
):
    """
    Loads histograms containing the trigger informations in MC for MC truth and orthogonal measurements
    and calculates the alpha factors. Orthogonal measurements are soon to be implemented in the
    weight producer, this task uses categories at the moment.
    """

    trigger = luigi.Parameter(
        default="allEvents",
        description="The trigger to calculate the scale factors for, must be set; default: allEvents",
    )

    variable_settings = MultiSettingsParameter(
        default=DotDict(),
        significant=False,
        description="parameter for changing different variable settings; format: "
        "'var1,option1=value1,option3=value3:var2,option2=value2'; options implemented: "
        "rebin; can also be the key of a mapping defined in 'variable_settings_groups; "
        "default: value of the 'default_variable_settings' if defined, else empty default",
        brace_expand=True,
    )

    def output(self):
        return {
            "trigger_a_plot": self.target(f"{self.trigger}_{self.variables[0]}_a_plot.pdf"),
            }

    def run(self):

        outputs = self.output()

        hists = {}

        for dataset in self.datasets:
            for variable in self.variables:
                h_in = self.load_histogram(dataset, variable)
                h_in = self.slice_histogram(h_in, self.processes, self.categories, self.shift)

                # apply variable settings
                for var in self.variable_settings:
                    if var in h_in.axes.name:
                        h_in = h_in[{var: hist.rebin(int(self.variable_settings[var]["rebin"]))}]
                    else:
                        logger.warning(f"Variable {var} not found in histogram, skipping rebinning")

                if variable in hists.keys():
                    hists[variable] += h_in
                else:
                    hists[variable] = h_in

        # calculate efficiencies
        efficiencies = calculate_efficiencies(hists[self.variables[0]][:, 0, 0, ..., :], self.trigger)  # category, process, shift, variable, bin # noqa
        # calculate scale factors and store them as a correctionlib evaluator
        scale_factors = safe_div(efficiencies[0], efficiencies[1])

        sfhist = hist.Hist(*hists[self.variables[0]][0, 0, 0, ..., 0].axes, data=scale_factors)
        sfhist.name = f"alpha_{self.trigger}_{self.variables[0]}"
        sfhist.label = "out"

        # plot 1D or 2D scalefactors
        if len(sfhist.axes.size) <= 2:
            # use CMS plotting style
            plt.style.use(mplhep.style.CMS)
            fig, ax = plt.subplots()
            cms_label_kwargs = {
                "ax": ax,
                "llabel": "Private work (CMS data/simulation)",
                "fontsize": 22,
                "data": False,
                "exp": "",
                "com": self.config_inst.campaign.ecm,
                "lumi": round(0.001 * self.config_inst.x.luminosity.get("nominal"), 2)
            }
            mplhep.cms.label(**cms_label_kwargs)

            if len(sfhist.axes.size) == 2:
                sfhist.plot2d(ax=ax)
            else:
                sfhist.plot1d(ax=ax, yerr=False, label=f"{self.trigger} alpha factors")

            plt.legend()
            plt.tight_layout()

            outputs["trigger_a_plot"].dump(
                fig,
                formatter="mpl",
            )
