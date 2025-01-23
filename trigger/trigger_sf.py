# coding: utf-8

"""
Tasks to calculate and save trigger scale factors.
"""

from __future__ import annotations

import luigi
import law
import order as od

from columnflow.tasks.framework.base import Requirements
from columnflow.util import maybe_import, DotDict
from columnflow.tasks.framework.histograms import (
    HistogramsUserSingleShiftBase,
)
from columnflow.tasks.framework.parameters import MultiSettingsParameter
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.histograms import MergeHistograms

logger = law.logger.get_logger(__name__)

hist = maybe_import("hist")
np = maybe_import("numpy")
scipy = maybe_import("scipy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


def safe_div(num, den, default=1.0):
    """
    Safely divide two arrays, setting the result to:
    - num / den where num >= 0 and den > 0
    - 0 where num == 0 and den == 0
    - 1 where num >= 0 and den == 0
    """
    return np.where(
        (num == 0) & (den == 0),
        default,
        np.where(
            (num >= 0) & (den > 0),
            num / den,
            1.0
        )
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


def calc_efficiency_errors(num, den):
    """
    Calculate the error on an efficiency given the numerator and denominator histograms.
    """

    efficiency = np.nan_to_num(num.values() / den.values(), nan=0, posinf=1, neginf=0)

    if np.any(efficiency > 1):
        logger.warning(
            "Some efficiencies for are greater than 1",
        )
    elif np.any(efficiency < 0):
        logger.warning(
            "Some efficiencies for are less than 0",
        )

    # Use the variance to scale the numerator and denominator to remove an average weight,
    # this reduces the errors for rare processes to more realistic values
    num_scale = np.nan_to_num(num.values() / num.variances(), nan=1)
    den_scale = np.nan_to_num(den.values() / den.variances(), nan=1)

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

    # stacking errors
    errors = np.concatenate(
        (np.expand_dims(error_low, axis=1), np.expand_dims(error_high, axis=1)),
        axis=1,
    )
    errors = errors.T

    return errors


def calc_sf_uncertainty(efficiencies: dict, errors: dict, alpha: np.ndarray):
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
        (sym_errors[0] / efficiencies[1]) ** 2 + (efficiencies[0] * sym_errors[1] / efficiencies[1] ** 2) ** 2  + (1 - alpha) ** 2  # noqa
    )

    return np.nan_to_num(uncertainty, nan=0, posinf=1, neginf=0)


def calculate_efficiencies(
        h: hist.Hist,
        trigger: str,
) -> dict:
    """
    Calculates the efficiencies for the different triggers.
    """
    efficiencies = {}
    efficiency_unc = {}
    # loop over processes
    for proc in range(h.axes[0].size):

        efficiency = np.nan_to_num(h[proc, ..., hist.loc(trigger)].values() / h[proc, ..., 0].values(),
                                   nan=0, posinf=1, neginf=0
                                   )

        if np.any(efficiency > 1):
            logger.warning(
                "Some efficiencies for are greater than 1, errorbars are capped at zero",
            )
        elif np.any(efficiency < 0):
            logger.warning(
                "Some efficiencies for are less than 0, errorbars are capped at zero",
            )

        efficiencies[proc] = efficiency

        efficiency_unc[proc] = calc_efficiency_errors(h[proc, ..., hist.loc(trigger)], h[proc, ..., 0])

    return efficiencies, efficiency_unc


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

    weight_producers = law.CSVParameter(
        default=("default",),
        description="Weight producers to use for plotting",
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

    suffix = luigi.Parameter(
        default="",
        description="Suffix to append to the output file names",
    )

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeHistograms=MergeHistograms,
    )

    def requires(self):
        return {
            weight_producer: {
                d: self.reqs.MergeHistograms.req(
                    self,
                    dataset=d,
                    branch=-1,
                    weight_producer=weight_producer,
                    _exclude={"branches"},
                    _prefer_cli={"variables"},
                )
                for d in self.datasets
            }
            for weight_producer in self.weight_producers
        }

    @property
    def weight_producers_repr(self):
        return "_".join(self.weight_producers)

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "weights", f"weights_{self.weight_producers_repr}")
        return parts

    def load_histogram(
        self,
        weight_producer: str,
        dataset: str | od.Dataset,
        variable: str | od.Variable,
    ) -> hist.Hist:
        """
        Helper function to load the histogram from the input for a given dataset and variable.

        :param weight_producer: The weight producer name.
        :param dataset: The dataset name or instance.
        :param variable: The variable name or instance.
        :return: The loaded histogram.
        """
        if isinstance(dataset, od.Dataset):
            dataset = dataset.name
        if isinstance(variable, od.Variable):
            variable = variable.name
        histogram = self.input()[weight_producer][dataset]["collection"][0]["hists"].targets[variable].load(formatter="pickle")  # noqa
        return histogram

    def output(self):
        return {
            "trigger_scale_factors": self.target(f"sf_{self.trigger}_{self.variables[0]}{self.suffix}.json"),
            "trigger_sf_plot": self.target(f"{self.trigger}_{self.variables[0]}_sf_plot{self.suffix}.pdf"),
            }

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["merged_hists"] = self.requires_from_branch()

        return reqs

    def run(self):
        import correctionlib
        import correctionlib.convert

        outputs = self.output()

        hists = {
            self.weight_producers[0]: {},
            self.weight_producers[1]: {},
        }

        for weight_producer in self.weight_producers:
            for dataset in self.datasets:
                for variable in self.variables:
                    h_in = self.load_histogram(weight_producer, dataset, variable)
                    h_in = self.slice_histogram(h_in, self.processes, self.categories, self.shift)
                    h_in = h_in[{"category": sum}]
                    # apply variable settings
                    for var in self.variable_settings:
                        if var in h_in.axes.name:
                            h_in = h_in[{var: hist.rebin(int(self.variable_settings[var]["rebin"]))}]
                        else:
                            logger.warning(f"Variable {var} not found in histogram, skipping rebinning")
                    if variable in hists[weight_producer].keys():
                        hists[weight_producer][variable] += h_in
                    else:
                        hists[weight_producer][variable] = h_in

        efficiencies = {
            self.weight_producers[0]: {},
            self.weight_producers[1]: {},
        }
        efficiency_unc = {
            self.weight_producers[0]: {},
            self.weight_producers[1]: {},
        }

        for weight_producer in self.weight_producers:
            # calculate efficiencies. process, shift, variable, bin
            efficiencies[weight_producer], efficiency_unc[weight_producer] = calculate_efficiencies(
                hists[weight_producer][self.variables[0]][:, 0, ..., :], self.trigger
                )

        # calculate scale factors, second weight producer is used
        scale_factors = np.nan_to_num(
            efficiencies[self.weight_producers[1]][0] / efficiencies[self.weight_producers[1]][1],
            nan=1,
            posinf=1,
            neginf=1,
            )
        # calculate alpha factors
        alpha_factors = np.nan_to_num(
            efficiencies[self.weight_producers[0]][1] / efficiencies[self.weight_producers[1]][1],
            nan=1,
            posinf=1,
            neginf=1,
            )

        # only use the efficiencies and uncertainties of the second weight producer
        efficiencies = efficiencies[self.weight_producers[1]]
        efficiency_unc = efficiency_unc[self.weight_producers[1]]
        # calculate scale factor uncertainties, only statistical uncertainties are considered right now
        uncertainties = calc_sf_uncertainty(
            efficiencies, efficiency_unc, alpha_factors
            )

        sfhist = hist.Hist(*hists[self.weight_producers[0]][self.variables[0]][0, 0, ..., 0].axes, data=scale_factors)
        sfhist.name = f"sf_{self.trigger}_{self.variables[0]}"
        sfhist.label = "out"

        nominal_scale_factors = correctionlib.convert.from_histogram(sfhist)
        nominal_scale_factors.description = f"{self.trigger} scale factors, binned in {self.variables[0]}"

        # set overflow bins behavior (default is to raise an error when out of bounds)
        nominal_scale_factors.data.flow = "clamp"

        # add uncertainties
        sfhist_up = hist.Hist(
            *hists[self.weight_producers[0]][self.variables[0]][0, 0, ..., 0].axes, data=scale_factors + uncertainties
            )
        sfhist_up.name = f"sf_{self.trigger}_{self.variables[0]}_up"
        sfhist_up.label = "out"

        upwards_scale_factors = correctionlib.convert.from_histogram(sfhist_up)
        upwards_scale_factors.description = f"{self.trigger} scale factors, binned in {self.variables[0]}, upwards variation" # noqa

        # set overflow bins behavior (default is to raise an error when out of bounds)
        upwards_scale_factors.data.flow = "clamp"

        sfhist_down = hist.Hist(
            *hists[self.weight_producers[0]][self.variables[0]][0, 0, ..., 0].axes, data=scale_factors - uncertainties
            )
        sfhist_down.name = f"sf_{self.trigger}_{self.variables[0]}_down"
        sfhist_down.label = "out"

        downwards_scale_factors = correctionlib.convert.from_histogram(sfhist_down)
        downwards_scale_factors.description = f"{self.trigger} scale factors, binned in {self.variables[0]}, downwards variation" # noqa

        # set overflow bins behavior (default is to raise an error when out of bounds)
        downwards_scale_factors.data.flow = "clamp"

        # create correction set and store it
        cset = correctionlib.schemav2.CorrectionSet(
            schema_version=2,
            description=f"{self.trigger} scale factors",
            corrections=[
                nominal_scale_factors,
                upwards_scale_factors,
                downwards_scale_factors,
                ],
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

            if len(sfhist.axes.size) == 2:
                fig, ax = plt.subplots()
            else:
                fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0), sharex=True)
                (ax, rax) = axs

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
                ax.errorbar(
                    x=sfhist.axes[0].centers, y=efficiencies[0], yerr=efficiency_unc[0],
                    fmt="o", label=f"{self.trigger} data")
                ax.errorbar(
                    x=sfhist.axes[0].centers, y=efficiencies[1], yerr=efficiency_unc[1],
                    fmt="o", label=f"{self.trigger} MC")
                rax.errorbar(x=sfhist.axes[0].centers, y=sfhist.values(), yerr=uncertainties, fmt="o")
                rax.axhline(y=1.0, linestyle="dashed", color="gray")
                rax_kwargs = {
                    "ylim": (0.85, 1.15),
                    "ylabel": "Scale factors",
                    "xlabel": f"{sfhist.axes[0].label}",
                    "yscale": "linear",
                }
                rax.set(**rax_kwargs)
                ax.set_ylabel("Efficiency")
                ax.set_ylim(0, 1.04)

            ax.legend()
            fig.tight_layout()

            outputs["trigger_sf_plot"].dump(
                fig,
                formatter="mpl",
            )
