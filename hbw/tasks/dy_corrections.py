# coding: utf-8

"""
Tasks to study and create weights for DY events.
"""

from __future__ import annotations

from functools import cached_property

import gzip

import law
import luigi

from collections import defaultdict

from columnflow.tasks.framework.base import Requirements
from hbw.tasks.histograms import HistogramsUserSingleShiftBase
# from columnflow.tasks.framework.histograms import HistogramsUserSingleShiftBase
from columnflow.util import maybe_import, dev_sandbox
from columnflow.tasks.framework.mixins import (
    # CalibratorClassesMixin, SelectorClassMixin, ReducerClassMixin, ProducerClassesMixin, HistProducerClassMixin,
    # CategoriesMixin, ShiftSourcesMixin, HistHookMixin, MLModelsMixin,
    DatasetsProcessesMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow

from hbw.tasks.base import HBWTask

hist = maybe_import("hist")
np = maybe_import("numpy")
special = maybe_import("scipy.special")
plt = maybe_import("matplotlib.pyplot")


# use scipy erf function definition
def window(x, r, s):
    """
    x: dependent variable (i.g., dilep_pt)
    r: regime boundary between two fit functions
    s: sign of erf function (+1 to active second fit function, -1 to active first fit function)
    """

    sci_erf = 0.5 * (special.erf(s * 0.1 * (x - r)) + 1)

    return sci_erf


# Fit funciton: Gaussian * constant connected by erf windows
def fit_function2(x, c, n, mu, sigma, a, r):

    """
    A fit function.
    x: dependent variable (i.g., dilep_pt)
    c: Gaussian offset
    n: Gaussian normalization
    mu and sigma: Gaussian parameters
    a and b: slope parameters
    r: regime boundary between Guassian and linear fits
    """

    gauss = c + (n * (1 / sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2))
    pol = a

    return window(x, r, -1) * gauss + window(x, r, 1) * pol


def get_fit2_args(*params, return_str=False):
    if return_str:
        # build string representation
        c2, n2, mu2, sigma2, a, r1 = params
        gauss2 = f"(({c2})+(({n2})*(1/{sigma2})*exp(-0.5*((x-{mu2})/{sigma2})^2)))"
        pol = f"({a})"
        w1 = f"(0.5*(erf(-0.1*(x-{r1}))+1))"
        w2 = f"(0.5*(erf(0.1*(x-{r1}))+1))"
        return f"({w1}*{gauss2} + {w2}*{pol})"
    else:
        start_values = [1, 1, 10, 3, 1, 50]
        lower_bounds = [0.5, 0, 0, 0, 0, 0]
        upper_bounds = [1.5, 10, 50, 20, 2, 60]
        return start_values, lower_bounds, upper_bounds


# Attach it
fit_function2.get_fit_args = get_fit2_args


# Fit function: Gaussian * Gaussian * linear connected by erf windows
def fit_function3(x, c1, n1, mu1, sigma1,  # Gaussian 1
                c2, n2, mu2, sigma2,  # Gaussian 2
                a, b,                 # Linear
                r1, r2):              # Boundaries

    """
    A fit function.
    x: dependent variable (i.g., dilep_pt)
    c: Gaussian offset
    n: Gaussian normalization
    mu and sigma: Gaussian parameters
    a and b: slope parameters
    r: regime boundary between Guassian and linear fits
    """

    # Region 1: Gaussian 1 (x < r1)
    gauss1 = c1 + n1 * (1 / sigma1) * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    w1 = window(x, r1, -1)

    # Region 2: Gaussian 2 (r1 < x < r2)
    gauss2 = c2 + n2 * (1 / sigma2) * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    w2 = window(x, r1, 1) * window(x, r2, -1)

    # Region 3: Linear (x > r2)
    pol = a + b * x
    w3 = window(x, r2, 1)

    return w1 * gauss1 + w2 * gauss2 + w3 * pol


# Attach helper to get params or string representation
def get_fit3_args(*params, return_str=False):
    if return_str:
        # build string representation
        c1, n1, mu1, sigma1, c2, n2, mu2, sigma2, a, b, r1, r2 = params
        gauss1 = f"(({c1})+(({n1})*(1/{sigma1})*exp(-0.5*((x-{mu1})/{sigma1})^2)))"
        gauss2 = f"(({c2})+(({n2})*(1/{sigma2})*exp(-0.5*((x-{mu2})/{sigma2})^2)))"
        pol = f"(({a})+({b})*x)"
        w1 = f"(0.5*(erf(-0.1*(x-{r1}))+1))"
        w2 = f"(0.5*(erf(0.1*(x-{r1}))+1))*(0.5*(erf(-0.1*(x-{r2}))+1))"
        w3 = f"(0.5*(erf(0.1*(x-{r2}))+1))"
        return f"({w1}*{gauss1} + {w2}*{gauss2} + {w3}*{pol})"
    else:
        start_values = [1, 1, 3, 10, 1, 1, 30, 30, 1, 0, 15, 50]
        lower_bounds = [0.6, 0, 0, 0, 0.8, 0, 10, 3, 0, -1, 0, 15]
        upper_bounds = [1.2, 10, 10, 20, 1.2, 10, 70, 50, 2, 2, 20, 60]
        return start_values, lower_bounds, upper_bounds


# Attach it
fit_function3.get_fit_args = get_fit3_args


# Fit function Gaussian * Gaussian * constant connected by erf windows
def fit_function4(x, c1, n1, mu1, sigma1,
                c2, n2, mu2, sigma2,
                a, r1, r2):
    gauss1 = c1 + n1 * (1 / sigma1) * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    w1 = window(x, r1, -1)

    gauss2 = c2 + n2 * (1 / sigma2) * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    w2 = window(x, r1, 1) * window(x, r2, -1)

    pol = a
    w3 = window(x, r2, 1)

    return w1 * gauss1 + w2 * gauss2 + w3 * pol


# Attach helper to get params or string representation
def get_fit4_args(*params, return_str=False):
    if return_str:
        # build string representation
        c1, n1, mu1, sigma1, c2, n2, mu2, sigma2, a, r1, r2 = params
        x = "x"
        gauss1 = f"(({c1})+(({n1})*(1/{sigma1})*exp(-0.5*(({x}-{mu1})/{sigma1})^2)))"
        gauss2 = f"(({c2})+(({n2})*(1/{sigma2})*exp(-0.5*(({x}-{mu2})/{sigma2})^2)))"
        pol = f"({a})"
        w1 = f"(0.5*(erf(-0.1*({x}-{r1}))+1))"
        w2 = f"(0.5*(erf(0.1*({x}-{r1}))+1))*(0.5*(erf(-0.1*({x}-{r2}))+1))"
        w3 = f"(0.5*(erf(0.1*({x}-{r2}))+1))"
        return f"({w1}*{gauss1} + {w2}*{gauss2} + {w3}*{pol})"
    else:
        start_values = [1, 1, 0.5, 10, 1, 1, 30, 30, 1, 15, 50]
        lower_bounds = [0.6, 0, 0, 0, 0.8, 0, 10, 3, 0, -1, 15]
        upper_bounds = [1.2, 1, 1, 20, 1.2, 10, 70, 50, 2, 20, 60]  # mu1 fixed to supress first Gaussian
        return start_values, lower_bounds, upper_bounds


# Attach it
fit_function4.get_fit_args = get_fit4_args


# Fit function: erf + Gaussian * constant connceted by another erf
def fit_function5(x, c2, n2, mu2, sigma2,  # Gaussian 2
                a,                # Linear constant
                r1, r2):              # Boundaries

    """
    A fit function.
    x: dependent variable (i.g., dilep_pt)
    c: Gaussian offset
    n: Gaussian normalization
    mu and sigma: Gaussian parameters
    a and b: slope parameters
    r: regime boundary between Guassian and linear fits
    """

    w1 = window(x, r1, -1)

    # Region 2: Gaussian 2 (r1 < x < r2)
    gauss2 = c2 + n2 * (1 / sigma2) * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    w2 = window(x, r1, 1) * window(x, r2, -1)

    # Region 3: Linear (x > r2)
    pol = a
    w3 = window(x, r2, 1)

    return w1 + w2 * gauss2 + w3 * pol


def get_fit5_args(*params, return_str=False):
    if return_str:
        # build string representation
        c2, n2, mu2, sigma2, a, r1, r2 = params
        gauss2 = f"(({c2})+(({n2})*(1/{sigma2})*exp(-0.5*((x-{mu2})/{sigma2})^2)))"
        print(f"gauss2: {gauss2}")
        pol = f"({a})"
        w1 = f"(0.5*(erf(-0.1*(x-{r1}))+1))"
        w2 = f"(0.5*(erf(0.1*(x-{r1}))+1))*(0.5*(erf(-0.1*(x-{r2}))+1))"
        w3 = f"(0.5*(erf(0.1*(x-{r2}))+1))"
        return f"({w1} + {w2}*{gauss2} + {w3}*{pol})"
    else:
        start_values = [1, 1, 10, 3, 1, 15, 50]
        lower_bounds = [0.5, 0, 0, 0, 0, 0, 15]
        upper_bounds = [1.5, 10, 50, 20, 2, 20, 60]
        return start_values, lower_bounds, upper_bounds


# Attach it
fit_function5.get_fit_args = get_fit5_args


def erf_window(x, x0, s, sign=1):
    """Smooth window using erf.

    sign=1 => turn-on window
    sign=-1 => turn-off window
    """
    return 0.5 * (1 + sign * special.erf((x - x0) / (np.sqrt(2) * s)))


def fit_function9(x, c1, n1, mu1, sigma1,
                c2, n2, mu2, sigma2,
                a, r1, r2, s1, s2):
    gauss1 = c1 + n1 * (1 / sigma1) * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    w1 = erf_window(x, r1, s1, -1)

    gauss2 = c2 + n2 * (1 / sigma2) * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    w2 = erf_window(x, r1, s1, 1) * erf_window(x, r2, s2, -1)

    pol = a
    w3 = erf_window(x, r2, s2, 1)

    return w1 * (-1) * gauss1 + w2 * (-1) * gauss2 + w3 * pol


def get_fit9_args(*params, return_str=False):
    if return_str:
        # build string representation
        c1, n1, mu1, sigma1, c2, n2, mu2, sigma2, a, r1, r2, s1, s2 = params
        x = "x"
        gauss1 = f"(({c1})+(({n1})*(1/{sigma1})*exp(-0.5*(({x}-{mu1})/{sigma1})^2)))"
        gauss2 = f"(({c2})+(({n2})*(1/{sigma2})*exp(-0.5*(({x}-{mu2})/{sigma2})^2)))"
        pol = f"({a})"
        w1 = f"(0.5*(1-erf(({x}-{r1})/(sqrt(2)*{s1}))))"
        w2 = f"(0.5*(1+erf(({x}-{r1})/(sqrt(2)*{s1}))))*(0.5*(1-erf(({x}-{r2})/(sqrt(2)*{s2}))))"
        w3 = f"(0.5*(1+erf(({x}-{r2})/(sqrt(2)*{s2}))))"
        return f"({w1}*(-1)*{gauss1} + {w2}*(-1)*{gauss2} + {w3}*{pol})"
    else:
        start_values = []
        lower_bounds = []
        upper_bounds = []
        fit_args = {
            "c1": [-0.8, -1.6, 1.0],
            "n1": [0.5, 0.01, 1.],
            "mu1": [0.1, 0.001, 7.],
            "sigma1": [1.5, 0.1, 30.],
            "c2": [-0.8, -2, 0.0],
            "n2": [10, 0.1, 30],
            "mu2": [40, 10, 90],
            "sigma2": [40, 10, 80],
            "a": [0.8, 0.6, 3],
            "r1": [10, 2, 30],
            "r2": [80, 30, 150],
            "s1": [10, 1, 10],
            "s2": [5, 1, 50],
        }
        for param in fit_args:
            start_values.append(fit_args[param][0])
            lower_bounds.append(fit_args[param][1])
            upper_bounds.append(fit_args[param][2])
        return start_values, lower_bounds, upper_bounds


fit_function9.get_fit_args = get_fit9_args


# Define a safe division function to avoid division by zero
def safe_div(num, den):
    return np.where(
        (num > 0) & (den > 0),  # NOTE: in high n_jet bins there are negative weights.
        num / den,
        1.0,
    )


# Helper fucntion to get the rate factor for each n_jet bin individually
def get_rate_factor(h, ptll_var):

    nominator = h["data"][{ptll_var: sum}].values() - h["MC_subtract"][{ptll_var: sum}].values()
    denominator = h["MC_corr_process"][{ptll_var: sum}].values()
    njet_rate = safe_div(nominator, denominator)

    return njet_rate


# build fit function string
def get_fit_str(
        njet: int,
        njet_overflow: int,
        rate_factor: float,
        h: hist.Hist,
        fit_function,
        era: str,
        outputs) -> dict:
    from scipy import optimize

    # Check if number of jets are valid
    total_n_jet = np.arange(1, 12)
    if njet not in total_n_jet:
        raise ValueError(f"Invalid njets value {njet}, expected int between 1 and 11.")

    # Helper function to plot the fit and string function
    def plot_fit_func(fit_func, fit_string, param_fit, rate_factor, ratio_values, njet, era, outputs):
        #  plot the fit function and the string function (as sanity)
        def str_function(x, fit_str):
            from scipy import special
            fit_str = fit_str.replace("^2", "**2")
            str_function = eval(fit_str, {"x": x, "exp": np.exp, "sqrt": np.sqrt, "erf": special.erf})
            return str_function

        s = np.linspace(0, 400, 1000)
        y = [(rate_factor_for_fit * fit_func(v, *param_fit)) for v in s]
        z = [str_function(v, fit_string) for v in s]
        fig, ax = plt.subplots()
        ax.plot(s, y, color="grey", label="Fit function")
        ax.plot(s, z, color="black", linestyle="--", label="String function")
        ax.errorbar(
            bin_centers,
            ratio_values,
            yerr=ratio_err,
            fmt=".",
            color="black",
            linestyle="none",
            ecolor="black",
            elinewidth=0.5,
        )
        ax.legend(loc="upper right")
        ax.set_xscale("log")
        ax.set_xlabel(r"$p_{T,ll}\;[\mathrm{GeV}]$")
        ax.set_ylabel("Ratio")
        ax.set_ylim(0.4, 1.5)
        ax.set_title(f"Fit function for NLO {era}, N(jets) =  {njet}")
        ax.grid(True)
        outputs.child(f"Fit_function_{era}_njet_{njet}.pdf", type="f").dump(fig, formatter="mpl")

    # Define binning
    if njet < njet_overflow:
        jet_tuple = (njet, njet + 1)
    elif njet >= njet_overflow:
        jet_tuple = (njet_overflow, 11)

    data_h = h["data"][hist.loc(jet_tuple[0]):hist.loc(jet_tuple[1]), ...]
    dy_h = h["MC_corr_process"][hist.loc(jet_tuple[0]):hist.loc(jet_tuple[1]), ...]
    mc_h = h["MC_subtract"][hist.loc(jet_tuple[0]):hist.loc(jet_tuple[1]), ...]

    # get bin centers
    bin_centers = dy_h.axes[-1].centers

    # get histogram values and errors, summing over njets axis
    dy_values = (dy_h.values()).sum(axis=0)
    data_values = (data_h.values()).sum(axis=0)
    mc_values = (mc_h.values()).sum(axis=0)
    dy_err = (dy_h.variances()).sum(axis=0)
    data_err = (data_h.variances()).sum(axis=0)
    mc_err = (mc_h.variances()).sum(axis=0)

    rate_factor_for_fit = (data_values.sum(axis=0) - mc_values.sum(axis=0)) / dy_values.sum(axis=0)

    # calculate (data-mc)/dy ratio and its error
    ratio_values = (data_values - mc_values) / dy_values
    ratio_err = (1 / dy_values) * np.sqrt(data_err + mc_err + (ratio_values)**2 * dy_err)

    # fill nans/infs and negative errors with 0
    ratio_values = np.nan_to_num(ratio_values, nan=0.0)
    ratio_values = np.where(np.isinf(ratio_values), 0.0, ratio_values)
    ratio_values = np.where(ratio_values < 0, 0.0, ratio_values)
    ratio_err = np.nan_to_num(ratio_err, nan=0.0)
    ratio_err = np.where(np.isinf(ratio_err), 0.0, ratio_err)
    ratio_err = np.where(ratio_err < 0, 0.0, ratio_err)

    # Retrieve fit parameters
    starting_values, lower_bounds, upper_bounds = fit_function.get_fit_args()

    # Fit
    param_fit, _ = optimize.curve_fit(
        fit_function,
        bin_centers,
        ratio_values / rate_factor_for_fit,
        p0=starting_values,
        method="trf",
        sigma=np.maximum(ratio_err, 1e-5),
        absolute_sigma=True,
        bounds=(lower_bounds, upper_bounds),
    )
    print("Fit parameters:", param_fit)

    # Build string
    fit_func_str = fit_function.get_fit_args(*param_fit, return_str=True)
    fit_str = f"{rate_factor}*{fit_func_str}"
    print("String of fit function:", fit_str)

    # Plot Fit and String fucntion with data/MC ratio as cross check
    plot_fit_func(fit_function, fit_str, param_fit, rate_factor, ratio_values, njet, era, outputs["fit_function"])

    return fit_str


def compute_weight_data(task: ComputeDYWeights, h: hist.Hist) -> dict:
    """
    Compute the DY weight data from the given histogram *h* that supposed to contain the following axis:

        - n_jet (int)
        - ptll (float)

    The returned dictionary follows a nested structure:

        year -> syst -> (min_njet, max_njet) -> [(lower_bound, upper_bound, formula), ...]

        - *year* is one of 2022, 2022EE, 2023, 2023BPix
        - *syst* is one of nominal, up, down (maybe up1, down1, up2, down2, ... in case of multiple sources)
        - *(min_njet, max_njet)* is a tuple of integers defining the right-exclusive range of njets
        - the inner-most list contains 3-tuples with lower and upper bounds of a formula
    """
    # prepare constants
    outputs = task.output()
    inf = float("inf")
    era = "_".join([
        f"{config_inst.campaign.x.year}{config_inst.campaign.x.postfix}" for config_inst in task.config_insts
    ])

    # initialize fit dictionary
    fit_dict = {
        era: {
            "nominal": {},
            # "up": {},
            # "down": {},
        },
    }

    ptll_var = "ptll_for_dy_corr"
    rate_factor_lst = get_rate_factor(h, ptll_var)
    for njet in np.arange(1, 11):
        rate_factor = rate_factor_lst[njet]
        if njet > 7:  # NOTE: DOn't use rate_factor for njet > 7, due to neg weights / no DY in those bins
            rate_factor = rate_factor_lst[7]
        print(f"Rate factor for njet={njet}: {rate_factor}")
        fit_str = get_fit_str(njet, task.njet_overflow, rate_factor, h, fit_function9, era, outputs)

        fit_dict[era]["nominal"][(njet, njet + 1)] = [(0.0, inf, fit_str)]

    return fit_dict


class DYCorrBase(
    HBWTask,
    HistogramsUserSingleShiftBase,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    Base class for DY correction tasks.
    """
    # TODO: enable MultiConfig in HistogramsUserSingleShiftBase
    single_config = False
    njet_overflow = luigi.IntParameter(
        default=2,
        description="Overflow bin for the n_jet variable in the fits",
        significant=True,
    )

    # NOTE: we assume that the corrected process is always the same for all configs
    corrected_process = DatasetsProcessesMixin.processes.copy(
        default=("dy",),
        description="Processes to consider for the scale factors",
        add_default_to_description=True,
    )
    processes = DatasetsProcessesMixin.processes_multi.copy(
        default=((
            "vv", "w_lnu", "st",
            "dy_m10to50", "dy_m50toinf",
            "tt", "ttv", "h", "data",
        ),),
        description="Processes to use for the DY corrections",
        add_default_to_description=True,
    )
    variables = HistogramsUserSingleShiftBase.variables.copy(
        default=("n_jet-ptll_for_dy_corr",),
        description="Variables to use for the DY corrections",
        add_default_to_description=True,
    )
    categories = HistogramsUserSingleShiftBase.categories.copy(
        default=("dycr__2mu",),
        description="Categories to use for the DY corrections",
        add_default_to_description=True,
    )
    shift = HistogramsUserSingleShiftBase.shift.copy(
        default="nominal",
        description="Shift to use for the DY corrections",
        add_default_to_description=True,
    )
    hist_producer = HistogramsUserSingleShiftBase.hist_producer.copy(
        default="met70",
        description="Histogram producer to use for the DY corrections",
        add_default_to_description=True,
    )
    ml_models = HistogramsUserSingleShiftBase.ml_models.copy(
        default=(),
        description="ML models to use for the DY corrections",
        add_default_to_description=True,
    )

    def create_branch_map(self):
        return {0: None}


class ComputeDYWeights(DYCorrBase):
    """
    Example command:

        > law run hbw.ComputeDYWeights \
            --config 22pre_v14 \
            --processes sm_nlo_data_bkg \
            --version prod8_dy \
            --hist-producer no_dy_weight \
            --categories mumu__dy__os \
            --variables njets-dilep_pt
    """

    reqs = Requirements(
        RemoteWorkflow.reqs,
        HistogramsUserSingleShiftBase.reqs,
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # datasets/processes instances
        self.process_insts = {
            config_inst: list(map(config_inst.get_process, processes))
            for config_inst, processes in zip(self.config_insts, self.processes)
        }
        self.dataset_insts = {
            config_inst: list(map(config_inst.get_dataset, datasets))
            for config_inst, datasets in zip(self.config_insts, self.datasets)
        }

        # only one category is allowed right now
        if len(self.categories) != 1:
            raise ValueError(f"{self.task_family} requires exactly one category, got {self.categories}")
        # ensure that the category matches a specific pattern: starting with "ee"/"mumu" and ending in "os"
        if "dycr" not in self.categories[0]:
            raise ValueError(f"category must start with 'dycr' to derive dy crorections, got {self.categories[0]}")
        self.category_inst = self.config_insts[0].get_category(self.categories[0])

        # only one variable is allowed
        if len(self.variables) != 1:
            raise ValueError(f"{self.task_family} requires exactly one variable, got {self.variables}")
        self.variable = self.variables[0]
        # for now, variable must be "n_jet-ptll_for_dy_corr"
        if self.variable != "n_jet-ptll_for_dy_corr":
            raise ValueError(f"variable must be 'n_jet-ptll_ptll_for_dy_corr', got {self.variable}")

        # Only one processes is allowed to correct for
        # NOTE: might have to change due to MultiConfig
        if len(self.corrected_process) != 1:
            raise ValueError(
                f"Only one corrected process is supported for njet corrections. Got {self.corrected_process}",
            )

    def store_parts(self):
        parts = super().store_parts()

        parts.insert_before("version", "njetflow", str(self.njet_overflow))
        return parts

    def output(self):
        return {
            "dy_weight": self.target("dy_weight_data.pkl"),
            "fit_function": self.target("fit_function", dir=True),
        }

    @cached_property
    def corrected_proc_inst(self):
        return {
            config_inst: config_inst.get_process(self.corrected_process[0])
            for config_inst in self.config_insts
        }

    @property
    def corrected_sub_proc_insts(self):
        return {
            config_inst: [proc for proc, _, _ in proc_inst.walk_processes(include_self=True)]
            for config_inst, proc_inst in self.corrected_proc_inst.items()
        }

    @cached_property
    def subtract_proc_insts(self):
        return {
            config_inst: [
                proc for proc in self.process_insts[config_inst]
                if proc.is_mc and proc not in self.corrected_sub_proc_insts[config_inst]
            ]
            for config_inst in self.config_insts
        }

    @cached_property
    def data_proc_insts(self):
        return {
            config_inst: [
                proc for proc in self.process_insts[config_inst]
                if proc.is_data and proc not in self.corrected_sub_proc_insts[config_inst]
            ]
            for config_inst in self.config_insts
        }

    def run(self):
        shifts = ["nominal", self.shift]
        hists = defaultdict(dict)

        for variable in self.variables:
            for i, config_inst in enumerate(self.config_insts):
                hist_per_config = None
                # sub_processes = self.processes[i]
                for dataset in self.datasets[i]:
                    # sum over all histograms of the same variable and config
                    if hist_per_config is None:
                        hist_per_config = self.load_histogram(
                            config=config_inst, dataset=dataset, variable=variable,
                        )
                    else:
                        h = self.load_histogram(
                            config=config_inst, dataset=dataset, variable=variable,
                        )
                        hist_per_config += h

                # slice histogram per config according to the sub_processes and categories

                slice_kwargs = {
                    "histogram": hist_per_config,
                    "config_inst": config_inst,
                    "categories": self.categories,
                    "shifts": shifts,
                    "reduce_axes": True,
                }
                hist_data = self.slice_histogram(
                    processes=self.data_proc_insts[config_inst],
                    **slice_kwargs,
                )
                hist_mc_subtract = self.slice_histogram(
                    processes=self.subtract_proc_insts[config_inst],
                    **slice_kwargs,
                )
                hist_corrected = self.slice_histogram(
                    processes=self.corrected_sub_proc_insts[config_inst],
                    **slice_kwargs,
                )

                if variable in hists.keys():
                    hists[variable]["data"] += hist_data
                    hists[variable]["MC_subtract"] += hist_mc_subtract
                    hists[variable]["MC_corr_process"] += hist_corrected
                else:
                    hists[variable]["data"] = hist_data
                    hists[variable]["MC_subtract"] = hist_mc_subtract
                    hists[variable]["MC_corr_process"] = hist_corrected

        # compute the dy weight data
        hists_corr = hists[self.variable]
        dy_weight_data = compute_weight_data(self, hists_corr)

        # store them
        self.output()["dy_weight"].dump(dy_weight_data, formatter="pickle")


class ExportDYWeights(DYCorrBase):
    """
    Example command:

        > law run hbt.ExportDYWeights \
            --configs 22pre_v14,22post_v14,... \
            --version prod8_dy
    """

    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh")

    reqs = Requirements(
        RemoteWorkflow.reqs,
        ComputeDYWeights=ComputeDYWeights,
    )

    def requires(self):
        return {
            config: self.reqs.ComputeDYWeights.req(self)
            for config in self.configs
        }

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["dy_correction_weight"] = {
            config: self.reqs.ComputeDYWeights.req(self)
            for config in self.configs
        }
        return reqs

    def output(self):
        return self.target("dy_correction_weight.json.gz")

    def run(self):
        import correctionlib.schemav2 as cs
        from hbw.tasks.create_clib_file import create_dy_weight_correction

        # load all weight data per config and merge them into a single dictionary
        inputs = self.input()
        configs = self.configs
        for config in configs:
            dy_weight_data = law.util.merge_dicts((
                inputs[config]["dy_weight"].load(formatter="pickle")
            ))

        print(dy_weight_data)

        # create and save the correction set
        cset = cs.CorrectionSet(
            schema_version=2,
            description="Corrections derived for the hh2bbWW analysis.",
            corrections=[create_dy_weight_correction(dy_weight_data)],
        )
        with self.output().localize("w") as outp:
            outp.path += ".json.gz"
            with gzip.open(outp.abspath, "wt") as f:
                f.write(cset.model_dump_json(exclude_unset=True))

            # validate the content
            law.util.interruptable_popen(f"correction summary {outp.abspath}", shell=True)
