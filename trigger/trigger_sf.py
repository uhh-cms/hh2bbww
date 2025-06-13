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
from columnflow.plotting.plot_util import use_flow_bins

from hbw.config.hist_hooks import rebin_hist
from trigger.trigger_util import (
    calculate_efficiencies,
    calc_sf_uncertainty,
    optimise_binning1d,
    optimise_binning2d,
)

logger = law.logger.get_logger(__name__)

hist = maybe_import("hist")
np = maybe_import("numpy")
ak = maybe_import("awkward")
scipy = maybe_import("scipy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


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

    hist_producers = law.CSVParameter(
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

    bins_optimised = luigi.BoolParameter(
        default=True,
        description="Optimise the binning to get relative uncertainties below 5%",
    )

    suffix = luigi.Parameter(
        default="",
        description="Suffix to append to the output file names",
    )

    premade_edges = luigi.BoolParameter(
        default=False,
        description="Use premade bin edges for the optimisation",
    )

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeHistograms=MergeHistograms,
    )

    def requires(self):
        return {
            hist_producer: {
                d: self.reqs.MergeHistograms.req(
                    self,
                    dataset=d,
                    branch=-1,
                    hist_producer=hist_producer,
                    _exclude={"branches"},
                    _prefer_cli={"variables"},
                )
                for d in self.datasets
            }
            for hist_producer in self.hist_producers
        }

    @property
    def hist_producers_repr(self):
        return "_".join(self.hist_producers)

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "weights", f"weights_{self.hist_producers_repr}")
        return parts

    def load_histogram(
        self,
        hist_producer: str,
        dataset: str | od.Dataset,
        variable: str | od.Variable,
    ) -> hist.Hist:
        """
        Helper function to load the histogram from the input for a given dataset and variable.

        :param hist_producer: The weight producer name.
        :param dataset: The dataset name or instance.
        :param variable: The variable name or instance.
        :return: The loaded histogram.
        """
        if isinstance(dataset, od.Dataset):
            dataset = dataset.name
        if isinstance(variable, od.Variable):
            variable = variable.name
        histogram = self.input()[hist_producer][dataset]["collection"][0]["hists"].targets[variable].load(formatter="pickle")  # noqa
        return histogram

    def calc_sf_and_unc(
        self,
        hists: dict,
        envelope: bool = False,
    ):
        """
        Calculate the trigger scale factors and their uncertainties.
        Also returns the efficiencies and their uncertainties used in the calculation.
        """
        efficiencies = {
            self.hist_producers[0]: {},
            self.hist_producers[1]: {},
        }
        efficiency_unc = {
            self.hist_producers[0]: {},
            self.hist_producers[1]: {},
        }

        sf_envelope = {
            "up": {},
            "down": {},
        }
        sf_diff = {}

        # calc sfs for different categories
        if not envelope:
            # get half point
            axes = hists[self.hist_producers[1]][self.variables[0]].axes.name
            h2 = hists[self.hist_producers[1]][self.variables[0]][{"process": 0}]
            for var in axes:
                if "npvs" not in var and "process" not in var:
                    h2 = h2[{var: sum}]

                half_point = int(np.where(np.cumsum(h2.values()) >= (np.sum(h2.values()) / 2))[0][0])
                self.half_point = half_point

            for region in sf_envelope.keys():
                slice_borders = {
                    "down": (0, half_point),
                    "up": (half_point, len(h2.values())),
                }
                hists2 = {
                    self.hist_producers[0]: {},
                    self.hist_producers[1]: {},
                }
                for hist_producer in self.hist_producers:
                    h1 = hists[hist_producer][self.variables[0]][{
                        "npvs": slice(slice_borders[region][0], slice_borders[region][1])}]
                    hists2[hist_producer][self.variables[0]] = h1[:, :, 0:len(h1[0, 0, :, 0].values()):sum, :]

                sf_envelope[region], _, _, _ = self.calc_sf_and_unc(hists2, envelope=True)

            for hist_producer in self.hist_producers:
                hists[hist_producer][self.variables[0]] = hists[hist_producer][self.variables[0]][{"npvs": sum}]

        for hist_producer in self.hist_producers:
            # calculate efficiencies. process, shift, variable, bin
            efficiencies[hist_producer], efficiency_unc[hist_producer] = calculate_efficiencies(
                hists[hist_producer][self.variables[0]][:, ..., :], self.trigger
            )

        # calculate scale factors, second weight producer is used
        scale_factors = np.nan_to_num(
            efficiencies[self.hist_producers[1]][0] / efficiencies[self.hist_producers[1]][1],
            nan=1,
            posinf=1,
            neginf=1,
        )
        scale_factors[scale_factors == 0] = 1
        # calculate alpha factors
        alpha_factors = np.nan_to_num(
            efficiencies[self.hist_producers[0]][1] / efficiencies[self.hist_producers[1]][1],
            nan=1,
            posinf=1,
            neginf=1,
        )

        # only use the efficiencies and uncertainties of the second weight producer
        efficiencies = efficiencies[self.hist_producers[1]]
        efficiency_unc = efficiency_unc[self.hist_producers[1]]
        # calculate scale factor uncertainties, only statistical uncertainties are considered right now
        uncertainties = calc_sf_uncertainty(
            efficiencies, efficiency_unc, alpha_factors
        )

        if not envelope:
            # symmetrise the envelope
            for key, sf in sf_envelope.items():
                sf_diff[key] = np.abs(scale_factors - sf)

            sf_env_unc = np.maximum(sf_diff["up"], sf_diff["down"])

            unc_with_sf_env = np.sqrt(uncertainties**2 + sf_env_unc**2)

        if envelope:
            return scale_factors, uncertainties, efficiencies, efficiency_unc
        else:
            return scale_factors, uncertainties, efficiencies, efficiency_unc, sf_env_unc, unc_with_sf_env, alpha_factors, sf_envelope  # noqa

    def output(self):
        return {
            "trigger_scale_factors": self.target(f"sf_{self.trigger}_{self.variables[0]}{self.suffix}.json"),
            "trigger_sf_plot":
            self.target(f"{self.categories[0]}_{self.trigger}_{self.variables[0]}_sf_plot{self.suffix}.pdf"),
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
            self.hist_producers[0]: {},
            self.hist_producers[1]: {},
        }


        edges1p2 = [0., 15., 25., 35., 44., 51., 57., 64., 80., 110., 240.] # noqa
        edges2p1 = [0., 15., 25., 35., 64., 240.] # noqa

        mmedges1p1 = [  0., 15., 25.,  37.,  47.,  55.,  66.,  81.,  96., 111., 146., 240.] # noqa
        mmedges2p1 = [  0.,  15.,  28.,  37.,  47., 240.] # noqa

        mixedges1 = [0., 15., 25., 35., 46., 56., 67., 80., 95., 168., 240.]
        mixedges2 = [0., 15., 25., 35., 46., 240.]

        edgesee = {"lepton0_pt": edges1p2, "lepton1_pt": edges2p1}
        edgesmu = {"lepton0_pt": mmedges1p1, "lepton1_pt": mmedges2p1}
        edgesmix = {"lepton0_pt": mixedges1, "lepton1_pt": mixedges2}
        pre_edges = {"ee": edgesee, "mm": edgesmu, "mixed": edgesmix}

        for hist_producer in self.hist_producers:
            for dataset in self.datasets:
                for variable in self.variables:
                    h_in = self.load_histogram(hist_producer, dataset, variable)
                    h_in = self.slice_histogram(h_in, self.processes, self.categories, self.shift)
                    h_in = h_in[{"category": sum}]
                    h_in = h_in[{"shift": sum}]
                    for var in self.variable_settings:
                        if var in h_in.axes.name:
                            # use over and underflow bins unless specified otherwise
                            if self.variable_settings[var].get("use_flow_bins", True):
                                h_in = use_flow_bins(h_in, var)
                            # rebin if specified
                            if "rebin" in self.variable_settings[var]:
                                h_in = h_in[{var: hist.rebin(int(self.variable_settings[var]["rebin"]))}]
                        else:
                            logger.warning(f"Variable {var} not found in histogram, skipping rebinning")
                    if self.premade_edges:
                        for var in h_in.axes.name[1:3]:
                            h_in = rebin_hist(h_in, var, pre_edges[self.trigger][var])
                    if variable in hists[hist_producer].keys():
                        hists[hist_producer][variable] += h_in
                    else:
                        hists[hist_producer][variable] = h_in

        from hbw.util import debugger
        debugger()
        old_axes = hists[self.hist_producers[0]][self.variables[0]][0, ..., 0].axes

        # optimise 1d binning
        if len(self.variables[0].split("-")[:-1]) == 1:

            if not self.bins_optimised:

                hists, _ = optimise_binning1d(
                    calculator=self,
                    hists=hists,
                    target_uncertainty=0.05,
                )

                self.bins_optimised = True

        elif len(self.variables[0].split("-")[:-1]) > 2 and not self.bins_optimised:
            logger.warning(
                "Binning optimisation not implemented for histograms with more than 2 dimensions, using default binning"
            )
            self.bins_optimised = True

        # normal procedure for optimised binning, sf_envelope, unc_with_sf_env
        if self.bins_optimised:

            # scale_factors, uncertainties, efficiencies, efficiency_unc, sf_env_unc, unc_with_sf_env, alpha_factors, sf_envelope = self.calc_sf_and_unc(hists, envelope=False)  # noqa
            scale_factors, uncertainties, efficiencies, efficiency_unc = self.calc_sf_and_unc(hists, envelope=True)  # noqa
            scale_factors[uncertainties == 0.0] = 1.0

            sfhist = hist.Hist(*hists[self.hist_producers[0]][self.variables[0]][0, ..., 0].axes, data=scale_factors)
            sfhist.name = f"sf_{self.trigger}_{self.variables[0]}"
            sfhist.label = "out"

            nominal_scale_factors = correctionlib.convert.from_histogram(sfhist)
            nominal_scale_factors.description = f"{self.trigger} scale factors, binned in {self.variables[0]}"

            # set overflow bins behavior (default is to raise an error when out of bounds)
            nominal_scale_factors.data.flow = "clamp"

            # add uncertainties
            sfhist_up = hist.Hist(
                *hists[self.hist_producers[0]][self.variables[0]][0, ..., 0].axes, data=scale_factors + uncertainties
            )
            sfhist_up.name = f"sf_{self.trigger}_{self.variables[0]}_up"
            sfhist_up.label = "out"

            upwards_scale_factors = correctionlib.convert.from_histogram(sfhist_up)
            upwards_scale_factors.description = f"{self.trigger} scale factors, binned in {self.variables[0]}, upwards variation" # noqa

            # set overflow bins behavior (default is to raise an error when out of bounds)
            upwards_scale_factors.data.flow = "clamp"

            sfhist_down = hist.Hist(
                *hists[self.hist_producers[0]][self.variables[0]][0, ..., 0].axes, data=scale_factors - uncertainties
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

            # plot 1D or 2D scalefactors
            if len(sfhist.axes.size) <= 2:
                # use CMS plotting style
                plt.style.use(mplhep.style.CMS)

                if len(sfhist.axes.size) == 2:
                    fig, ax = plt.subplots()
                else:
                    fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[7, 5], hspace=0), sharex=True)
                    (ax, rax) = axs
                    # fig, ax = plt.subplots()

                if len(sfhist.axes.size) == 2:
                    plot2d = True
                    if plot2d:
                        sfhist.plot2d(ax=ax, cmap="viridis")
                        # Annotate each bin with its relative uncertainty
                        for i in range(len(sfhist.axes[0].centers)):
                            for j in range(len(sfhist.axes[1].centers)):
                                unc = uncertainties[i, j]  # rel_unc[i, j]
                                value = scale_factors[i, j]
                                if unc > 0:
                                    if sfhist.axes[0].centers[i] < 400 and sfhist.axes[1].centers[j] < 400:
                                        ax.text(sfhist.axes[0].centers[i], sfhist.axes[1].centers[j],
                                                f'{value:.2f}\n$\\pm${unc:.2f}',
                                                ha='center', va='center', color='white', fontsize=12)
                        ax.plot([1, 400], [1, 400], linestyle="dashed", color="gray")
                        # ax.set_xscale("log")
                        # ax.set_yscale("log")

                        # ax.set_xlim(15, 400)
                        # ax.set_ylim(15, 400)
                        # ax.set_xticks([25, 50, 100, 150, 250, 400])
                        # ax.set_yticks([25, 50, 100, 150, 250, 400])

                        # ax.set_ylim(15, 60)
                        # ax.set_xlim(15, 200)
                        # ax.set_yticks([20, 30, 40, 50, 60])
                        # ax.set_xticks([25, 50, 100, 150])

                        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
                        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
                        ax.set_xlabel(r"Leading lepton $p_T$ / GeV")
                        ax.set_ylabel(r"Subleading lepton $p_T$ / GeV")
                    else:
                        fig, axs = plt.subplots(4, 1, gridspec_kw=dict(hspace=0), sharex=True)
                        for key in [4, 3, 2, 1]:
                            axs[key - 1].errorbar(
                                x=sfhist[:, key].axes.centers[0], y=sfhist[:, key].values(),
                                yerr=uncertainties[:, key],
                                fmt="o",
                                label=f"{sfhist[0,:].axes.edges[0][key]}<lep2 $p_T$<{sfhist[0,:].axes.edges[0][key+1]}")

                            axs[key - 1].set_ylim(0.70, 1.13)
                            axs[key - 1].legend(loc="lower right")
                            axs[key - 1].plot([15, 400], [1, 1], linestyle="dashed", color="gray")

                        axs[0].set_ylabel("Data/MC")
                        axs[-1].set_xlabel(r"Leading lepton $p_T$ / GeV")
                        ax = axs[0]
                else:
                    # trig_label = r"$e\mu$ trigger" if self.trigger == "mixed" else f"{self.trigger} trigger"
                    if "Data" in self.config_inst.get_process(self.processes[0]).label:
                        from hbw.util import debugger
                        debugger()
                        proc_label0 = r"$\varepsilon_{\text{Data}}$"
                        proc_label1 = r"$\varepsilon_{\text{MC},t\bar{t}+DY}$"
                    else:
                        proc_label1 = r"$\varepsilon_{\text{Data}}$"
                        proc_label0 = r"$\varepsilon_{\text{MC},t\bar{t}}$"
                    # proc_label0 = self.config_inst.get_process(self.processes[0]).label
                    # proc_label1 = self.config_inst.get_process(self.processes[1]).label
                    # proc_label0 = proc_label0[:-4] if "DL" in proc_label0 else proc_label0
                    # proc_label1 = proc_label1[:-4] if "DL" in proc_label1 else proc_label1
                    ax.errorbar(
                        x=sfhist.axes[0].centers, y=efficiencies[0], yerr=efficiency_unc[0],
                        fmt="o", label=f"{proc_label0}")
                    ax.errorbar(
                        x=sfhist.axes[0].centers, y=efficiencies[1], yerr=efficiency_unc[1],
                        fmt="o", label=f"{proc_label1}")
                    # rax.errorbar(x=sfhist.axes[0].centers, y=scale_factors, yerr=unc_with_sf_env, fmt="o",
                    #              color="tab:orange")
                    # rax.errorbar(x=sfhist.axes[0].centers, y=sfhist.values(), xerr=4.5, fmt=",", color="tab:grey")
                    # comb_unc = np.sqrt(uncertainties**2 + sf_env_unc**2 + (1-alpha_factors)**2)
                    # rax.errorbar(x=sfhist.axes[0].centers, y=sfhist.values(), yerr=comb_unc, fmt=",",
                    #              label="total uncertainty", elinewidth=4)
                    rax.errorbar(x=sfhist.axes[0].centers, y=sfhist.values(), yerr=uncertainties, fmt="o")
                    # rax.errorbar(x=sfhist.axes[0].centers-2, y=sfhist.values(), yerr=uncertainties, fmt=",",
                    #              label="statistical", elinewidth=4)
                    # rax.errorbar(x=sfhist.axes[0].centers, y=sfhist.values(), yerr=np.sqrt((1-alpha_factors)**2),
                    #              fmt=",", label=r"$\alpha$", elinewidth=4)
                    # rax.errorbar(x=sfhist.axes[0].centers+2, y=sfhist.values(), yerr=sf_env_unc, fmt=",",
                    #              label="npv envelope", elinewidth=4)
                    # #   # rax.errorbar(x=sfhist.axes[0].centers, y=sf_envelope["up"], yerr=0.0, fmt="o", color="lightblue",  # noqa
                    # #   #              label=f"npv>{self.half_point}", markersize=5)
                    # #   # rax.errorbar(x=sfhist.axes[0].centers, y=sf_envelope["down"], yerr=0.0, fmt="o", color="steelblue",  # noqa
                    # #   #              label=f"npv<{self.half_point}", markersize=5)
                    rax.axhline(y=1.0, linestyle="dashed", color="gray")
                    rax_kwargs = {
                        "ylim": (0.92, 1.08),
                        "xlim": (0, 200),
                        "ylabel": "Scale factors",
                        "xlabel": r"Leading lepton $p_T$ / GeV",  # f"{sfhist.axes[0].label}",  #
                        "yscale": "linear",
                    }
                    rax.legend(loc="upper left", handletextpad=-0.3, ncol=2)
                    rax.set(**rax_kwargs)
                    ax.set_ylabel("Efficiency")
                    ax.set_ylim(0, 1.5)
                    # ax.set_xlim(0, 200)
                    # ax.set_xlabel(r"Leading lepton $p_T$ / GeV")

                    label = self.config_inst.get_category(self.categories[0]).label
                    ax.annotate(label, xy=(0.05, 0.85), xycoords="axes fraction",
                                fontsize=20)

                ax.legend(fontsize=26)
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
                fig.tight_layout()

        # optimising the binning for 2d histograms results in a non uniform binning, a different workflow is needed
        elif len(self.variables[0].split("-")[:-1]) == 2:

            # optimise second axis, return accondingly optimised edges of first axis
            histslices, edges2 = optimise_binning2d(
                calculator=self,
                hists=hists,
                target_uncertainty1=0.02,
                target_uncertainty2=0.04
            )

            sliced_scale_factors = {}
            sliced_uncertainties = {}
            sliced_efficiencies = {}
            sliced_efficiency_unc = {}
            rel_unc = {}

            x_edges = hists[self.hist_producers[0]][self.variables[0]].axes[1].edges

            for key, sliced_hist in histslices.items():
                scale_factors, uncertainties, efficiencies, efficiency_unc = self.calc_sf_and_unc(sliced_hist, envelope=True)  # noqa
                scale_factors[uncertainties == 0.0] = None
                sliced_scale_factors[key] = scale_factors
                sliced_uncertainties[key] = uncertainties
                sliced_efficiencies[key] = efficiencies
                sliced_efficiency_unc[key] = efficiency_unc
                rel_unc[key] = uncertainties / scale_factors

            plt.style.use(mplhep.style.CMS)

            plot_unrolled = False
            if plot_unrolled:
                fig, axs = plt.subplots(len(x_edges) - 1, 1, gridspec_kw=dict(hspace=0), sharex=True)

                for key, h in histslices.items():
                    axs[key].errorbar(
                        x=h[self.hist_producers[0]][self.variables[0]].axes[1].centers, y=sliced_scale_factors[key],
                        yerr=sliced_uncertainties[key],
                        fmt="o", label=f"{x_edges[key]}<lep2pt<{x_edges[key+1]}")

                    axs[key].set_ylim(0.87, 1.13)
                    axs[key].legend()

                axs[int(len(x_edges) / 2)].set_ylabel("Data/MC")
                ax = axs[0]

            else:

                sfs = np.ones(shape=(400, 400))
                sfs[0, :] = None
                sfs[:, 0] = None
                for i in range(1, 400):
                    for j in range(1, 400):
                        idx = np.searchsorted(x_edges, i) - 1
                        idy = np.searchsorted(edges2[idx], j) - 1
                        if j > x_edges[idx + 1]:
                            sfs[i, j] = None
                        else:
                            sfs[i, j] = sliced_scale_factors[idx][idy]

                sfhist = hist.Hist(*old_axes, data=sfs)
                sfhist.name = f"sf_{self.trigger}_{self.variables[0]}"
                sfhist.label = "out"
                plt.style.use(mplhep.style.CMS)
                fig, ax = plt.subplots()
                sfhist.plot2d(ax=ax,)
                ax.plot([0, 150], [0, 150], linestyle="dashed", color="gray")
                ax.set_xlim(0, 150)
                ax.set_xticks([50, 100, 150])
                ax.set_ylim(0, 150)
                ax.set_yticks([50, 100, 150])
                ax.set_xlabel(r"Leading lepton $p_T$ / GeV")
                ax.set_ylabel(r"Subleading lepton $p_T$ / GeV")

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
            fig.tight_layout()
        # alternative plt nonuniformimage
        #  save outputs
        outputs["trigger_scale_factors"].dump(
            cset_json,
            formatter="json",
        )
        outputs["trigger_sf_plot"].dump(
            fig,
            formatter="mpl",
        )
