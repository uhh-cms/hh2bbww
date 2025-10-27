# coding: utf-8

"""
Tasks to calculate and save trigger scale factors.
"""

from __future__ import annotations

from functools import cached_property

from collections import OrderedDict, defaultdict

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
from columnflow.plotting.plot_util import apply_variable_settings

from hbw.hist_util import apply_rebinning_edges
from hbw.trigger.trigger_util import (
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

    envelope_var = luigi.Parameter(
        default="",
        description="Variable used to build regions for systematic uncertainties, default: None",
    )

    plot_uncertainties = luigi.BoolParameter(
        default=False,
        description="Plot the uncertainties, instead of the scale factors",
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

    @cached_property
    def inputs(self):
        return self.input()

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
        histogram = self.inputs[hist_producer][dataset]["collection"][0]["hists"].targets[variable].load(formatter="pickle")  # noqa
        return histogram

    def calc_sf_and_unc(
        self,
        hists: dict,
        envelope: bool = False,
    ):
        """
        Calculate the trigger scale factors and their uncertainties.
        Also returns the efficiencies and their uncertainties used in the calculation.

        NOTE: envelope seems to be inverted in usage!
        """
        efficiencies = {
            self.hist_producers[0]: {},
            self.hist_producers[1]: {},
        }
        efficiency_unc = {
            self.hist_producers[0]: {},
            self.hist_producers[1]: {},
        }

        # calc sfs for different categories
        if not envelope:

            sf_envelope = {
                "up": {},
                "down": {},
            }
            sf_uncs = {
                "up": {},
                "down": {},
            }
            sf_diff = {}

            # get half point
            axes = hists[self.hist_producers[1]][self.processes[0]].axes.name
            h2 = hists[self.hist_producers[1]][self.processes[0]]
            for var in axes:
                if self.envelope_var not in var and "process" not in var:
                    h2 = h2[{var: sum}]

            half_point = int(np.where(np.cumsum(h2.values()) >= (np.sum(h2.values()) / 2))[0][0])
            self.half_point = half_point

            # check if half_point is under or over shooting
            default_ratio = h2[slice(0, half_point)].sum().value / h2[slice(half_point, len(h2.values()))].sum().value
            shift_ratio = h2[slice(0, half_point + 1)].sum().value / h2[slice(half_point + 1, len(h2.values()))].sum().value  # noqa: E501
            if abs(default_ratio - 1) > abs(shift_ratio - 1):
                self.half_point += 1

            self.half_point_label = int(h2.axes.edges[0][half_point])

            for region in sf_envelope.keys():
                slice_borders = {
                    "down": (0, self.half_point),
                    "up": (self.half_point, len(h2.values())),
                }
                hists2 = {
                    self.hist_producers[0]: {},
                    self.hist_producers[1]: {},
                }
                for hist_producer in self.hist_producers:
                    for proc in self.processes:
                        h1 = hists[hist_producer][proc][{
                            self.envelope_var: slice(slice_borders[region][0], slice_borders[region][1])}]

                        hists2[hist_producer][proc] = h1[..., 0:len(h1.axes[-2].centers):sum, :]

                sf_envelope[region], sf_uncs[region], _, _ = self.calc_sf_and_unc(hists2, envelope=True)

            for hist_producer in self.hist_producers:
                for proc in self.processes:
                    hists[hist_producer][proc] = hists[hist_producer][proc][{self.envelope_var: sum}]

        for hist_producer in self.hist_producers:
            # calculate efficiencies. process, shift, variable, bin
            efficiencies[hist_producer], efficiency_unc[hist_producer] = calculate_efficiencies(
                hists[hist_producer], self.trigger
            )

        # calculate scale factors, second weight producer is used
        scale_factors = np.nan_to_num(
            efficiencies[self.hist_producers[1]][self.processes[0]] /
            efficiencies[self.hist_producers[1]][self.processes[1]],
            nan=1,
            posinf=1,
            neginf=1,
        )
        scale_factors[scale_factors == 0] = 1
        # calculate alpha factors
        alpha_factors = np.nan_to_num(
            efficiencies[self.hist_producers[0]][self.processes[1]] /
            efficiencies[self.hist_producers[1]][self.processes[1]],
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
            return scale_factors, uncertainties, efficiencies, efficiency_unc, sf_env_unc, unc_with_sf_env, alpha_factors, sf_envelope, sf_uncs  # noqa
            # return scale_factors, unc_with_sf_env, efficiencies, efficiency_unc, sf_env_unc, unc_with_sf_env, alpha_factors, sf_envelope  # noqa

    def output(self):
        return {
            "trigger_scale_factors": self.target(f"sf_{self.trigger}_{self.variables[0]}{self.suffix}.json"),
            "trigger_sf_plot":
            self.target(f"{self.categories[0]}_{self.trigger}_{self.variables[0]}_sf_plot{self.suffix}.pdf"),
        }

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name, "variable": var_name})
            for cat_name in sorted(self.categories)
            for var_name in sorted(self.variables)
        ]

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

        # bunch of hard coded config
        # edges1p2 = [0., 15., 25., 32., 40., 45., 55., 65., 80., 110., 240.] # noqa
        # edges1p2 = [0., 15., 25., 32., 40., 45., 55., 65., 80., 110., 240.] # noqa
        # edges1p2 = [0., 15., 25., 32., 38., 45., 55., 65., 75., 80., 110., 240., 400.] # noqa
        # edges2p1 = [0., 15., 25., 30., 40., 65., 240., 400.] # noqa
        # edges1p2 = [0., 15., 25., 32., 38., 45., 55., 65., 75., 80., 110., 240., 400.] # noqa
        # edges2p1 = [0., 15., 20., 25., 30., 35., 40., 65., 240., 400.] # noqa

        # mmedges1p1 = [  0., 15., 25.,  37.,  47.,  55.,  66.,  81.,  96., 111., 146., 240.] # noqa
        # mmedges2p1 = [  0.,  15.,  28.,  37.,  47., 240.] # noqa

        # mixedges1 = [0., 15., 25., 35., 46., 56., 67., 80., 95., 168., 240.]
        # mixedges2 = [0., 15., 25., 35., 46., 240.]

        # basic 5 GeV binning
        mixedges1 = [0., 25.] + [25. + (i * 5.) for i in range(1, 40)]
        mixedges2 = [0., 15.] + [15. + (i * 5.) for i in range(1, 40)]

        edges1p2 = [0., 25.] + [25. + (i * 5.) for i in range(1, 40)]
        edges2p1 = [0., 15.] + [15. + (i * 5.) for i in range(1, 40)]

        mmedges1p1 = [0., 25.] + [25. + (i * 5.) for i in range(1, 40)]
        mmedges2p1 = [0., 15.] + [15. + (i * 5.) for i in range(1, 40)]

        # binning for 2D scale factors
        mixedges1 = [0., 25., 30., 35., 40.] + [40. + (i * 5.) for i in range(1, 9)] + [90., 100., 120., 150., 180., 400.]  # noqa: E501
        mixedges2 = [0., 15., 20., 25., 30., 35., 45., 60., 80., 100., 120., 400.]

        edges1p2 = [0., 25., 32., 38., 45., 50., 55., 65., 75., 85., 95., 120., 200., 400.]
        edges2p1 = [0., 15., 23., 32., 38., 55.] + [95., 120., 400.]

        mmedges1p1 = [0., 25., 29., 35., 40.] + [40. + (i * 5.) for i in range(1, 7)] + [77., 90., 120., 400.]
        mmedges2p1 = [0., 15., 22., 29., 35., 45.] + [90., 120., 400.]

        edgesee = {"lepton0_pt": edges1p2, "lepton1_pt": edges2p1}
        edgesmu = {"lepton0_pt": mmedges1p1, "lepton1_pt": mmedges2p1}
        edgesmix = {"lepton0_pt": mixedges1, "lepton1_pt": mixedges2}
        pre_edges = {"ee": edgesee, "mm": edgesmu, "mixed": edgesmix}

        # Load the histograms
        # get the shifts to extract and plot
        plot_shifts = law.util.make_list(self.config_inst.get_shift(self.shift))

        # prepare config objects
        variable_tuple = self.variable_tuples[self.variables[0]]
        variable_insts = [
            self.config_inst.get_variable(var_name)
            for var_name in variable_tuple
        ]
        category_inst = self.config_inst.get_category(self.categories[0])
        leaf_category_insts = category_inst.get_leaf_categories() or [category_inst]
        process_insts = list(map(self.config_inst.get_process, self.processes))
        sub_process_insts = {
            proc: [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        }

        # histogram data per process
        hists = defaultdict(OrderedDict)

        # NOTE: loading histograms as implemented here might not be consistent anymore with
        # how it's done in PlotVariables1D (important when datasets/shifts differ per config)
        with self.publish_step(f"Loading {self.variables[0]} in {category_inst.name}"):
            for hist_producer, inputs in self.input().items():
                for dataset, inp in inputs.items():
                    dataset_inst = self.config_inst.get_dataset(dataset)
                    h_in = self.load_histogram(hist_producer, dataset, self.variables[0])

                    # loop and extract one histogram per process
                    for process_inst in process_insts:
                        # skip when the dataset is already known to not contain any sub process
                        if not any(map(dataset_inst.has_process, sub_process_insts[process_inst])):
                            continue

                        # work on a copy
                        h = h_in.copy()

                        # axis selections
                        h = h[{
                            "process": [
                                hist.loc(p.name)
                                for p in sub_process_insts[process_inst]
                                if p.name in h.axes["process"]
                            ],
                            "category": [
                                hist.loc(c.name)
                                for c in leaf_category_insts
                                if c.name in h.axes["category"]
                            ],
                            "shift": [
                                hist.loc(s.name)
                                for s in plot_shifts
                                if s.name in h.axes["shift"]
                            ],
                        }]

                        # axis reductions
                        h = h[{"process": sum, "category": sum, "shift": sum}]

                        # add the histogram
                        if hist_producer in hists and process_inst.name in hists[hist_producer]:
                            hists[hist_producer][process_inst.name] = hists[hist_producer][process_inst.name] + h
                        else:
                            hists[hist_producer][process_inst.name] = h

                # there should be hists to plot
                if not hists:
                    raise Exception(
                        "no histograms found to plot; possible reasons:\n" +
                        "  - requested variable requires columns that were missing during histogramming\n" +
                        "  - selected --processes did not match any value on the process axis of the input histogram",
                    )

                # sort hists by process order
                hists[hist_producer] = OrderedDict(
                    (process_inst, hists[hist_producer][process_inst])
                    for process_inst in sorted(hists[hist_producer], key=process_insts.index)
                )

            hists = dict(hists)

        for key, h_dict in hists.items():
            hists[key] = apply_variable_settings(h_dict, variable_insts, self.variable_settings)[0]
            if self.premade_edges:
                for proc, h in h_dict.items():
                    for var in variable_insts[:-1]:
                        if var.name in self.variable_settings.keys():
                            if "rebin" in self.variable_settings[var.name].keys():
                                logger.warning(
                                    f"Variable {var.name} already rebinned in variable settings, rebinning using"
                                    "premade edges might not work and will be skipped. Either remove the 'rebin' option"
                                    "from the variable settings or set 'premade_edges' to False."
                                )
                                continue
                        # magic number to get the variable name "leptonX_pt" from "PREFIX_leptonX_pt"
                        if var.name[-10:] in pre_edges[self.trigger].keys():
                            hists[key][proc] = apply_rebinning_edges(
                                hists[key][proc],
                                var.name,
                                pre_edges[self.trigger][var.name[-10:]],
                            )
                        else:
                            logger.warning(
                                f"Variable {var.name} not found in pre_edges, skipping rebinning."
                            )
        # ####################################################

        old_axes = hists[self.hist_producers[0]][self.processes[0]][..., 0].axes

        # optimise 1d binning
        if len(variable_insts[:-1]) == 1:

            if not self.bins_optimised:

                hists, _ = optimise_binning1d(
                    calculator=self,
                    hists=hists,
                    target_uncertainty=0.05,
                )

                self.bins_optimised = True

        elif len(variable_insts[:-1]) > 2 and not self.bins_optimised:
            logger.warning(
                "Binning optimisation not implemented for histograms with more than 2 dimensions, using default binning"
            )
            self.bins_optimised = True

        # normal procedure for optimised binning, sf_envelope, unc_with_sf_env
        if self.bins_optimised:

            if self.envelope_var != "":
                raise Exception("triggerSF with envelope_var - not currently wanted")
                scale_factors, uncertainties, efficiencies, efficiency_unc, sf_env_unc, unc_with_sf_env, alpha_factors, sf_envelope, sf_uncs = self.calc_sf_and_unc(hists, envelope=False)  # noqa: E501
            else:
                scale_factors, uncertainties, efficiencies, efficiency_unc = self.calc_sf_and_unc(hists, envelope=True)  # noqa

            scale_factors[uncertainties == 0.0] = 1.0

            sfhist = hist.Hist(*hists[self.hist_producers[0]][self.processes[0]][..., 0].axes, data=scale_factors)
            sfhist.name = f"sf_{self.trigger}_{self.variables[0]}"
            sfhist.label = "out"

            nominal_scale_factors = correctionlib.convert.from_histogram(sfhist)
            nominal_scale_factors.description = f"{self.trigger} scale factors, binned in {self.variables[0]}"

            # set overflow bins behavior (default is to raise an error when out of bounds)
            nominal_scale_factors.data.flow = "clamp"

            # add uncertainties
            sfhist_up = hist.Hist(
                *hists[self.hist_producers[0]][self.processes[0]][..., 0].axes, data=scale_factors + uncertainties
            )
            sfhist_up.name = f"sf_{self.trigger}_{self.variables[0]}_up"
            sfhist_up.label = "out"

            upwards_scale_factors = correctionlib.convert.from_histogram(sfhist_up)
            upwards_scale_factors.description = f"{self.trigger} scale factors, binned in {self.variables[0]}, upwards variation"  # noqa: E501

            # set overflow bins behavior (default is to raise an error when out of bounds)
            upwards_scale_factors.data.flow = "clamp"

            sfhist_down = hist.Hist(
                *hists[self.hist_producers[0]][self.processes[0]][..., 0].axes, data=scale_factors - uncertainties
            )
            sfhist_down.name = f"sf_{self.trigger}_{self.variables[0]}_down"
            sfhist_down.label = "out"

            downwards_scale_factors = correctionlib.convert.from_histogram(sfhist_down)
            downwards_scale_factors.description = f"{self.trigger} scale factors, binned in {self.variables[0]}, downwards variation"  # noqa: E501

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
                    fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[10, 7], hspace=0), sharex=True)
                    (ax, rax) = axs
                    # fig, ax = plt.subplots()

                if len(sfhist.axes.size) == 2:
                    logger.info("2d plotting")
                    # raise Exception("2d plotting - I do not want this ATM")
                    plot2d = True
                    if plot2d:
                        if self.plot_uncertainties:
                            uncertainties[uncertainties == 0.0] = None
                            sfhist_unc = hist.Hist(*hists[self.hist_producers[0]][self.processes[0]][..., 0].axes, data=uncertainties)  # noqa: E501
                            sfhist_unc.plot2d(ax=ax, cmap="viridis")
                            # for i in range(len(sfhist_unc.axes[0].centers)):
                            #     for j in range(len(sfhist_unc.axes[1].centers)):
                            #         unc = uncertainties[i, j]  # rel_unc[i, j]
                            #         value = uncertainties[i, j]
                            #         if value < 0.055:
                            #             color = "white"
                            #         else:
                            #             color = "black"
                            #         if unc > 0:
                            #             if sfhist_unc.axes[0].centers[i] < 400 and sfhist_unc.axes[1].centers[j] < 400:  # noqa: E501
                            #                 ax.text(sfhist_unc.axes[0].centers[i], sfhist_unc.axes[1].centers[j],
                            #                         f'{value:.2f}',  # \n$\\pm${unc:.2f}',
                            #                         ha='center', va='center', color=color, fontsize=14)
                        else:
                            sfhist.values()[uncertainties == 0.0] = None
                            artists = sfhist.plot2d(ax=ax, cmap="viridis")
                            # cbar = ax.figure.colorbar(artists[0], ax=ax)
                            artists[0].set_clim(0.85, 1.15)
                            # cbar.set_clim(0.85, 1.15)
                        # Annotate each bin with its relative uncertainty
                            # for i in range(len(sfhist.axes[0].centers)):
                            #     for j in range(len(sfhist.axes[1].centers)):
                            #         unc = uncertainties[i, j]  # rel_unc[i, j]
                            #         value = scale_factors[i, j]
                            #         if unc > 0:
                            #             if sfhist.axes[0].centers[i] < 400 and sfhist.axes[1].centers[j] < 400:
                            #                 if value < 0.97:
                            #                     color = "white"
                            #                 else:
                            #                     color = "black"
                            #                 ax.text(sfhist.axes[0].centers[i], sfhist.axes[1].centers[j],
                            #                         f'{value:.2f}',  # \n$\\pm${unc:.2f}',
                            #                         ha='center', va='center', color=color, fontsize=14)
                        ax.plot([1, 400], [1, 400], linestyle="dashed", color="gray")
                        ax.set_xscale("log")
                        ax.set_yscale("log")

                        ax.set_xlim(25, 400)
                        ax.set_ylim(15, 400)
                        ax.set_xticks([25, 50, 100, 150, 250, 400])
                        ax.set_yticks([25, 50, 100, 150, 250, 400])

                        # ax.set_ylim(15, 60)
                        # ax.set_xlim(15, 200)
                        # ax.set_yticks([20, 30, 40, 50, 60])
                        # ax.set_xticks([25, 50, 100, 150])

                        # ax.set_ylim(15, 100)
                        # ax.set_xlim(25, 350)
                        # ax.set_ylim(15, 80)
                        # ax.set_xlim(25, 110)
                        # ax.set_yticks([20, 30, 40, 50, 60, 70])
                        # ax.set_xticks([25, 30, 40, 50, 60, 70, 80, 90, 100])

                        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
                        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
                        ax.set_xlabel(r"Leading lepton $p_T$ [GeV]")
                        ax.set_ylabel(r"Subleading lepton $p_T$ [GeV]")

                        label = self.config_inst.get_category(self.categories[0]).label
                        ax.annotate(label, xy=(0.05, 0.95), xycoords="axes fraction",
                                    fontsize=20)
                    else:
                        fig, axs = plt.subplots(4, 1, gridspec_kw=dict(hspace=0), sharex=True)
                        for key in [4, 3, 2, 1]:
                            axs[key - 1].errorbar(
                                x=sfhist[:, key].axes.centers[0], y=sfhist[:, key].values(),
                                yerr=uncertainties[:, key],
                                fmt="o",
                                label=f"{sfhist[0, :].axes.edges[0][key]}<lep2 $p_T$<{sfhist[0, :].axes.edges[0][key + 1]}")  # noqa: E501

                            axs[key - 1].set_ylim(0.70, 1.13)
                            axs[key - 1].legend(loc="lower right")
                            axs[key - 1].plot([15, 400], [1, 1], linestyle="dashed", color="gray")

                        axs[0].set_ylabel("Data/MC")
                        axs[-1].set_xlabel(r"Leading lepton $p_T$ / GeV")
                        ax = axs[0]
                else:
                    logger.info("1d plotting")
                    # 1d plotting
                    label_dict = {
                        "data_met": r"$\varepsilon_{\text{Data}}^{\text{meas.}}$",
                        "sf_bkg": r"$\varepsilon_{\text{MC}}^{\text{meas.}}$",
                        "sf_bkg_reduced": r"$\varepsilon_{\text{MC}}^{\text{meas.}}$",
                    }
                    # trig_label = r"$e\mu$ trigger" if self.trigger == "mixed" else f"{self.trigger} trigger"
                    proc_label1 = label_dict[self.processes[1]]
                    proc_label0 = label_dict[self.processes[0]]
                    # proc_label0 = self.config_inst.get_process(self.processes[0]).label
                    # proc_label1 = self.config_inst.get_process(self.processes[1]).label
                    # proc_label0 = proc_label0[:-4] if "DL" in proc_label0 else proc_label0
                    # proc_label1 = proc_label1[:-4] if "DL" in proc_label1 else proc_label1
                    if self.envelope_var != "":
                        raise Exception("1d triggerSF with envelope_var - not currently wanted")
                        ax.errorbar(x=sfhist.axes[0].centers, y=sfhist.values(), xerr=2.4, fmt=",", label="nominal", lw=2, color="gray") # noqa
                        ax.errorbar(
                            x=sfhist.axes[0].centers, y=sf_envelope["up"], yerr=sf_uncs["up"], fmt="o", color="darkorange",  # noqa
                            label=f"{self.config_inst.get_variable(self.envelope_var).x_title}"
                            fr"$\geq${self.half_point_label}",
                            markersize=5
                        )
                        ax.errorbar(
                            x=sfhist.axes[0].centers, y=sf_envelope["down"], yerr=sf_uncs["down"], fmt="o", color="steelblue",  # noqa
                            label=f"{self.config_inst.get_variable(self.envelope_var).x_title}<{self.half_point_label}",
                            markersize=5
                        )
                        rax.errorbar(
                            x=sfhist.axes[0].centers, y=sf_envelope["up"] / sfhist.values(), yerr=sf_uncs["up"] / sfhist.values(), fmt="o", color="darkorange",  # noqa
                            label=f"{self.config_inst.get_variable(self.envelope_var).x_title}"
                            fr"$\geq${self.half_point_label}",
                            markersize=5
                        )
                        rax.errorbar(
                            x=sfhist.axes[0].centers, y=sf_envelope["down"] / sfhist.values(), yerr=sf_uncs["down"] / sfhist.values(), fmt="o", color="steelblue",  # noqa
                            label=f"{self.config_inst.get_variable(self.envelope_var).x_title}<{self.half_point_label}",
                            markersize=5
                        )
                        # handles, labels = rax.get_legend_handles_labels()
                        # rax.legend(handles[::-1], labels[::-1], loc="upper left", handletextpad=-0.2, ncol=2)
                        # rax.legend(loc="upper left", handletextpad=-0.2, ncol=2)

                        rax.axhline(y=1.0, linestyle="dashed", color="gray")
                        rax_kwargs = {
                            # "ylim": (0.82, 1.18),  # leading lep
                            "ylim": (0.92, 1.08),  # subleading, others
                            "xlim": (0, 100),
                            "ylabel": "Ratio to nominal",
                            "xlabel": f"{variable_insts[0].x_title} [GeV]",
                            "yscale": "linear",
                        }
                        rax.set(**rax_kwargs)
                        ax.set_ylabel("Scalefactors")
                        ax.set_ylim(0.82, 1.18)
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend(
                            handles[::-1], labels[::-1], loc="upper right", handletextpad=-0.2, ncol=2, fontsize=24
                        )
                        # ax.set_xlim(0, 200)
                        # ax.set_xlabel(r"Leading lepton $p_T$ / GeV")
                    else:
                        logger.info("1d triggerSF without envelope_var")
                        ax.errorbar(
                            x=sfhist.axes[0].centers, y=efficiencies[self.processes[0]],
                            yerr=efficiency_unc[self.processes[0]],
                            fmt="o", label=f"{proc_label0}")
                        ax.errorbar(
                            x=sfhist.axes[0].centers, y=efficiencies[self.processes[1]],
                            yerr=efficiency_unc[self.processes[1]],
                            fmt="o", label=f"{proc_label1}")
                        rax.errorbar(
                            x=sfhist.axes[0].centers, y=sfhist.values(),
                            yerr=uncertainties, fmt="o",
                            label=f"rel. unc = {uncertainties[:2] / sfhist.values()[:2]}",
                            # color="red",
                        )
                        rax.axhline(y=1.0, linestyle="dashed", color="gray")
                        rax_kwargs = {
                            "ylim": (0.82, 1.18),  # leading lep
                            # "ylim": (0.92, 1.08),  # subleading, others
                            "xlim": (0, 200),
                            # "ylabel": "Scale factors",
                            "ylabel": "Data / MC",
                            "xlabel": f"{variable_insts[0].x_title} [GeV]",
                            "yscale": "linear",
                        }
                        rax.set(**rax_kwargs)
                        ax.set_ylabel("Efficiency")
                        ax.set_ylim(0.68, 1.18)
                        ax.legend(fontsize=24)

                    label = self.config_inst.get_category(self.categories[0]).label
                    ax.annotate(label, xy=(0.05, 0.85), xycoords="axes fraction",
                                fontsize=20)

                cms_label_kwargs = {
                    "ax": ax,
                    "llabel": "Work in progress",
                    "fontsize": 22,
                    "data": False,
                    # "exp": "",
                    "com": self.config_inst.campaign.ecm,
                    "lumi": round(0.001 * self.config_inst.x.luminosity.get("nominal"), 2)
                }
                mplhep.cms.label(**cms_label_kwargs)
                fig.tight_layout()

        # optimising the binning for 2d histograms results in a non uniform binning, a different workflow is needed
        elif len(self.variables[0].split("-")[:-1]) == 2:
            raise Exception("2d rebinning - I do not want this (use --premade-edges --bins-optimised)")

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
                        fmt="o", label=f"{x_edges[key]}<lep2pt<{x_edges[key + 1]}")

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
                ax.set_xlabel(r"Leading lepton $p_T$ [GeV]")
                ax.set_ylabel(r"Subleading lepton $p_T$ [GeV]")

            cms_label_kwargs = {
                "ax": ax,
                "llabel": "Work in progress",
                "fontsize": 22,
                "data": False,
                # "exp": "",
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
        self.publish_message(f"Trigger scale factors and plot saved to folder {outputs['trigger_sf_plot'].abspath}")
