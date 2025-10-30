# coding: utf-8

"""
Tasks to create trigger scale factors and corresponding plots.

Current issues:
- we require a lot of memory to produce/merge/load our histograms (~9GB), loading takes ~7 Minutes per 2d scale factor
    - this is likely the case due to the super fine input binning (400 bins per lepton pt axis), needed to be
    able to optimise the binning on the spot for the scale factors
    - we could instead define one variable per channel with the final binning, which would be much faster but
    less flexible
    - maybe there's some memory leak - histograms take only ~500MB per dataset and we merge them into one
      histogram per weight producer and process (2x2=4 histograms), so should be much less memory intensive
      (especially since we apply rebinning + slicing early on)

TODOs:
- move plotting code into separate functions
- allow plotting SF + uncertainties in one go

"""

from __future__ import annotations

import law
import luigi

from collections import defaultdict

from columnflow.tasks.framework.base import Requirements
from hbw.tasks.histograms import HistogramsUserSingleShiftBase
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.tasks.framework.mixins import (
    DatasetsProcessesMixin,
)
from columnflow.tasks.histograms import MergeHistograms
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.types import TYPE_CHECKING

from hbw.tasks.base import HBWTask

from hbw.hist_util import apply_rebinning_edges
from hbw.trigger.trigger_util import (
    calculate_efficiencies,
    calc_sf_uncertainty,
    optimise_binning1d,
    optimise_binning2d,
)

logger = law.logger.get_logger(__name__)

np = maybe_import("numpy")
if TYPE_CHECKING:
    hist = maybe_import("hist")


class ComputeTriggerSF(
    HBWTask,
    HistogramsUserSingleShiftBase,
    DatasetsProcessesMixin,
    law.LocalWorkflow,
):
    """
    Base task to calculate trigger scale factors.
    NOTE: should we make this to HTCondorWorkflow? This task sometimes uses quite some memory (~9GB).
    """
    # new parameters specfific to this task
    # trigger = luigi.Parameter(
    #     default="allEvents",
    #     description="The trigger to calculate the scale factors for, must be set; default: allEvents",
    # )
    hist_producers = law.CSVParameter(
        default=("no_trig_sf", "dl_orth2_with_l1_seeds"),
        description="Weight producers to use for plotting",
    )
    suffix = luigi.Parameter(
        default="",
        description="Suffix to append to the output file names",
    )

    # these have been parameters in Balduin's setup
    envelope_var = ""
    bins_optimised = True
    premade_edges = True
    # TODO: just make uncertainty + SF plot in one go instead
    plot_uncertainties = luigi.BoolParameter(
        default=False,
        description="Plot the uncertainties, instead of the scale factors",
    )
    # plot_uncertainties = True

    # set default parameters specific to this task
    processes = DatasetsProcessesMixin.processes_multi.copy(
        default=((
            "data_met", "sf_bkg_reduced",
        ),),
        description="Processes to use for the Trigger SF calculation",
        add_default_to_description=True,
    )
    variables = HistogramsUserSingleShiftBase.variables.copy(
        default=("trg_lepton0_pt-trg_lepton1_pt-trig_ids",),
        description="Variables to use for the Trigger SF calculation",
        add_default_to_description=True,
    )
    categories = HistogramsUserSingleShiftBase.categories.copy(
        default=("2mu", "2e", "emu"),
        description="Categories to use for the Trigger SF calculation",
        add_default_to_description=True,
    )
    shift = HistogramsUserSingleShiftBase.shift.copy(
        default="nominal",
        description="Shift to use for the Trigger SF calculation",
        add_default_to_description=True,
    )
    ml_models = HistogramsUserSingleShiftBase.ml_models.copy(
        default=(),
        description="ML models to use for the Trigger SF calculation",
        add_default_to_description=True,
    )
    reducer = HistogramsUserSingleShiftBase.reducer.copy(
        default="triggersf",
        description="Reducer to use for the Trigger SF calculation",
        add_default_to_description=True,
    )
    producers = HistogramsUserSingleShiftBase.producers.copy(
        default=("event_weights", "pre_ml_cats", "trigger_prod_dls"),
        description="Producers to use for the Trigger SF calculation",
        add_default_to_description=True,
    )

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeHistograms=MergeHistograms,
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

    @property
    def hist_producers_repr(self):
        return "_".join(self.hist_producers)

    @property
    def trigger(self):
        return self.trigger_func(self.branch_data.category)

    def trigger_func(self, category):
        # category = self.branch_data.category
        return {
            "2e": "ee",
            "2mu": "mm",
            "emu": "mixed",
        }[category]

    pre_edges = {
        "2e": {
            "lepton0_pt": [0., 25., 32., 38., 45., 50., 55., 65., 75., 85., 95., 120., 200., 400.],
            "lepton1_pt": [0., 15., 23., 32., 38., 55.] + [95., 120., 400.],
        },
        "2mu": {
            "lepton0_pt": [0., 25., 29., 35., 40.] + [40. + (i * 5.) for i in range(1, 7)] + [77., 90., 120., 400.],
            "lepton1_pt": [0., 15., 22., 29., 35., 45.] + [90., 120., 400.],
        },
        "emu": {
            "lepton0_pt": [0., 25., 30., 35., 40.] + [40. + (i * 5.) for i in range(1, 9)] + [90., 100., 120., 150., 180., 400.],  # noqa: E501
            "lepton1_pt": [0., 15., 20., 25., 30., 35., 45., 60., 80., 100., 120., 400.],
        },
    }

    def edges(self, variable, category):
        if not any(var in variable for var in ("lepton0_pt", "lepton1_pt")):
            logger.warning(f"No rebinning setup for variable {variable} in trigger SF")
            return None
            # raise ValueError(f"Variable {variable} not supported for trigger SF binning")
        if category not in ("2e", "2mu", "emu"):
            raise ValueError(f"Category {category} not supported for trigger SF binning")
        variable = "lepton0_pt" if "lepton0_pt" in variable else "lepton1_pt"
        return self.pre_edges[category][variable]

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        parts.insert_before("version", "weights", f"weights_{self.hist_producers_repr}")
        return parts

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name, "variable": var_name})
            for cat_name in sorted(self.categories)
            for var_name in sorted(self.variables)
        ]

    @property
    def config_inst(self):
        return self.config_insts[0]

    def requires(self):
        datasets = [self.datasets] if self.single_config else self.datasets
        return {
            hist_producer: {
                config_inst.name: {
                    d: self.reqs.MergeHistograms.req_different_branching(
                        self,
                        config=config_inst.name,
                        dataset=d,
                        hist_producer=hist_producer,
                        branch=-1,
                        _prefer_cli={"variables"},
                    )
                    for d in datasets[i]
                    if config_inst.has_dataset(d)
                }
                for i, config_inst in enumerate(self.config_insts)
            }
            for hist_producer in self.hist_producers
        }

    def output(self):
        category = self.branch_data.category
        trigger = self.trigger_func(category)
        variable = self.branch_data.variable

        # is_2d = len(variable.split("-")) == 2

        outp = {
            "trigger_scale_factors": self.target(f"sf_{trigger}_{variable}{self.suffix}.json"),
            "trigger_sf_plot":
            self.target(f"{category}_{trigger}_{variable}_sf_plot{self.suffix}.pdf"),
        }
        # if is_2d and self.plot_uncertainties:
        #     outp["trigger_sf_uncertainty_plot"] = self.target(
        #         f"{category}_{trigger}_{variable}_sf_uncertainty{self.suffix}.pdf"
        #     )
        return outp

    def run(self):
        with self.publish_step(f"Loading variable {self.branch_data.variable} in category {self.branch_data.category}"):
            hists = self.prepare_hists()
        self.rest(hists)

    def prepare_hists(self) -> dict[str, dict[str, hist.Hist]]:
        """
        Helper function that loads and sums all required histograms for one variable and category.
        Returns a nested dictionary with structure:
        {
            hist_producer1: {
                process1: histogram,
                process2: histogram,
                ...
            },
            hist_producer2: {
                process1: histogram,
                process2: histogram,
                ...
            },
            ...
        }

        :return: Nested dictionary of histograms
        """
        # histogram data per process
        hists = defaultdict(dict)
        shifts = ["nominal", self.shift]
        variable = self.branch_data.variable
        category = self.branch_data.category

        for hist_producer, inputs in self.input().items():
            for i, config_inst in enumerate(self.config_insts):
                hist_per_config = None
                # sub_processes = self.processes[i]
                for dataset in self.datasets[i]:
                    # sum over all histograms of the same variable and config
                    if hist_per_config is None:
                        hist_per_config = self.load_histogram(
                            inputs=inputs, config=config_inst, dataset=dataset, variable=variable,
                        )
                    else:
                        hist_per_config += self.load_histogram(
                            inputs=inputs, config=config_inst, dataset=dataset, variable=variable,
                        )

                for var in variable.split("-"):
                    if "trig_ids" in var:
                        continue
                    # apply rebinning
                    # NOTE: we could also sequentialize this task, but then we'd need to rebin+select category later
                    edges = self.edges(var, category)
                    if edges is None:
                        continue
                    hist_per_config = apply_rebinning_edges(
                        h=hist_per_config,
                        axis_name=var,
                        edges=edges,
                    )

                processes = self.process_insts[config_inst]
                for process_inst in processes:
                    hist_per_config_and_process = self.slice_histogram(
                        histogram=hist_per_config,
                        config_inst=config_inst,
                        processes=process_inst,
                        categories=category,
                        shifts=shifts,
                        reduce_axes=True,
                    )

                    if hist_producer in hists and process_inst.name in hists[hist_producer]:
                        hists[hist_producer][process_inst.name] += hist_per_config_and_process
                    else:
                        hists[hist_producer][process_inst.name] = hist_per_config_and_process

        return hists

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
        # pick processes from 1st config
        processes = self.processes[0]

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
            axes = hists[self.hist_producers[1]][processes[0]].axes.name
            h2 = hists[self.hist_producers[1]][processes[0]]
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
                    for proc in processes:
                        h1 = hists[hist_producer][proc][{
                            self.envelope_var: slice(slice_borders[region][0], slice_borders[region][1])}]

                        hists2[hist_producer][proc] = h1[..., 0:len(h1.axes[-2].centers):sum, :]

                sf_envelope[region], sf_uncs[region], _, _ = self.calc_sf_and_unc(hists2, envelope=True)

            for hist_producer in self.hist_producers:
                for proc in processes:
                    hists[hist_producer][proc] = hists[hist_producer][proc][{self.envelope_var: sum}]

        for hist_producer in self.hist_producers:
            # calculate efficiencies. process, shift, variable, bin
            efficiencies[hist_producer], efficiency_unc[hist_producer] = calculate_efficiencies(
                hists[hist_producer], self.trigger,
            )

        # calculate scale factors, second weight producer is used
        scale_factors = np.nan_to_num(
            efficiencies[self.hist_producers[1]][processes[0]] /
            efficiencies[self.hist_producers[1]][processes[1]],
            nan=1,
            posinf=1,
            neginf=1,
        )
        scale_factors[scale_factors == 0] = 1
        # calculate alpha factors
        alpha_factors = np.nan_to_num(
            efficiencies[self.hist_producers[0]][processes[1]] /
            efficiencies[self.hist_producers[1]][processes[1]],
            nan=1,
            posinf=1,
            neginf=1,
        )

        # only use the efficiencies and uncertainties of the second weight producer
        efficiencies = efficiencies[self.hist_producers[1]]
        efficiency_unc = efficiency_unc[self.hist_producers[1]]
        # calculate scale factor uncertainties, only statistical uncertainties are considered right now
        uncertainties = calc_sf_uncertainty(
            efficiencies, efficiency_unc, alpha_factors,
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

    def rest(self, hists):
        import correctionlib
        import correctionlib.convert

        # import awkward as ak
        # import scipy
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import mplhep
        import hist

        outputs = self.output()

        # prepare config objects
        variable_tuple = self.variable_tuples[self.variables[0]]
        variable_insts = [
            self.config_inst.get_variable(var_name)
            for var_name in variable_tuple
        ]
        # pick processes from 1st config
        processes = self.processes[0]

        # ####################################################

        old_axes = hists[self.hist_producers[0]][processes[0]][..., 0].axes

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
                "Binning optimisation not implemented for histograms "
                "with more than 2 dimensions, using default binning",
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

            sfhist = hist.Hist(*hists[self.hist_producers[0]][processes[0]][..., 0].axes, data=scale_factors)
            sfhist.name = f"sf_{self.trigger}_{self.variables[0]}"
            sfhist.label = "out"

            nominal_scale_factors = correctionlib.convert.from_histogram(sfhist)
            nominal_scale_factors.description = f"{self.trigger} scale factors, binned in {self.variables[0]}"

            # set overflow bins behavior (default is to raise an error when out of bounds)
            nominal_scale_factors.data.flow = "clamp"

            # add uncertainties
            sfhist_up = hist.Hist(
                *hists[self.hist_producers[0]][processes[0]][..., 0].axes, data=scale_factors + uncertainties,
            )
            sfhist_up.name = f"sf_{self.trigger}_{self.variables[0]}_up"
            sfhist_up.label = "out"

            upwards_scale_factors = correctionlib.convert.from_histogram(sfhist_up)
            upwards_scale_factors.description = f"{self.trigger} scale factors, binned in {self.variables[0]}, upwards variation"  # noqa: E501

            # set overflow bins behavior (default is to raise an error when out of bounds)
            upwards_scale_factors.data.flow = "clamp"

            sfhist_down = hist.Hist(
                *hists[self.hist_producers[0]][processes[0]][..., 0].axes, data=scale_factors - uncertainties,
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
                            sfhist_unc = hist.Hist(*hists[self.hist_producers[0]][processes[0]][..., 0].axes, data=uncertainties)  # noqa: E501
                            sfhist_unc.plot2d(ax=ax, cmap="viridis")
                        else:
                            sfhist.values()[uncertainties == 0.0] = None
                            artists = sfhist.plot2d(ax=ax, cmap="viridis")
                            artists[0].set_clim(0.85, 1.15)

                        ax.plot([1, 400], [1, 400], linestyle="dashed", color="gray")
                        ax.set_xscale("log")
                        ax.set_yscale("log")

                        ax.set_xlim(25, 400)
                        ax.set_ylim(15, 400)
                        ax.set_xticks([25, 50, 100, 150, 250, 400])
                        ax.set_yticks([25, 50, 100, 150, 250, 400])

                        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
                        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
                        ax.set_xlabel(r"Leading lepton $p_T$ [GeV]")
                        ax.set_ylabel(r"Subleading lepton $p_T$ [GeV]")

                        label = self.config_inst.get_category(self.branch_data.category).label
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
                    proc_label1 = label_dict[processes[1]]
                    proc_label0 = label_dict[processes[0]]
                    # proc_label0 = self.config_inst.get_process(processes[0]).label
                    # proc_label1 = self.config_inst.get_process(processes[1]).label
                    # proc_label0 = proc_label0[:-4] if "DL" in proc_label0 else proc_label0
                    # proc_label1 = proc_label1[:-4] if "DL" in proc_label1 else proc_label1
                    if self.envelope_var != "":
                        raise Exception("1d triggerSF with envelope_var - not currently wanted")
                        ax.errorbar(x=sfhist.axes[0].centers, y=sfhist.values(), xerr=2.4, fmt=",", label="nominal", lw=2, color="gray") # noqa
                        ax.errorbar(
                            x=sfhist.axes[0].centers, y=sf_envelope["up"], yerr=sf_uncs["up"], fmt="o", color="darkorange",  # noqa
                            label=f"{self.config_inst.get_variable(self.envelope_var).x_title}"
                            fr"$\geq${self.half_point_label}",
                            markersize=5,
                        )
                        ax.errorbar(
                            x=sfhist.axes[0].centers, y=sf_envelope["down"], yerr=sf_uncs["down"], fmt="o", color="steelblue",  # noqa
                            label=f"{self.config_inst.get_variable(self.envelope_var).x_title}<{self.half_point_label}",
                            markersize=5,
                        )
                        rax.errorbar(
                            x=sfhist.axes[0].centers, y=sf_envelope["up"] / sfhist.values(), yerr=sf_uncs["up"] / sfhist.values(), fmt="o", color="darkorange",  # noqa
                            label=f"{self.config_inst.get_variable(self.envelope_var).x_title}"
                            fr"$\geq${self.half_point_label}",
                            markersize=5,
                        )
                        rax.errorbar(
                            x=sfhist.axes[0].centers, y=sf_envelope["down"] / sfhist.values(), yerr=sf_uncs["down"] / sfhist.values(), fmt="o", color="steelblue",  # noqa
                            label=f"{self.config_inst.get_variable(self.envelope_var).x_title}<{self.half_point_label}",
                            markersize=5,
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
                            handles[::-1], labels[::-1], loc="upper right", handletextpad=-0.2, ncol=2, fontsize=24,
                        )
                        # ax.set_xlim(0, 200)
                        # ax.set_xlabel(r"Leading lepton $p_T$ / GeV")
                    else:
                        logger.info("1d triggerSF without envelope_var")
                        ax.errorbar(
                            x=sfhist.axes[0].centers, y=efficiencies[processes[0]],
                            yerr=efficiency_unc[processes[0]],
                            xerr=sfhist.axes[0].widths / 2,
                            fmt="o", label=f"{proc_label0}")
                        ax.errorbar(
                            x=sfhist.axes[0].centers, y=efficiencies[processes[1]],
                            yerr=efficiency_unc[processes[1]],
                            xerr=sfhist.axes[0].widths / 2,
                            fmt="o", label=f"{proc_label1}")
                        rax.errorbar(
                            x=sfhist.axes[0].centers, y=sfhist.values(),
                            yerr=uncertainties, fmt="o",
                            xerr=sfhist.axes[0].widths / 2,
                            label=f"rel. unc = {uncertainties[:2] / sfhist.values()[:2]}",
                            # color="red",
                        )
                        rax.axhline(y=1.0, linestyle="dashed", color="gray")
                        rax_kwargs = {
                            "ylim": (0.82, 1.18),  # leading lep
                            # "ylim": (0.92, 1.08),  # subleading, others
                            "xscale": "log",
                            # "ylabel": "Scale factors",
                            "ylabel": "Data / MC",
                            "xlabel": f"{variable_insts[0].x_title} [GeV]",
                            "yscale": "linear",
                        }
                        if "pt" in self.branch_data.variable:
                            rax_kwargs["xlim"] = (25, 400)
                            rax_kwargs["xscale"] = "log"
                            # Set custom tick locations for log scale
                            rax.set(**rax_kwargs)
                            rax.set_xticks([25, 50, 100, 200, 400])
                            rax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
                            rax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
                        else:
                            rax_kwargs["xscale"] = "linear"
                            rax.set(**rax_kwargs)

                        ax.set_ylabel("Efficiency")
                        ax.set_ylim(0.68, 1.18)
                        ax.legend(fontsize=24)

                    label = self.config_inst.get_category(self.branch_data.category).label
                    ax.annotate(label, xy=(0.05, 0.85), xycoords="axes fraction",
                                fontsize=20)

                cms_label_kwargs = {
                    "ax": ax,
                    "llabel": "Work in progress",
                    "fontsize": 22,
                    "data": False,
                    # "exp": "",
                    "com": self.config_inst.campaign.ecm,
                    "lumi": round(0.001 * sum([
                        config_inst.x.luminosity.get("nominal")
                        for config_inst in self.config_insts
                    ]), 2),
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
                target_uncertainty2=0.04,
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
                sfhist.plot2d(ax=ax)
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
                "lumi": round(0.001 * sum([
                    config_inst.x.luminosity.get("nominal")
                    for config_inst in self.config_insts
                ]), 2),
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
