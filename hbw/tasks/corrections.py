"""
Tasks for creating correctionlib files.
"""

import law
import luigi

from functools import cached_property

from columnflow.tasks.framework.base import Requirements, ShiftTask
from columnflow.tasks.framework.mixins import (
    SelectorClassMixin, CalibratorClassesMixin,
    DatasetsProcessesMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.selection import MergeSelectionStats
from columnflow.util import maybe_import, dev_sandbox

# from columnflow.config_util import get_datasets_from_process
from hbw.tasks.base import HBWTask

ak = maybe_import("awkward")
hist = maybe_import("hist")
np = maybe_import("numpy")


logger = law.logger.get_logger(__name__)
logger_dev = law.logger.get_logger(f"{__name__}-dev")


class GetBtagNormalizationSF(
    HBWTask,
    SelectorClassMixin,
    CalibratorClassesMixin,
    DatasetsProcessesMixin,
    ShiftTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    This task computes btag re-normalization scale factors based on the selection statistics.
    It merges the selection statistics across datasets and processes, and computes the scale factors
    based on the provided rescaling mode (either "nevents" or "xs"). The scale factors are stored
    as correctionlib evaluators in a JSON file.

    The scale factors are computed in different binnings, such as:
    - `ht`, `njet`, `nhf`
    - `ht`, `njet`
    - `ht`
    - `njet`
    - `nhf`

    The output is a JSON file containing the scale factors, which can be used for re-normalization
    of btag weights in the analysis.

    Resources:
    - https://btv-wiki.docs.cern.ch/PerformanceCalibration/shapeCorrectionSFRecommendations/#effect-on-event-yields
    """
    resolution_task_cls = MergeSelectionStats

    single_config = True
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeSelectionStats=MergeSelectionStats,
    )

    store_as_dict = False

    reweighting_step = "selected_no_bjet"

    rescale_mode = luigi.ChoiceParameter(
        default="nevents",
        choices=("nevents", "xs"),
    )
    base_key = luigi.ChoiceParameter(
        default="rescaled_sum_mc_weight",
        # NOTE: "num_events" does not work because I did not store the corresponding key in the stats :/
        choices=("rescaled_sum_mc_weight", "sum_mc_weight", "num_events"),
    )

    # default sandbox, might be overwritten by selector function
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    processes = DatasetsProcessesMixin.processes.copy(
        default=("tt", "dy_m50toinf", "dy_m10to50"),
        description="Processes to consider for the scale factors",
        add_default_to_description=True,
    )

    # njet_overflow = 7
    # nhf_overflow = 4
    njet_overflow = luigi.IntParameter(
        default=0,
        description="Maximum number of jets to consider for the scale factors. If None, no overflow bin is applied.",
    )
    nhf_overflow = luigi.IntParameter(
        default=0,
        description="Maximum number of jets to consider for the scale factors. If None, no overflow bin is applied.",
    )

    def create_branch_map(self):
        # single branch without payload
        return {0: None}

    @cached_property
    def process_insts(self):
        process_insts = [self.config_inst.get_process(process) for process in self.processes]
        return process_insts

    @cached_property
    def dataset_insts(self):
        dataset_insts = [self.config_inst.get_dataset(dataset) for dataset in self.datasets]
        return dataset_insts

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["selection_stats"] = {
            dataset.name: self.reqs.MergeSelectionStats.req_different_branching(
                self,
                dataset=dataset.name,
                branch=-1,
            )
            for dataset in self.dataset_insts
        }
        return reqs

    def requires(self):
        reqs = {}
        reqs["selection_stats"] = {
            dataset.name: self.reqs.MergeSelectionStats.req_different_branching(
                self,
                dataset=dataset.name,
                branch=-1,
            )
            for dataset in self.dataset_insts
        }
        return reqs

    def store_parts(self):
        parts = super().store_parts()

        processes_repr = "__".join(self.processes)
        parts.insert_before("version", "processes", processes_repr)

        significant_params = (self.rescale_mode, self.base_key)
        parts.insert_before("version", "params", "__".join(significant_params))

        overflow_repr = ""
        if self.njet_overflow > 0:
            overflow_repr += f"njet{self.njet_overflow}"
        if self.nhf_overflow > 0:
            overflow_repr += f"_nhf{self.nhf_overflow}"
        if overflow_repr:
            parts.insert_before("version", "overflow", overflow_repr)

        return parts

    def output(self):
        return {
            "btag_renormalization_sf": self.target("btag_renormalization_sf.json"),
            "btag_renormalization_sf_plot": self.target("btag_renormalization_sf_plot.pdf", optional=True),
            "plots": self.target("plots", dir=True, optional=True),
        }

    def reduce_hist(self, hist, mode: list[str]):
        """
        Helper function that reduces the histogram to the requested axes based on the mode.
        """
        hist = hist[{"process": sum, "steps": self.reweighting_step}]

        # check validity of mode
        ax_names = [ax.name for ax in hist.axes]
        if not all(ax in ax_names for ax in mode):
            raise ValueError(f"Invalid mode {mode} for axes {ax_names}")

        # remove axes not in mode
        for ax in hist.axes:
            if ax.name not in mode:
                hist = hist[{ax.name: sum}]

        return hist

    def apply_overflow_bin(self, hist):
        # from columnflow.plotting.plot_util import use_flow_bins
        ax_names = [ax.name for ax in hist.axes]
        if self.njet_overflow > 0 and "njet" in ax_names:
            hist = hist[{"njet": slice(0, self.njet_overflow + 1)}]
        if self.nhf_overflow > 0 and "nhf" in ax_names:
            hist = hist[{"nhf": slice(0, self.nhf_overflow + 1)}]
        return hist

    def run(self):
        import correctionlib
        import correctionlib.convert
        outputs = self.output()
        inputs = self.input()

        # load the selection merged_hists
        hists_per_dataset = {
            dataset: inp["collection"][0]["hists"].load(formatter="pickle")
            for dataset, inp in inputs["selection_stats"].items()
        }

        def safe_div(num, den):
            return np.where(
                (num > 0) & (den > 0),
                num / den,
                1.0,
            )

        # rescale the histograms
        if "rescaled" in self.base_key:
            for dataset, hists in hists_per_dataset.items():
                process = self.config_inst.get_dataset(dataset).processes.get_first()
                if self.rescale_mode == "xs":
                    # scale such that the sum of weights is the cross section
                    xs = process.get_xsec(self.config_inst.campaign.ecm).nominal
                    dataset_factor = xs / hists["sum_mc_weight"][{"steps": "Initial"}].value
                elif self.rescale_mode == "nevents":
                    # scale such that mean weight is 1
                    n_events = hists["num_events"][{"steps": self.reweighting_step}].value
                    dataset_factor = n_events / hists["sum_mc_weight"][{"steps": self.reweighting_step}].value
                else:
                    raise ValueError(f"Invalid rescale mode {self.rescale_mode}")
                for key in tuple(hists.keys()):
                    if "sum" not in key:
                        continue
                    h = hists[key].copy() * dataset_factor
                    hists[f"rescaled_{key}"] = h

        # if necessary, merge the histograms across datasets
        if len(hists_per_dataset) > 1:
            merged_hists = {}
            from columnflow.tasks.selection import MergeSelectionStats
            for dataset, hists in hists_per_dataset.items():
                MergeSelectionStats.merge_counts(merged_hists, hists)
        else:
            merged_hists = hists_per_dataset[self.dataset_insts[0].name]

        # initialize the scale factor map
        sf_map = {}

        # TODO: mode "" (reduce everything to a single bin) not yet working
        for mode in (
            ("ht", "njet", "nhf"),
            ("ht", "njet"),
            ("ht",),
            ("njet",),
            ("nhf",),
            # ("",),
        ):
            mode_str = "_".join(mode)
            numerator = merged_hists[f"{self.base_key}_per_process_ht_njet_nhf"]
            numerator = self.reduce_hist(numerator, mode)
            numerator = self.apply_overflow_bin(numerator).values()

            for key in merged_hists.keys():
                if (
                    not key.startswith(f"{self.base_key}_btag_weight") or
                    not key.endswith("_per_process_ht_njet_nhf")
                ):
                    continue

                # extract the weight name
                weight_name = key.replace(f"{self.base_key}_", "").replace("_per_process_ht_njet_nhf", "")

                # create the scale factor histogram
                h = merged_hists[key]
                h = self.reduce_hist(h, mode)
                h = self.apply_overflow_bin(h)
                denominator = h.values()

                # get axes for the output histogram
                out_axes = []
                for ax in h.axes:
                    if isinstance(ax, hist.axis.Variable):
                        out_axes.append(ax)
                    elif isinstance(ax, hist.axis.Integer):
                        # convert from Integer to Variable to allow clamping
                        out_axes.append(hist.axis.Variable(ax.edges, name=ax.name, label=ax.label))
                    else:
                        raise ValueError(f"Unsupported axis type {type(ax)}")

                # calculate the scale factor and store it as a correctionlib evaluator
                sf = safe_div(numerator, denominator)
                sfhist = hist.Hist(*out_axes, data=sf)
                sfhist.name = f"{mode_str}_{weight_name}"
                sfhist.label = "out"

                if mode_str == "ht_njet_nhf":
                    self.plot_ht_njet_hft_btag_weight(sfhist)

                # import correctionlib.convert
                btag_renormalization = correctionlib.convert.from_histogram(sfhist)
                btag_renormalization.description = f"{weight_name} per {mode_str} re-normalization"

                # set overflow bins behavior (default is to raise an error when out of bounds)
                if any(isinstance(ax, hist.axis.Variable) for ax in out_axes):
                    btag_renormalization.data.flow = "clamp"

                # store the evaluator
                sf_map[sfhist.name] = btag_renormalization

        # create correction set and store it
        logger.info(f"Storing corrections with keys {sf_map.keys()}")
        cset = correctionlib.schemav2.CorrectionSet(
            schema_version=2,
            description="btag re-normalization SFs",
            corrections=list(sf_map.values()),
        )
        cset_json = cset.json(exclude_unset=True)
        if self.store_as_dict:
            import json
            cset_json = json.loads(cset_json)

        outputs["btag_renormalization_sf"].dump(
            cset_json,
            formatter="json",
        )

    skip_variations = True

    def plot_ht_njet_hft_btag_weight(self, sfhist):
        """
        Plot the btag weight SFs for the ht_njet_nhf mode.
        """
        if self.skip_variations and sfhist.name != "ht_njet_nhf_btag_weight":
            return
        logger.info(f"Plotting btag weight SFs for sfhist {sfhist.name}")
        output = self.output()["plots"]
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.use("Agg")  # Use a non-interactive backend

        nhf_edges = sfhist.axes["nhf"].edges
        njet_edges = sfhist.axes["njet"].edges
        ht_edges = sfhist.axes["ht"].edges

        # # create one grid of figures with one figure per njet and nhf bin
        fig, axs = plt.subplots(
            len(njet_edges) - 1, len(nhf_edges) - 1,
            figsize=(len(nhf_edges) * 3, len(njet_edges) * 3),
            constrained_layout=True,
            gridspec_kw={
                "left": 0.05, "right": 0.95,
                "bottom": 0.05, "top": 0.95,
                "hspace": 0.30, "wspace": 0.30,
            },
            sharex=True,
            # sharey=True,
        )

        # create one 1D plot per nhf and njet bin
        for nhf_idx, nhf_edge in enumerate(nhf_edges[:-1]):
            for njet_idx, njet_edge in enumerate(njet_edges[:-1]):
                logger.info(f"Plotting SF for NHF: {nhf_edge}, NJet: {njet_edge}")
                # fig, ax = plt.subplots(figsize=(10, 6))
                ax = axs[njet_idx, nhf_idx]
                sfbin = sfhist[{"nhf": nhf_idx, "njet": njet_idx}]
                sfbin.plot1d(
                    ax=ax,
                    # histtype="fill",
                    histtype="step",
                    yerr=False,
                )
                ax.set_title(f"NHF: {nhf_edge}, NJet: {njet_edge}")
                ax.set_xlabel("HT (GeV)")
                ax.set_ylabel("SF")
                ax.set(xlim=(ht_edges[0], ht_edges[-1]))
                # output.child(f"btag_sf_ht_njet_nhf_{nhf_idx}_{njet_idx}.pdf", type="f").dump(fig, formatter="mpl")
                # plt.close(fig)

        # # save the plot
        plt.tight_layout()
        output.child(f"{sfhist.name}_plot.pdf", type="f").dump(fig, formatter="mpl")
        if sfhist.name == "ht_njet_nhf_btag_weight":
            self.output()["btag_renormalization_sf_plot"].dump(fig, formatter="mpl")
