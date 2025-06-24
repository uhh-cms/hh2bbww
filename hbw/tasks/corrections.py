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

        return parts

    def output(self):
        return {
            "btag_renormalization_sf": self.target("btag_renormalization_sf.json"),
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
            numerator = self.reduce_hist(numerator, mode).values()

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
                denominator = h.values()

                # get axes for the output histogram
                out_axes = []
                for ax in h.axes:
                    if isinstance(ax, hist.axis.Variable):
                        out_axes.append(ax)
                    elif isinstance(ax, hist.axis.Integer):
                        out_axes.append(hist.axis.Variable(ax.edges, name=ax.name, label=ax.label))
                    else:
                        raise ValueError(f"Unsupported axis type {type(ax)}")

                # calculate the scale factor and store it as a correctionlib evaluator
                sf = safe_div(numerator, denominator)
                sfhist = hist.Hist(*out_axes, data=sf)
                sfhist.name = f"{mode_str}_{weight_name}"
                sfhist.label = "out"

                # import correctionlib.convert
                btag_renormalization = correctionlib.convert.from_histogram(sfhist)
                btag_renormalization.description = f"{weight_name} per {mode_str} re-normalization"

                # set overflow bins behavior (default is to raise an error when out of bounds)
                # NOTE: claming seems to not work for int axes. Hopefully the number of jets considered to
                # create these SFs is always large enough to not hit the overflow bin.
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
