"""
Tasks for creating correctionlib files.
"""

import law

from functools import cached_property

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    SelectorMixin, CalibratorsMixin,
)
from columnflow.tasks.selection import MergeSelectionStats
from columnflow.util import maybe_import, dev_sandbox
from columnflow.config_util import get_datasets_from_process
from hbw.tasks.base import HBWTask

ak = maybe_import("awkward")
hist = maybe_import("hist")
np = maybe_import("numpy")


logger = law.logger.get_logger(__name__)


class GetBtagNormalizationSF(
    HBWTask,
    SelectorMixin,
    CalibratorsMixin,
    # law.LocalWorkflow,
):
    reqs = Requirements(MergeSelectionStats=MergeSelectionStats)

    store_as_dict = False

    # default sandbox, might be overwritten by selector function
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    processes = law.CSVParameter(default=("tt",), description="Processes to consider for the scale factors")

    @cached_property
    def process_insts(self):
        processes = [self.config_inst.get_process(process) for process in self.processes]
        return processes

    @cached_property
    def dataset_insts(self):
        datasets = set()
        for process_inst in self.process_insts:
            datasets.update(get_datasets_from_process(self.config_inst, process_inst))
        return list(datasets)

    def requires(self):
        reqs = {}
        reqs["selection_stats"] = {
            dataset.name: self.reqs.MergeSelectionStats.req(
                self,
                dataset=dataset.name,
                tree_index=0,
                branch=-1,
                _exclude=self.reqs.MergeSelectionStats.exclude_params_forest_merge,
            )
            for dataset in self.dataset_insts
        }
        return reqs

    def store_parts(self):
        parts = super().store_parts()

        processes_repr = "__".join(self.processes)
        parts.insert_before("version", "processes", processes_repr)

        return parts

    def output(self):
        return {
            "btag_renormalization_sf": self.target("btag_renormalization_sf.json"),
        }

    def reduce_hist(self, hist, mode):
        """
        Helper function that reduces the histogram to the requested axes based on the mode.
        """
        hist = hist[{"process": sum, "steps": "selected_no_bjet"}]
        if mode == "ht_njet":
            return hist
        elif mode == "njet":
            return hist[{"ht": sum}]
        elif mode == "ht":
            return hist[{"n_jets": sum}]
        elif mode == "":
            return hist[{"ht": sum, "n_jets": sum}]
        else:
            raise ValueError(f"Invalid mode: {mode}")

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

        # rescale the histograms to the cross section
        for dataset, hists in hists_per_dataset.items():
            process = self.config_inst.get_dataset(dataset).processes.get_first()
            xs = process.get_xsec(self.config_inst.campaign.ecm).nominal
            dataset_factor = xs / hists["sum_mc_weight"][{"steps": "all"}].value
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
        for mode in ("ht_njet", "njet", "ht"):
            numerator = merged_hists["rescaled_sum_mc_weight_per_process_ht_njet"]
            numerator = self.reduce_hist(numerator, mode).values()

            for key in merged_hists.keys():
                if not key.startswith("rescaled_sum_mc_weight_btag_weight") or not key.endswith("_per_process_ht_njet"):
                    continue

                # extract the weight name
                weight_name = key.replace("rescaled_sum_mc_weight_", "").replace("_per_process_ht_njet", "")

                # create the scale factor histogram
                h = merged_hists[key]
                h = self.reduce_hist(h, mode)
                denominator = h.values()

                # calculate the scale factor and store it as a correctionlib evaluator
                sf = safe_div(numerator, denominator)

                sfhist = hist.Hist(*h.axes, data=sf)
                sfhist.name = f"{mode}_{weight_name}"
                sfhist.label = "out"

                # import correctionlib.convert
                btag_renormalization = correctionlib.convert.from_histogram(sfhist)
                btag_renormalization.description = f"{weight_name} per {mode} re-normalization"

                # set overflow bins behavior (default is to raise an error when out of bounds)
                # NOTE: claming seems to not work for int axes. Hopefully the number of jets considered to
                # create these SFs is always large enough to not hit the overflow bin.
                if any(isinstance(ax, hist.axis.Variable) for ax in h.axes):
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
