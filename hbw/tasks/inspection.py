# coding: utf-8

"""
Custom tasks for inspecting the configuration or certain task outputs.
"""

# from functools import cached_property

import law
import luigi

from columnflow.tasks.framework.mixins import (
    ProducersMixin, MLModelsMixin,
)
from columnflow.tasks.framework.base import ConfigTask, Requirements
from columnflow.tasks.framework.mixins import DatasetsProcessesMixin, SelectorMixin, CalibratorsMixin
from columnflow.tasks.framework.parameters import SettingsParameter
from columnflow.tasks.reduction import ReducedEventsUser
from columnflow.tasks.selection import MergeSelectionStats
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import get_ak_routes, update_ak_array

from hbw.tasks.base import HBWTask, ColumnsBaseTask
from hbw.util import round_sig

ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


class SelectionSummary(
    HBWTask,
    DatasetsProcessesMixin,
    SelectorMixin,
    CalibratorsMixin,
):
    reqs = Requirements(MergeSelectionStats=MergeSelectionStats)

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    keys_of_interest = law.CSVParameter(
        # default=("num_events", "num_events_per_process", "sum_mc_weight", "sum_mc_weigth_per_process"),
        default=tuple(),
    )

    # @cached_property
    # def datasets(self):
    #     return [dataset.name for dataset in self.config_inst.datasets]

    def requires(self):
        print("SelectionSummary requires")
        reqs = {}
        for dataset in self.datasets:
            reqs[dataset] = self.reqs.MergeSelectionStats.req(
                self,
                dataset=dataset,
                tree_index=0,
                branch=-1,
                _exclude=self.reqs.MergeSelectionStats.exclude_params_forest_merge,
            )
        return reqs

    @property
    def keys_repr(self):
        return "_".join(sorted(self.keys_of_interest))

    def output(self):
        output = {
            "selection_summary": self.target("selection_summary.txt"),
        }
        return output

    def write_selection_summary(self, outp):
        import csv
        outp.touch()
        lumi = self.config_inst.x.luminosity
        inputs = self.input()

        empty_datasets = []

        keys_of_interest = self.keys_of_interest or ["selection_eff", "expected_yield", "num_events_selected"]
        header_map = {
            "xsec": "CrossSection [pb]",
            "empty": "Empty?",
            "selection_eff": "Efficiency",
            "expected_yield": "Yields",
            "num_events_selected": "NSelected",
        }

        with open(outp.path, "w") as f:
            writer = csv.writer(f)

            writer.writerow(["Dataset"] + [header_map.get(key, key) for key in keys_of_interest])
            for dataset in self.datasets:
                stats = inputs[dataset]["collection"][0]["stats"].load(formatter="json")
                # hists = inputs[dataset]["collection"][0]["hists"].load(formatter="pickle")

                xsec = self.config_inst.get_dataset(dataset).processes.get_first().xsecs.get(
                    self.config_inst.campaign.ecm, None,
                )

                def safe_div(num, den):
                    return num / den if den != 0 else 0

                missing_keys = {"sum_mc_weight", "sum_mc_weight_selected"} - set(stats.keys())
                if missing_keys:
                    logger.warning(f"Missing keys in stats in dataset {dataset}: {missing_keys}")
                    continue

                selection_eff = safe_div(stats["sum_mc_weight_selected"], stats["sum_mc_weight"])
                if xsec is not None:
                    expected_yield = xsec * selection_eff * lumi

                if stats["num_events_selected"] == 0:
                    empty_datasets.append(dataset)

                selection_summary = {
                    "xsec": xsec.nominal,
                    "empty": True if stats["num_events_selected"] == 0 else False,
                    "selection_eff": selection_eff,
                    "expected_yield": expected_yield.nominal,
                }
                for key in keys_of_interest:
                    if key in selection_summary.keys():
                        continue
                    if key in stats:
                        selection_summary[key] = stats[key]
                    else:  # default to empty string
                        selection_summary[key] = ""

                row = [dataset] + [selection_summary[key] for key in keys_of_interest]
                print(row)
                writer.writerow([dataset] + [selection_summary[key] for key in keys_of_interest])

        self.publish_message(f"Empty datasets: {empty_datasets}")

    def run(self):
        output = self.output()
        self.write_selection_summary(output["selection_summary"])


class DumpAnalysisSummary(
    HBWTask,
    ConfigTask,
):

    keys_of_interest = law.CSVParameter(
        default=tuple(),
        description="Keys of interest to be printed in the summary",
    )

    @property
    def keys_repr(self):
        return "_".join(sorted(self.keys_of_interest))

    def requires(self):
        return {}

    def output(self):
        output = {
            "dataset_summary": self.target(f"dataset_summary_{self.keys_repr}.txt"),
        }
        return output

    def write_dataset_summary(self, outp):
        import csv
        outp.touch()
        with open(outp.path, "w") as f:
            writer = csv.writer(f)
            keys_of_interest = self.keys_of_interest or ["das_keys", "process", "xsec"]
            header_map = {
                "name": "Dataset name",
                "n_events": "Number of events",
                "n_files": "Number of files",
                "das_keys": "DAS keys",
                "rucio": "Rucio DAS keys",
                "process": "Process name",
                "xsec": "Cross section [pb]",
                "xsec_unc": "Cross section +- unc [pb]",
                "xsec_full": "Cross section +- unc [pb]",
            }
            writer.writerow([header_map[key] for key in keys_of_interest])
            for dataset in self.config_inst.datasets:
                xsec = dataset.processes.get_first().xsecs.get(13.6, None)
                try:
                    dataset_summary = {
                        "name": dataset.name,
                        "n_events": dataset.n_events,
                        "n_files": dataset.n_files,
                        "das_keys": dataset.get_info("nominal").keys[0],
                        "rucio": "cms:" + dataset.get_info("nominal").keys[0],
                        "process": dataset.processes.get_first().name,
                        "xsec": round_sig(xsec.nominal, 4) if xsec else "0",
                        "xsec_unc": xsec.str("pdg", combine_uncs="all") if xsec else "0",
                        # "xsec_full": xsec.str("pdg") if xsec else "",
                    }
                except Exception as e:
                    from hbw.util import debugger
                    debugger("Failed to get dataset summary", e)
                writer.writerow([dataset_summary[key] for key in keys_of_interest])

    def run(self):
        output = self.output()
        self.write_dataset_summary(output["dataset_summary"])


class CheckConfig(
    HBWTask,
    MLModelsMixin,
    ProducersMixin,
    ReducedEventsUser,
    law.LocalWorkflow,
):
    """
    Task that inherits from relevant mixins to build the config inst based on CSP+ML init functions.
    It only prints some informations from the config inst.
    Does not require anything, does not output anything.
    """
    # columnar sandbox is always nice to have :)
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    version = None

    skip_debugger = luigi.BoolParameter(
        default=False,
        description="Whether to start a ipython debugger session or not; default: False",
    )

    settings = SettingsParameter(default={})

    def requires(self):
        return {}

    def output(self):
        output = {
            "always_incomplete_dummy": self.target("dummy.txt"),
        }
        return output

    def run(self):
        config = self.config_inst
        dataset = self.dataset_inst
        variables = config.variables
        all_cats = [cat for cat, _, _ in config.walk_categories()]
        leaf_cats = config.get_leaf_categories()
        processes = [proc for proc, _, _ in config.walk_processes()]  # noqa

        self.publish_message(
            f"\nLooking at config '{config.name}' with dataset '{dataset.name}' and "
            f"shift '{self.shift}' after running inits of calibrators "
            f"{self.calibrators}, selector '{self.selector}', producer "
            f"{self.producers}, and ml models {self.ml_models} \n",
        )
        self.publish_message(
            f"{'=' * 10} Categories ({len(all_cats)}):\n{[cat.name for cat in all_cats]} \n\n"
            f"{'=' * 10} Leaf Categories ({len(leaf_cats)}):\n{[cat.name for cat in leaf_cats]} \n\n"
            f"{'=' * 10} Variables ({len(variables)}):\n{variables.names()} \n\n",
        )

        if not self.skip_debugger:
            self.publish_message("starting debugger ....")
            from hbw.util import debugger
            debugger()


class CheckColumns(
    ColumnsBaseTask,
    law.LocalWorkflow,
):
    """
    Task to inspect columns after Reduction, Production and MLEvaluation.
    """

    debugger = luigi.BoolParameter(
        default=True,
        description="Whether to start a ipython debugger session or not; default: True",
    )

    def output(self):
        return {"always_incomplete_dummy": self.target("dummy.txt")}

    def run(self):
        import awkward as ak
        inputs = self.input()

        config = self.config_inst
        dataset = self.dataset_inst

        self.publish_message(
            f"\nLooking at columns from reduction, producers {self.producers}, and ml models "
            f"{self.ml_models} using config '{config.name}' with dataset '{dataset.name}' and "
            f"shift '{self.shift}', calibrators {self.calibrators}, and selector '{self.selector}'\n",
        )

        files = {"events": [inputs["events"]["collection"][0]["events"]][0]}
        for i, producer in enumerate(self.producers):
            files[producer] = inputs["producers"][i]["columns"]
        for i, ml_model in enumerate(self.ml_models):
            files[ml_model] = inputs["ml"][i]["mlcolumns"]

        # open each file and check, which columns are present
        # NOTE: we could use the Chunked Reader here aswell, but since we do not do any data processing,
        #       it should be fine to shortly load the complete files into memory
        for key, fname in files.items():
            columns = ak.from_parquet(fname.path)
            fields = [route.string_column for route in get_ak_routes(columns)]
            self.publish_message(f"{'=' * 10} {key} fields:\n{fields} \n")

        if self.debugger:
            # when starting a debugger session, combine all columns into one ak.Array
            events = ak.from_parquet(files["events"])
            events = update_ak_array(events, *[ak.from_parquet(fname) for fname in files.values()])
            self.publish_message("starting debugger ....")
            from hbw.util import debugger
            debugger()
