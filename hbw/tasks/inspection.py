# coding: utf-8

"""
Custom tasks for inspecting the configuration or certain task outputs.
"""

import law
import luigi

from columnflow.tasks.framework.mixins import (
    ProducersMixin, MLModelsMixin,
)
from columnflow.tasks.framework.base import ConfigTask
from columnflow.tasks.framework.parameters import SettingsParameter
from columnflow.tasks.reduction import ReducedEventsUser
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import get_ak_routes, update_ak_array

from hbw.tasks.base import HBWTask, ColumnsBaseTask
from hbw.util import round_sig

ak = maybe_import("awkward")


class DumpAnalysisSummary(
    HBWTask,
    ConfigTask,
):

    keys_of_interest = law.CSVParameter(
        default=tuple(),
        description="Keys of interest to be printed in the summary",
    )

    def requires(self):
        return {}

    def output(self):
        output = {
            "dataset_summary": self.target("dataset_summary.txt"),
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

        if self.debugger:
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
