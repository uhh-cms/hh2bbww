# coding: utf-8

"""
Custom tasks for inspecting the configuration or certain task outputs.
"""

import law
import luigi

from columnflow.tasks.framework.mixins import (
    ProducersMixin, MLModelsMixin,
)
from columnflow.tasks.framework.parameters import SettingsParameter
from columnflow.tasks.reduction import ReducedEventsUser
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import get_ak_routes, update_ak_array

from hbw.tasks.base import HBWTask, ColumnsBaseTask

ak = maybe_import("awkward")


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

    debugger = luigi.BoolParameter(
        default=True,
        description="Whether to start a ipython debugger session or not; default: True",
    )

    settings = SettingsParameter(default={})

    def requires(self):
        return {}

    def output(self):
        return {"always_incomplete_dummy": self.target("dummy.txt")}

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
