# coding: utf-8

"""
Custom tasks for inspecting the configuration or certain task outputs.
"""

from collections import defaultdict

from functools import cached_property

import law
import luigi

from scinum import Number


from columnflow.tasks.framework.base import ConfigTask, Requirements
from columnflow.tasks.framework.mixins import (
    ProducersMixin,
    MLModelsMixin,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ProducerClassesMixin,
    HistProducerClassMixin,
    CategoriesMixin,
    DatasetsProcessesMixin,
    HistHookMixin,
)
from columnflow.tasks.histograms import MergeHistograms
from columnflow.tasks.framework.plotting import (
    PlotBase,
    ProcessPlotSettingMixin,
    VariablePlotSettingMixin,
)
from columnflow.tasks.framework.parameters import SettingsParameter
from columnflow.tasks.reduction import ReducedEventsUser
from columnflow.tasks.selection import MergeSelectionStats
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import get_ak_routes, update_ak_array
from columnflow.tasks.framework.remote import RemoteWorkflow

from hbw.tasks.base import HBWTask, ColumnsBaseTask
from hbw.util import round_sig

ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


def create_table_from_csv(csv_file_path, transpose=False, with_header=True):
    import csv
    from tabulate import tabulate

    # Read the CSV file
    with open(csv_file_path, mode="r", newline="") as file:
        reader = csv.reader(file)
        data = list(reader)

    # Transpose the data if requested
    if transpose:
        data = list(zip(*data))

    # Optionally, if you want to use the first row as headers
    headers = None
    if with_header:
        headers = data[0]  # First row as headers
        data = data[1:]  # Rest as table data

    # Generate the table using tabulate
    table = tabulate(data, headers=headers, tablefmt="grid")

    # Print the table
    print(table)
    return table


class SelectionSummary(
    HBWTask,
    DatasetsProcessesMixin,
    CalibratorClassesMixin,
    SelectorClassMixin,
    # SelectorMixin,
    # CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    resolution_task_cls = MergeSelectionStats
    single_config = True

    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeSelectionStats=MergeSelectionStats,
    )

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    keys_of_interest = law.CSVParameter(
        # default=("num_events", "num_events_per_process", "sum_mc_weight", "sum_mc_weigth_per_process"),
        default=tuple(),
    )

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts

    def create_branch_map(self):
        # single branch without payload
        return {0: None}

    def requires(self):
        reqs = {}
        for dataset in self.datasets:
            reqs[dataset] = self.reqs.MergeSelectionStats.req(
                self,
                dataset=dataset,
                branch=-1,
                workflow="local",
            )
        return reqs

    def workflow_requires(self):
        reqs = super().workflow_requires()
        for dataset in self.datasets:
            reqs[dataset] = self.reqs.MergeSelectionStats.req(
                self,
                dataset=dataset,
                branch=-1,
            )
        return reqs

    @property
    def keys_repr(self):
        return "_".join(sorted(self.keys_of_interest))

    @cached_property
    def stats(self):
        inp = self.input()
        return {
            dataset: inp[dataset]["collection"][0]["stats"].load(formatter="json")
            for dataset in self.datasets
        }

    def output(self):
        output = {
            "selection_summary_csv": self.target("selection_summary.csv"),
            "selection_summary_table": self.target("selection_summary.txt"),
            "selection_steps_summary_csv": self.target("selection_steps_summary.csv"),
            "selection_steps_summary_table": self.target("selection_steps_summary.txt"),
        }
        return output

    def write_selection_summary(self, outp):
        import csv
        outp.touch()
        lumi = self.config_inst.x.luminosity

        empty_datasets = []

        keys_of_interest = self.keys_of_interest or ["selection_eff", "expected_yield", "num_events_selected", "xsec"]
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
                dataset_inst = self.config_inst.get_dataset(dataset)
                stats = self.stats[dataset]
                # hists = inputs[dataset]["collection"][0]["hists"].load(formatter="pickle")

                xsec = dataset_inst.processes.get_first().xsecs.get(
                    self.config_inst.campaign.ecm, None,
                )

                def safe_div(num, den):
                    return num / den if den != 0 else 0

                sumw_key = "sum_mc_weight" if dataset_inst.is_mc else "num_events"

                missing_keys = {f"{sumw_key}", f"{sumw_key}_selected"} - set(stats.keys())
                if missing_keys:
                    logger.warning(f"Missing keys in stats in dataset {dataset}: {missing_keys}")
                    continue

                selection_eff = safe_div(stats[f"{sumw_key}_selected"], stats[f"{sumw_key}"])
                if dataset_inst.is_data:
                    expected_yield = Number(stats["num_events_selected"])
                elif xsec is not None:
                    expected_yield = xsec * selection_eff * lumi

                if stats["num_events_selected"] == 0:
                    empty_datasets.append(dataset)

                selection_summary = {
                    "xsec": xsec.nominal if xsec else -1,
                    "empty": True if stats["num_events_selected"] == 0 else False,
                    "selection_eff": round_sig(selection_eff, 4),
                    "expected_yield": round_sig(expected_yield.nominal, 4),
                }
                for key in keys_of_interest:
                    if key in selection_summary.keys():
                        continue
                    if key in stats:
                        selection_summary[key] = round_sig(stats[key], 4)
                    else:  # default to empty string
                        selection_summary[key] = ""

                row = [dataset] + [selection_summary[key] for key in keys_of_interest]
                writer.writerow(row)

        self.publish_message(f"Empty datasets: {empty_datasets}")

    def write_selection_steps_summary(self, outp):
        import csv
        outp.touch()

        with open(outp.path, "w") as f:
            writer = csv.writer(f)

            steps = [
                k.replace("num_events_step_", "") for k in self.stats[self.datasets[0]].keys()
                if "num_events_step_" in k
            ]

            writer.writerow(["Datasets"] + steps)

            for dataset in self.datasets:
                dataset_inst = self.config_inst.get_dataset(dataset)
                stats = self.stats[dataset]

                sumw_key = "num_events" if dataset_inst.is_data else "sum_mc_weight"

                row = [dataset] + [stats.get(f"{sumw_key}_step_{step}", 0) / stats.get(sumw_key, 1.) for step in steps]
                writer.writerow(row)

    def run(self):
        output = self.output()

        # write overall summary
        self.write_selection_summary(output["selection_summary_csv"])
        table = create_table_from_csv(output["selection_summary_csv"].path)
        output["selection_summary_table"].dump(table, formatter="text")

        # write step-by-step summary
        self.write_selection_steps_summary(output["selection_steps_summary_csv"])
        table = create_table_from_csv(output["selection_steps_summary_csv"].path, transpose=True)
        output["selection_steps_summary_table"].dump(table, formatter="text")


class DumpAnalysisSummary(
    HBWTask,
    ConfigTask,
):
    single_config = True
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
            "latex_table": self.target("latex_table.tex"),
            "dataset_summary": self.target(f"dataset_summary_{self.keys_repr}.txt"),
        }
        return output

    def format_das_key(self, das_key):
        das_key_split = das_key.split("/")
        if len(das_key_split) != 4:
            raise Exception(f"Unexpected DAS key format: {das_key}")

        # generalize campaign name to [campaign]
        das_key_split[2] = "[campaign]" + "-" + das_key_split[2].split("-")[-1]
        das_keys_formatted = "/".join(das_key_split)

        # escape underscores for LaTeX
        das_keys_formatted = das_keys_formatted.replace("_", "\\_")

        # wrap in \texttt{}
        das_keys_formatted = "\\texttt{" + das_keys_formatted + "}"
        return das_keys_formatted

    def build_table(self):
        root_processes = {
            # "data": "skip",
            "hh_ggf": "\HHggF",
            "hh_vbf": "\HHVBF",
            "tt": r"\ttbar",
            "st": r"\singlet",
            "dy": r"\DY",
            "w_lnu": r"\Wjets",
            "vv": r"\diboson",
            "vvv": r"\triboson",
            "ttv": r"\ttV",
            "ttvv": r"\ttVV",
            "tttt": r"\tttt",
            "h_ggf": r"\ggF",
            "h_vbf": r"\VBF",
            "zh": r"\ZH",
            "zh_gg": r"\ggZH",
            "wh": r"\WH",
            "tth": r"\ttH",
            "ttvh": r"\ttVH",
            "thq": r"\tHq",
            "thw": r"\tHW",
        }
        table_dict = defaultdict(list)
        for dataset in self.config_inst.datasets:
            if dataset.is_data or dataset.has_tag("is_hh"):
                continue
            process = dataset.processes.get_first()
            xsec = process.xsecs.get(13.6, None)
            try:
                das_keys = dataset.get_info("nominal").keys[0]
                dataset_summary = {
                    "name": dataset.name,
                    "das_keys": dataset.get_info("nominal").keys[0],
                    "xsec": round_sig(xsec.nominal, 4) if xsec else "0",
                }
                dataset_summary["das_key_formatted"] = self.format_das_key(das_keys)
            except Exception as e:
                from hbw.util import debugger
                debugger("Failed to get dataset summary", e)
            parent = parent_key = None
            while not parent:
                for proc_name, key in root_processes.items():
                    if process.has_parent_process(proc_name) or process.name == proc_name:
                        parent = proc_name
                        parent_key = key
                        table_dict[parent_key].append(dataset_summary)
                        break
                if not parent:
                    raise Exception(f"Could not find parent process for {process.name}")

        # Generate the LaTeX table
        latex_table = self.generate_latex_table(table_dict)

        # Write to file or return
        output_file = self.output()["latex_table"]
        output_file.dump(latex_table, formatter="text")

        return table_dict

    def generate_latex_table(self, table_dict):
        lines = []
        lines.append(r"\begin{tabular}{llr}")
        lines.append(r"  Process & Sample & XS $\times$ BR [pb] \\")
        lines.append(r"  \hline")
        lines.append("")

        for i, (process_key, datasets) in enumerate(table_dict.items()):
            for j, dataset in enumerate(datasets):
                if j == 0:
                    # First row: include process name
                    process_name = process_key
                else:
                    # Subsequent rows: empty process column
                    process_name = ""

                lines.append(f"  {process_name} & {dataset['das_key_formatted']} & {dataset['xsec']} \\\\")

            # Add spacing between groups
            if i < len(table_dict) - 1:
                lines.append(r"[\cmsTabSkip]")
            lines.append("")

        lines.append(r"\end{tabular}")
        return "\n".join(lines)

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
        self.build_table()
        self.write_dataset_summary(output["dataset_summary"])


class DummyWorkflow(HBWTask, law.LocalWorkflow):
    # columnar sandbox is always nice to have :)
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    version = None

    skip_debugger = luigi.BoolParameter(
        default=False,
        description="Whether to start a ipython debugger session or not; default: False",
    )
    # reqs = Requirements(RemoteWorkflow.reqs)

    def create_branch_map(self):
        return {0: None}

    def workflow_requires(self):
        return {}

    def requires(self):
        return {}

    def output(self):
        output = {
            "always_incomplete_dummy": self.target("dummy.txt"),
        }
        return output


class CheckConfig(
    ReducedEventsUser,
    ProducersMixin,
    MLModelsMixin,
    DummyWorkflow,
):
    """
    Task that inherits from relevant mixins to build the config inst based on CSP+ML init functions.
    It only prints some informations from the config inst.
    Does not require anything, does not output anything.
    """

    settings = SettingsParameter(default={})

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


class CheckMixins(
    CalibratorClassesMixin,
    SelectorClassMixin,
    ProducerClassesMixin,
    MLModelsMixin,
    HistProducerClassMixin,
    CategoriesMixin,
    ProcessPlotSettingMixin,
    VariablePlotSettingMixin,
    HistHookMixin,
    DummyWorkflow,
):
    resolution_task_cls = MergeHistograms
    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_1d.plot_variable_per_process",
        add_default_to_description=True,
    )

    def run(self):
        if not self.skip_debugger:
            self.publish_message("starting debugger ....")
            from hbw.util import debugger
            debugger()


class DatasetSummary(
    HBWTask,
    ConfigTask,
):
    single_config = True

    def output(self):
        output = {
            "dataset_summary": self.target("dataset_summary.yaml"),
        }
        return output

    def run(self):
        multi_config_dataset_summary = {}
        for config in self.config_insts:
            dataset_summary = defaultdict(dict)
            cpn_name = config.campaign.name
            for dataset in config.datasets:
                dataset_campaign = dataset.x("campaign", cpn_name)
                dataset_summary[dataset_campaign][dataset.name] = {
                    "n_events": dataset.n_events,
                    "n_files": dataset.n_files,
                }
            multi_config_dataset_summary[config.name] = dict(dataset_summary)

        self.output()["dataset_summary"].dump(multi_config_dataset_summary, formatter="yaml")


class CheckColumns(
    ColumnsBaseTask,
    law.LocalWorkflow,
):
    """
    Task to inspect columns after Reduction, Production and MLEvaluation.
    """
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

        if not self.skip_debugger:
            # when starting a debugger session, combine all columns into one ak.Array
            events = ak.from_parquet(files["events"])
            events = update_ak_array(events, *[ak.from_parquet(fname) for fname in files.values()])
            self.publish_message("starting debugger ....")
            from hbw.util import debugger
            debugger()
