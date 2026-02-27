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


def make_table_func(
    newcommands: str = "",
    tablecommandname: str = r"\maketable",
    table_content: str = r"Table content & value & value \\",
    caption: str = "Caption.",
    label: str = "tab:label",
    cols: str = "lcc",
):
    table = rf"""

{newcommands}
\newcommand{{{tablecommandname}}}[0]{{%
\begin{{table}}[!htbp]
  \centering
  \caption{{{caption}}}%
  \label{{{label}}}
  \renewcommand{{\arraystretch}}{{1.3}}
  \begin{{small}}
    \begin{{tabular}}{{{cols}}}{table_content}
    \end{{tabular}}
  \end{{small}}
  \renewcommand{{\arraystretch}}{{1.0}}
\end{{table}}
}}%
"""
    return table


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
            "xs_table": self.target("xs_table.tex"),
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

    def build_xs_table_uncs(self):
        """
        Builds a LaTeX table summarizing the cross sections and uncertainties of various processes.
        """
        processes_dict = {
            "tt": r"\ttbar",
            "st_tchannel": r"\sttchannel",
            "st_schannel": r"\stschannel",
            "st_twchannel": r"\tW",
            "dy_m50toinf": r"\DYmfifty",
            "w_lnu": r"\Wjets",
            "ww": r"\WW",
            "wz": r"\WZ",
            "zz": r"\ZZ",
            "vvv": r"\triboson",
            "ttw": r"\ttW",
            "ttz": r"\ttZ",
            "ttvv": r"\ttVV",
            "tttt": r"\tttt",
            "h_ggf": r"\Hggf",
            "h_vbf": r"\Hvbf",
            "zh": r"\ZH",
            "zh_gg": r"\ggZH",
            "wh": r"\WH",
            "tth": r"\ttH",
            "ttvh": r"\ttVH",
            "thq": r"\tHq",
            "thw": r"\tHW",
        }
        unc_keys = ("scale", "pdf", "mtop", "alpha_s", "th")
        for process, latex_name in processes_dict.items():
            proc_inst = self.config_inst.get_process(process)
            if not proc_inst:
                raise Exception(f"Process '{process}' not found in config '{self.config_inst.name}'")
            ecm = self.config_inst.campaign.ecm
            if process == "ttz":
                ecm = 13  # ttZ xs unc missing in 13.6
            xs = proc_inst.xsecs.get(ecm, {})
            processes_dict[process] = {
                "latex_name": latex_name,
                "xsec": xs,
            }
            for unc_key in unc_keys:
                if unc_key in xs:
                    unc_down, unc_up = xs.get(names=unc_key, direction=("down", "up"), factor=True)
                    # rounded_unc_down = round_sig(100 - unc_down * 100, 2)
                    # rounded_unc_up = round_sig(unc_up * 100 - 100, 2)
                    rounded_unc_down = round(100 - unc_down * 100, 1)
                    rounded_unc_up = round(unc_up * 100 - 100, 1)
                    if rounded_unc_down == rounded_unc_up:
                        unc_repr = f"$\\pm{rounded_unc_up}\\%$"
                    else:
                        unc_repr = f"$+{rounded_unc_up}\\%$ / $-{rounded_unc_down}\\%$"
                    processes_dict[process][f"{unc_key}"] = unc_repr

        for unc_key in unc_keys:
            print(f"\n### {unc_key} uncertainties ###")
            for process, info in processes_dict.items():
                if unc_key in info:
                    latex_name = info["latex_name"]
                    unc_repr = info[unc_key]
                    print(rf"{latex_name} & {unc_repr}~\cite{{TODO}} \\")

    def build_signal_xs_table(self):
        root_processes = {
            "hh_ggf": r"\HHggf",
            "hh_vbf": r"\HHvbf",
        }
        sub_processes = {
            "hh_ggf": {
                "hh_ggf_kl1_kt1": ("1", "", ""),
                "hh_ggf_kl0_kt1": ("0", "", ""),
                "hh_ggf_kl2p45_kt1": ("2.45", "", ""),
                "hh_ggf_kl5_kt1": ("5", "", ""),
            },
            "hh_vbf": {
                "hh_vbf_kv1_k2v1_kl1": ("1", "1", "1"),
                "hh_vbf_kv1_k2v0_kl1": ("1", "0", "1"),
                "hh_vbf_kv1p74_k2v1p37_kl14p4": ("1.74", "1.37", "14.4"),
                "hh_vbf_kvm0p012_k2v0p03_kl10p2": ("0.012", "0.03", "10.2"),
                "hh_vbf_kvm0p758_k2v1p44_klm19p3": ("0.758", "1.44", "-19.3"),
                "hh_vbf_kvm0p962_k2v0p959_klm1p43": ("0.962", "0.959", "-1.43"),
                "hh_vbf_kvm1p21_k2v1p94_klm0p94": ("-1.21", "1.94", "-0.94"),
                "hh_vbf_kvm1p6_k2v2p72_klm1p36": ("-1.6", "2.72", "-1.36"),
                "hh_vbf_kvm1p83_k2v3p57_klm3p39": ("-1.83", "3.57", "-3.39"),
                "hh_vbf_kv2p12_k2v3p87_klm5p96": ("2.12", "3.87", "-5.96"),
            },
        }
        tablecommandname = r"\makesignalxstable"
        caption = r"Cross sections of the \HHggf and \HHvbf processes for different coupling parameters."
        label = "tab:signal_xs_table"
        cols = "llllr"
        lines = [
            r"\hline",
            r"Process & $\kappa_\lambda$ & $\kappa_V$ & $\kappa_{2V}$ & Cross section [pb] \\",
            r"\hline",
        ]

        for process, latex_name in root_processes.items():
            lines.append(r"\hline")
            proc_inst = self.config_inst.get_process(process)
            if not proc_inst:
                raise Exception(f"Process '{process}' not found in config '{self.config_inst.name}'")
            xsec = proc_inst.xsecs.get(self.config_inst.campaign.ecm, None)
            if xsec and xsec.nominal == 0.1:
                xsec = None

            # Add main process line
            # main_xsec = f"{round_sig(xsec.nominal, 4)}" if xsec else ""
            # lines.append(f"  {latex_name} &  &  &  & {main_xsec} \\\\")

            # Add sub-process lines if they exist
            if process in sub_processes:
                # lines.append(r"\hline")
                for sub_proc, sub_latex_name in sub_processes[process].items():
                    kl, kv, k2v = [r"\textemdash" if val == "" else f"${val}$" for val in sub_latex_name]
                    sub_proc_inst = self.config_inst.get_process(sub_proc)
                    if not sub_proc_inst:
                        raise Exception(f"Sub-process '{sub_proc}' not found in config '{self.config_inst.name}'")
                    sub_xsec = sub_proc_inst.xsecs.get(self.config_inst.campaign.ecm, None)
                    if sub_xsec:
                        sub_xsec_str = f"{round_sig(sub_xsec.nominal, 4)}"
                        lines.append(f" {latex_name} & {kl} & {kv} & {k2v} & {sub_xsec_str} \\\\")
                    latex_name = ""

        table_content = "\n".join(lines)
        table = make_table_func(
            tablecommandname=tablecommandname,
            table_content=table_content,
            caption=caption,
            label=label,
            cols=cols,
        )
        print(table)
        output_file = self.output()["latex_table"]
        output_file.dump(table, formatter="text")

    def build_bkg_xs_table(self):
        root_processes_bkg = {
            # "data": "skip",
            # "hh_ggf": r"\HHggF",
            # "hh_vbf": r"\HHVBF",
            "tt": r"\ttbar",
            "st": r"\singlet",
            "dy": r"\DY",
            "w_lnu": r"\Wjets",
            "vv": r"\diboson",
            "ttv": r"\ttV",
            "h": r"\PH",
            "other": r"\other",
        }
        sub_processes_bkg = {
            "tt": {
                "tt_dl": r"\ttdl",
                "tt_sl": r"\ttsl",
                "tt_fh": r"\ttfh",
            },
            "st": {
                "st_tchannel": r"\sttchannel",
                "st_schannel": r"\stschannel",
                "st_twchannel": r"\tW",
            },
            "dy": {
                "dy_m10to50": r"\DYmten",
                "dy_m50toinf": r"\DYmfifty",
            },
            "vv": {
                "ww": r"\WW",
                "wz": r"\WZ",
                "zz": r"\ZZ",
            },
            "ttv": {
                "ttw": r"\ttW",
                "ttz": r"\ttZ",
            },
            "h": {
                "h_ggf": r"\Hggf",
                "h_vbf": r"\Hvbf",
                "zh": r"\ZH",
                "zh_gg": r"\ggZH",
                "wh": r"\WH",
                "tth": r"\ttH",
                "ttvh": r"\ttVH",
                "thq": r"\tHq",
                "thw": r"\tHW",
            },
            "other": {
                "tttt": r"\tttt",
                "ttvv": r"\ttVV",
                "vvv": r"\triboson",
            },
        }
        self.build_xs_table_subprocs(root_processes_bkg, sub_processes_bkg)

    def build_xs_table_subprocs(self, root_processes, sub_processes):
        tablecommandname = r"\makexstable"
        caption = "Cross sections of the main processes and their sub-processes."
        label = "tab:xs_table"
        cols = "llr"
        lines = [
            r"\hline",
            r"Process & Sub-process & Cross section [pb] \\",
            r"\hline",
        ]

        for process, latex_name in root_processes.items():
            lines.append(r"\hline")
            proc_inst = self.config_inst.get_process(process)
            if not proc_inst:
                raise Exception(f"Process '{process}' not found in config '{self.config_inst.name}'")
            xsec = proc_inst.xsecs.get(self.config_inst.campaign.ecm, None)
            if xsec and xsec.nominal == 0.1:
                xsec = None

            # Add main process line
            main_xsec = f"{round_sig(xsec.nominal, 4)}" if xsec else ""
            lines.append(f"  {latex_name} &  & {main_xsec} \\\\")

            # Add sub-process lines if they exist
            if process in sub_processes:
                # lines.append(r"\hline")
                for sub_proc, sub_latex_name in sub_processes[process].items():
                    sub_proc_inst = self.config_inst.get_process(sub_proc)
                    if not sub_proc_inst:
                        raise Exception(f"Sub-process '{sub_proc}' not found in config '{self.config_inst.name}'")
                    sub_xsec = sub_proc_inst.xsecs.get(self.config_inst.campaign.ecm, None)
                    if sub_xsec:
                        sub_xsec_str = f"{round_sig(sub_xsec.nominal, 4)}"
                        lines.append(f"    &  {sub_latex_name} & {sub_xsec_str} \\\\")

        table_content = "\n".join(lines)
        table = make_table_func(
            tablecommandname=tablecommandname,
            table_content=table_content,
            caption=caption,
            label=label,
            cols=cols,
        )
        print(table)
        output_file = self.output()["latex_table"]
        output_file.dump(table, formatter="text")

    def build_table(self):
        root_processes = {
            # "data": "skip",
            "hh_ggf": r"\HHggF",
            "hh_vbf": r"\HHVBF",
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

        # return "\n".join(lines)
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
        self.build_xs_table_uncs()
        self.build_signal_xs_table()
        self.build_bkg_xs_table()
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
