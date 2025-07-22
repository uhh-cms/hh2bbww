# coding: utf-8

"""
Tasks related to the creation and modification of datacards for inference purposes.
"""

from __future__ import annotations

from typing import Callable
from collections import defaultdict

import luigi
import law
import order as od

from columnflow.tasks.framework.base import Requirements, AnalysisTask
from columnflow.tasks.framework.parameters import SettingsParameter
from columnflow.tasks.framework.mixins import (
    CalibratorClassesMixin, SelectorClassMixin, ReducerClassMixin, ProducerClassesMixin, MLModelsMixin,
    InferenceModelMixin, HistHookMixin, HistProducerClassMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.histograms import MergeHistograms
from columnflow.tasks.cms.inference import CreateDatacards
from columnflow.util import dev_sandbox, maybe_import
from hbw.tasks.base import HBWTask
from hbw.hist_util import apply_rebinning_edges

array = maybe_import("array")
uproot = maybe_import("uproot")
np = maybe_import("numpy")
hist = maybe_import("hist")


logger = law.logger.get_logger(__name__)


def get_hist_name(cat_name: str, proc_name: str, syst_name: str | None = None) -> str:
    hist_name = f"{cat_name}/{proc_name}"
    if syst_name:
        hist_name += f"__{syst_name}"
    return hist_name


def get_cat_proc_syst_name_from_key(key: str) -> tuple[str, str, str | None]:
    """
    Extracts the category, process, and systematics name from a ROOT key.
    The key is expected to be in the format "cat_name/proc_name__syst_name".
    If no systematics name is present, it returns None for the syst_name.
    """
    key = key.split(";")[0]  # remove ";1" appendix
    parts = key.split("/")
    cat_name = parts[0]
    proc_syst = parts[1].split("__")
    proc_name = proc_syst[0]
    syst_name = proc_syst[1] if len(proc_syst) > 1 else "nominal"
    return cat_name, proc_name, syst_name


def get_cat_proc_syst_names(root_file):
    cat_names = set()
    proc_names = set()
    syst_names = set()
    for key in root_file.keys():
        # remove ";1" appendix
        key = key.split(";")[0]

        key = key.split("/")
        # first part of the key is the category
        cat_names.add(key[0])

        # second part is the hist name: <proc_name>__<syst_name>
        if len(key) == 2:
            hist_name = key[1].split("__")

            proc_name = hist_name[0]
            proc_names.add(proc_name)

            if len(hist_name) == 2:
                syst_name = hist_name[1]
                syst_names.add(syst_name)

    return cat_names, proc_names, syst_names


def get_rebin_values(
        rebin_hist,
        signal_hist,
        background_hist,
        N_bins_final: int = 10,
        min_bkg_events: int = 10,
        blinding_threshold: float | None = None,
):
    """
    Function that determines how to rebin a histogram to *N_bins_final* bins such that
    the resulting histogram is flat
    """
    msg = f"Rebinning histogram {rebin_hist.name} to {N_bins_final} bins."
    if min_bkg_events:
        msg += f" Requires at least {min_bkg_events} background events per bin."
    if blinding_threshold:
        msg += f" Blinding threshold is set to {blinding_threshold}."
    logger.info(msg)
    N_bins_input = rebin_hist.axes[0].size
    if N_bins_input != background_hist.axes[0].size:
        raise ValueError(
            f"Input histogram has {N_bins_input} bins, but background "
            f"histogram has {background_hist.axes[0].size} bins",
        )

    # determine events per bin the final histogram should have
    events_per_bin = rebin_hist.sum().value / N_bins_final
    logger.info(f"============ {round(events_per_bin, 3)} events per bin")

    # bookkeeping number of bins and number of events
    bin_count = 1
    N_events = 0
    N_signal = 0
    N_bkg_value = N_bkg_variance = 0

    x_max = rebin_hist.axes[0].edges[N_bins_input]
    x_min = rebin_hist.axes[0].edges[0]
    rebin_values = [x_max]

    h_view = rebin_hist.view()
    background_view = background_hist.view()
    signal_view = signal_hist.view()

    max_error = lambda value: value ** 2 / min_bkg_events

    blind_bool_func = lambda value, bkg_value: (
        value / np.sqrt(value + bkg_value) >= blinding_threshold
        if blinding_threshold else False
    )

    # starting at N_bins_input - 1 excludes overflow
    # ending loop at 0 to exclude underflow
    # starting at the end to allow checking for empty background bins
    for i in range(N_bins_input - 1, 0, -1):
        if bin_count == N_bins_final:
            # break as soon as N-1 bin edges have been determined --> final bin is x_max
            break

        N_signal += signal_view["value"][i]
        N_events += h_view["value"][i]
        N_bkg_value += background_view["value"][i]
        N_bkg_variance += background_view["variance"][i]
        if i % 100 == 0:
            logger.info(f"//////////// Bin {i} of {N_bins_input}, {N_events} events")
        if N_events >= events_per_bin:
            # when *N_events* surpasses threshold, check if background variance is small enough
            if N_bkg_variance < max_error(N_bkg_value):
                # when background variance is small enough, append the corresponding bin edge and count
                this_edge = rebin_hist.axes[0].edges[i]
                logger.info(
                    f"++++++++++ Append bin edge {bin_count} of {N_bins_final} at edge "
                    f"{this_edge}",
                )

                # recalculate events per bin
                last_bin_index = rebin_hist.axes[0].index(this_edge)
                _sum = rebin_hist.values()[:last_bin_index].sum()
                events_per_bin = _sum / (N_bins_final - bin_count + 1)
                logger.info(f"============ Continuing with {round(events_per_bin, 3)} events per bin")

                # check if this bin should be blinded
                should_be_blinded = blind_bool_func(N_signal, N_bkg_value)
                if should_be_blinded:
                    logger.warning(f"Blinding condition fulfilled, first bin edge is set to {this_edge}")
                    rebin_values = []

                # append bin edge and reset event counts
                rebin_values.append(this_edge)
                bin_count += 1
                N_events = N_signal = N_bkg_value = N_bkg_variance = 0
            else:
                this_edge = rebin_hist.axes[0].edges[i]
                logger.warning_once(
                    f"get_rebin_values_{bin_count}",
                    f"Background variance {N_bkg_variance} is too large for bin {i} with value {N_bkg_value}, "
                    f"skipping bin edge {this_edge}",
                )

    rebin_values.append(x_min)
    # change order of the bin edges to be ascending
    rebin_values = rebin_values[::-1]
    logger.info(f"final bin edges: {rebin_values}")
    return rebin_values


def apply_binning(hist, rebin_values: list):
    N_bins = len(rebin_values) - 1
    rebin_values_ptr = array.array("d", rebin_values)
    # rebin_values_ptr = np.array(rebin_values, dtype="float64")
    h_out = hist.Rebin(N_bins, hist.GetName(), rebin_values_ptr)
    return h_out


def check_empty_bins(hist, fill_empty: float = 1e-5, required_entries: int = 3) -> int:
    """
    Checks for empty bins, negative bin content, or bins with less than *required_entires* entries.
    When set to a value >= 0, empty or negative bin contents and errors are replaced with *fill_empty*.
    """
    print(f"============ Checking histogram {hist.GetName()} with {hist.GetNbinsX()} bins")
    import math
    max_error = lambda value: math.inf
    if required_entries > 0:
        # error above sqrt(N)/N means that we have less than N MC events
        # (assuming each MC event has the same weight)
        max_error = lambda value: value * math.sqrt(required_entries) / required_entries
    count = 0
    for i in range(1, hist.GetNbinsX() + 1):
        value = hist.GetBinContent(i)
        error = hist.GetBinError(i)
        if value <= 0:
            logger.info(f"==== Found empty or negative bin {i}, (value: {value}, error: {error})")
            count += 1
            if fill_empty >= 0:
                logger.info(f"     Bin {i} value + error will be filled with {fill_empty}")
                hist.SetBinContent(i, fill_empty)
                hist.SetBinError(i, fill_empty)

        if error > max_error(value):
            logger.warning(
                f"==== Bin {i} has less than {required_entries} entries (value: {value}, error: {error}); "
                f"Rebinning procedure might have to be restarted with less bins than {hist.GetNbinsX()}",
            )
    return count


def print_hist(hist, max_bins: int = 20):
    logger.info("Printing bin number, lower edge and bin content")
    for i in range(0, hist.GetNbinsX() + 2):
        if i > max_bins:
            return

        logger.info(f"{i} \t {hist.GetBinLowEdge(i)} \t {hist.GetBinContent(i)}")


from hbw.util import timeit_multiple


@timeit_multiple
def resolve_category_groups(param: dict[str, any], config_inst: od.Config):
    """
    Resolve the category groups for the given parameter *param* and the *config_inst*.

    :param param: The parameter to resolve
    """
    all_cats = set([x.name for x, _, _ in config_inst.walk_categories()])
    outp_param = {}
    for cat_name in list(param.keys()):
        resolved_cats = config_inst.x.category_groups.get(cat_name, cat_name)
        if resolved_cats:
            for resolved_cat in law.util.make_tuple(resolved_cats):
                if resolved_cat in all_cats:
                    outp_param[resolved_cat] = param[cat_name]

    return outp_param


@timeit_multiple
def resolve_category_groups_new(param: dict[str, any], config_inst: od.Config):
    # NOTE: this is only kept for timing comparisons of the `find_config_objects` function
    outp_param = {}
    for cat_name in list(param.keys()):
        resolved_cats = AnalysisTask.find_config_objects(
            (cat_name,), config_inst, od.Category,
            object_groups=config_inst.x.category_groups, deep=True,
        )
        if resolved_cats:
            for resolved_cat in law.util.make_tuple(resolved_cats):
                outp_param[resolved_cat] = param[cat_name]

    return outp_param


class HBWInferenceModelBase(
    HBWTask,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ReducerClassMixin,
    ProducerClassesMixin,
    MLModelsMixin,
    HistProducerClassMixin,
    InferenceModelMixin,
    HistHookMixin,
):
    resolution_task_cls = MergeHistograms
    single_config = False

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    unblind = luigi.BoolParameter(
        default=False,
    )


class ModifyDatacardsFlatRebin(
    HBWInferenceModelBase,
    law.LocalWorkflow,
    RemoteWorkflow,
):

    bins_per_category = SettingsParameter(
        default={},
        description="Number of bins per category in the format `cat_name=n_bins,...`; ",
    )

    # inference_category_rebin_processes = SettingsParameter(
    #     default={},
    #     significant=False,
    #     description="Dummy Parameter; only used via config_inst default",
    # )

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateDatacards=CreateDatacards,
    )

    @classmethod
    def resolve_param_values(cls, params):
        params = super().resolve_param_values(params)

        config_inst = params.get("config_insts")[0]

        # resolve default and groups for `bins_per_category`
        if not params.get("bins_per_category"):
            params["bins_per_category"] = config_inst.x.default_bins_per_category
        params["bins_per_category"] = resolve_category_groups(params["bins_per_category"], config_inst)

        config_inst.x.inference_category_rebin_processes = resolve_category_groups(
            config_inst.x.inference_category_rebin_processes,
            config_inst,
        )
        return params

    def get_n_bins(self, DEFAULT_N_BINS=8):
        """ Method to get the requested number of bins for the current category. Defaults to *DEFAULT_N_BINS*"""
        # NOTE: we assume single config here...
        config_category = self.branch_data.config_data[self.config_insts[0].name].category
        n_bins = self.bins_per_category.get(config_category, None)
        if not n_bins:
            logger.warning(f"No number of bins setup for category {config_category}; will default to {DEFAULT_N_BINS}.")
            n_bins = DEFAULT_N_BINS
        return int(n_bins)

    def get_rebin_processes(self):
        """
        Method to resolve the requested processes on which to flatten the histograms of the current category.
        Defaults to all processes of the current category.
        """
        config_inst = self.config_insts[0]
        config_category = self.branch_data.config_data[self.config_insts[0].name].category
        processes = self.branch_data.processes.copy()

        get_config_process = lambda proc: proc.config_data[self.config_insts[0].name].process
        rebin_process_condition = config_inst.x.inference_category_rebin_processes.get(config_category, None)
        if not rebin_process_condition:
            logger.warning(
                f"No rebin condition found for category {config_category}; rebinning will be flat "
                f"on all processes {[get_config_process(proc) for proc in processes]}",
            )
            return processes

        # transform `rebin_process_condition` into Callable if required
        if not isinstance(rebin_process_condition, Callable):
            _rebin_processes = list(rebin_process_condition)
            rebin_process_condition = lambda _proc_name: _proc_name in _rebin_processes

        for proc in processes.copy():
            proc_name = get_config_process(proc)
            # check for each process if the *rebin_process_condition*  is fulfilled
            if not rebin_process_condition(proc_name):
                processes.remove(proc)
        logger.info(
            f"Category {config_category} will be rebinned flat in processes "
            f"{[get_config_process(proc) for proc in processes]}",
        )
        return processes

    def get_background_processes(self):
        background_processes = [
            proc for proc in self.branch_data.processes.copy()
            if proc.name != "data_obs" and not proc.is_signal
        ]
        return background_processes

    def get_signal_processes(self):
        signal_processes = [
            proc for proc in self.branch_data.processes.copy()
            if proc.is_signal and (
                proc.name.startswith("ggHH_kl_1_kt_1") or proc.name.startswith("qqHH_CV_1_C2V_1_kl_1")
            )
        ]
        return signal_processes

    def create_branch_map(self):
        return list(self.inference_model_inst.categories)

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["datacards"] = self.reqs.CreateDatacards.req(self)

        return reqs

    def requires(self):
        reqs = {
            "datacards": self.reqs.CreateDatacards.req(self),
        }
        return reqs

    def output(self):
        cat_obj = self.branch_data.config_data[self.config_insts[0].name]
        basename = lambda name, ext: f"{name}__cat_{cat_obj.category}__var_{cat_obj.variable}.{ext}"
        n_bins = self.get_n_bins()
        return {
            "card": self.target(basename(f"datacard_rebin_{n_bins}", "txt")),
            "shapes": self.target(basename(f"shapes_rebin_{n_bins}", "root")),
            "edges": self.target(basename(f"edges_{n_bins}", "json")),
            "inspection": self.target(basename(f"inspection_{n_bins}", "json")),
        }

    def update_rates_and_obs_str(self, datacard: str) -> str:
        """
        Replace rates and observation with -1 in the datacard when blinding is enabled
        """
        rates_post = rates = datacard.split("\nrate")[1].split("\n")[0]
        for number in rates.split(" "):
            if number != "":
                # replace with -1 and keep the same length
                rates_post = rates_post.replace(number, f"-1{' ' * (len(number) - 2)}", 1)

        datacard = datacard.replace(rates, rates_post)

        n_obs = datacard.split("\nobservation")[1].split("\n")[0]
        datacard = datacard.replace(n_obs, f"{' ' * (len(n_obs) - 2)}-1")  # replace with -1 and keep the same length
        return datacard

    def run(self):
        inputs = self.input()
        outputs = self.output()

        inp_shapes = inputs["datacards"]["shapes"]
        inp_datacard = inputs["datacards"]["card"]

        # create a copy of the datacard with modified name of the shape file
        datacard = inp_datacard.load(formatter="text")
        datacard = datacard.replace(inp_shapes.basename, outputs["shapes"].basename)

        if not self.unblind and not self.inference_model_inst.skip_data:
            datacard = self.update_rates_and_obs_str(datacard)

        with uproot.open(inp_shapes.fn) as f_in:
            # determine which histograms are present
            cat_names, proc_names, syst_names = get_cat_proc_syst_names(f_in)

            if len(cat_names) != 1:
                raise Exception("Expected 1 category per file")

            cat_name = list(cat_names)[0]
            if cat_name != self.branch_data.name:
                raise Exception(
                    f"Category name in the histograms {cat_name} does not agree with the "
                    f"datacard category name {self.branch_data.name}",
                )

            # get all nominal histograms
            nominal_hists = {
                proc_name: f_in[get_hist_name(cat_name, proc_name)].to_hist()
                # proc_name: f_in[get_hist_name(cat_name, proc_name)].to_pyroot()
                for proc_name in proc_names
            }

            # determine all processes required for the category *cat_name* to determine the rebin values
            rebin_processes = self.get_rebin_processes()
            rebin_inf_proc_names = [proc.name for proc in rebin_processes]

            if diff := set(rebin_inf_proc_names).difference(nominal_hists.keys()):
                raise Exception(f"Histograms {diff} requested for rebinning but no corresponding "
                                "nominal histograms found")

            rebin_hists = [nominal_hists[proc_name] for proc_name in rebin_inf_proc_names]
            rebin_hist = sum(rebin_hists[1:], rebin_hists[0])  # sum all histograms to get the total histogram

            background_processes = self.get_background_processes()
            background_hists = [nominal_hists[proc.name] for proc in background_processes]
            background_hist = sum(background_hists[1:], background_hists[0])

            signal_processes = self.get_signal_processes()
            signal_hists = [nominal_hists[proc.name] for proc in signal_processes]
            signal_hist = sum(signal_hists[1:], signal_hists[0])

            logger.info(f"Finding rebin values for category {cat_name} using processes {rebin_inf_proc_names}")
            if "data_obs" in proc_names and self.inference_model_inst.skip_data is False:
                if "sig_vbf" in cat_name:
                    # for the VBF categories, we use a blinding threshold
                    blinding_threshold = 0.004
                elif "sig_" in cat_name:
                    # for the GGF categories, we use a blinding threshold
                    blinding_threshold = 0.008
                else:
                    # for all other categories, we do not blind the bins
                    blinding_threshold = 0.008
            else:
                blinding_threshold = None
            rebin_values = get_rebin_values(
                rebin_hist,
                signal_hist,
                background_hist,
                N_bins_final=self.get_n_bins(),
                min_bkg_events=3,
                blinding_threshold=blinding_threshold,
            )
            outputs["edges"].dump(rebin_values, formatter="json")

            # apply rebinning on all histograms and store resulting hists in a ROOT file
            out_file = uproot.recreate(outputs["shapes"].fn)
            negative_norms = {cat_name: defaultdict(dict)}
            variance_too_large = {cat_name: defaultdict(dict)}
            limited_stats = {cat_name: defaultdict(dict)}
            for key, h in f_in.items():

                key = key.split(";")[0]
                if "TH1" in str(h):
                    try:
                        h = f_in[key].to_hist()
                    except AttributeError:
                        continue

                    cat_name, proc_name, syst_name = get_cat_proc_syst_name_from_key(key)
                    h_rebin = apply_rebinning_edges(h, h.axes[0].name, rebin_values)
                    h_rebin_sum = h_rebin.sum()

                    # log whether norms are negative or variances too large
                    if (h_rebin_sum.value <= 0):
                        negative_norms[cat_name][proc_name][syst_name] = {
                            "value": h_rebin_sum.value,
                            "variance": h_rebin_sum.variance,
                        }
                    if (h_rebin_sum.variance >= h_rebin_sum.value ** 2):
                        variance_too_large[cat_name][proc_name][syst_name] = {
                            "value": h_rebin_sum.value,
                            "variance": h_rebin_sum.variance,
                        }
                    if syst_name == "nominal" and (h_rebin_sum.variance >= h_rebin_sum.value):
                        limited_stats[cat_name][proc_name][syst_name] = {
                            "value": h_rebin_sum.value,
                            "variance": h_rebin_sum.variance,
                        }

                    logger.debug(f"Inserting histogram with name {key}")
                    out_file[key] = h_rebin

        # TODO: update cards based on changed yields etc. At the moment I changed the
        # "observation" and "rate" lines in the datacard to -1 in c/f

        outputs["card"].dump(datacard, formatter="text")
        outputs["inspection"].dump({
            "negative_norms": negative_norms,
            "variance_too_large": variance_too_large,
            "limited_stats": limited_stats,
        }, indent=4, formatter="json")


from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase1D, ProcessPlotSettingMixin, VariablePlotSettingMixin,
)


class PlotShiftedInferencePlots(
    HBWInferenceModelBase,
    PlotBase1D,
    ProcessPlotSettingMixin,
    VariablePlotSettingMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    Task to create plots from the inference model output for each category, process, and shift.

    Example task call:
    law run hbw.PlotShiftedInferencePlots --custom-style-config legend_single_col \
        --inference-model weight1_hwwzztt_fullsyst
    """
    # TODO: disable unnecessary parameters
    # datasets = processes = variables = None

    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_1d.plot_shifted_variable",
        add_default_to_description=True,
    )
    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        ModifyDatacardsFlatRebin=ModifyDatacardsFlatRebin,
    )

    def create_branch_map(self):
        return list(self.inference_model_inst.categories)

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["rebinned_datacards"] = self.reqs.ModifyDatacardsFlatRebin.req(self)

        return reqs

    def requires(self):
        reqs = {
            "rebinned_datacards": self.reqs.ModifyDatacardsFlatRebin.req(self),
        }
        return reqs

    def output(self):
        return {
            # NOTE: removing target directory on remote target seems to not work, therefore we add incomplete dummy
            # such that we can rerun without having to remove this directory manually
            "combined_plots": self.target(f"combined_plots__{self.branch_data.name}.pdf"),
            "plots": self.target(f"plots__{self.branch_data.name}", dir=True),
        }

    def prepare_cf_hist(self, h: hist.Hist, variable_inst, shift_bin="nominal") -> hist.Hist:
        """
        Add a shift axis to the histogram if it does not exist.
        """
        if len(h.axes) != 1:
            raise ValueError("Expected histogram with only one axis")
        var_ax = h.axes[0]
        out_axes = [
            hist.axis.StrCategory([shift_bin], name="shift", growth=True),
            hist.axis.Variable(var_ax.edges, name=variable_inst.name, label=variable_inst.x_title),
        ]
        return hist.Hist(*out_axes, storage=h.storage_type(), data=[h.view(flow=True)])

    def skip_process(self, process_inst, category_inst) -> bool:
        # skip non-SM signal processes
        if "hh_ggf" in process_inst.name and "kl1_kt1" not in process_inst.name:
            return True
        if "hh_vbf" in process_inst.name and "kv1_k2v1_kl1" not in process_inst.name:
            return True
        return False

    @law.decorator.log
    @law.decorator.localize
    @law.decorator.safe_output
    def run(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        inf_inst = self.inference_model_inst
        inputs = self.input()
        output = self.output()

        inp_shapes = inputs["rebinned_datacards"]["shapes"]

        config_inst = self.config_insts[0]
        config_data = self.branch_data.config_data[config_inst.name]
        category_inst = config_inst.get_category(config_data.category)
        variable_inst = config_inst.get_variable(config_data.variable)
        variable_inst.x.rebin = 1  # do not try rebinning here, we already did that in the datacard modification

        with uproot.open(inp_shapes.fn) as f_in:
            # determine which histograms are present
            cat_names, proc_names, syst_names = get_cat_proc_syst_names(f_in)

            if len(cat_names) != 1:
                raise Exception("Expected 1 category per file")

            cat_name = list(cat_names)[0]
            if cat_name != self.branch_data.name:
                raise Exception(
                    f"Category name in the histograms {cat_name} does not agree with the "
                    f"datacard category name {self.branch_data.name}",
                )

            # create plots for each process and shift
            with PdfPages(output["combined_plots"].abspath) as pdf:
                for config_proc_name in inf_inst.processes:
                    proc_name = inf_inst.inf_proc(config_proc_name)
                    proc_inst = config_inst.get_process(config_proc_name)
                    if self.skip_process(proc_inst, category_inst):
                        logger.info(f"Skipping process {proc_inst.name} for category {cat_name}")
                        continue
                    h_nom = f_in[get_hist_name(cat_name, proc_name)].to_hist()
                    h_nom = self.prepare_cf_hist(h_nom, variable_inst, shift_bin="nominal")
                    # TODO: when no syst_names, make nominal plot only
                    for syst_name in sorted(syst_names):
                        if not syst_name.endswith("Down"):
                            continue

                        hist_name = get_hist_name(cat_name, proc_name, syst_name)
                        h_down = f_in.get(hist_name)
                        if not h_down:
                            continue

                        # mapping of shift names to config shift names
                        config_shift_name = syst_name.replace("Down", "_down")
                        if "murf_envelope" in config_shift_name:
                            config_shift_name = "murf_envelope_down"
                        elif "pdf" in config_shift_name:
                            config_shift_name = "pdf_down"

                        shift_source = config_shift_name.replace("_down", "")
                        plot_name = f"{cat_name}__{proc_inst.name}__{shift_source}.pdf"
                        print(f"Preparing plot {plot_name}")

                        shift_insts = {
                            "nominal": config_inst.get_shift("nominal").copy_shallow(),
                            "down": config_inst.get_shift(f"{shift_source}_down").copy_shallow(),
                            "up": config_inst.get_shift(f"{shift_source}_up").copy_shallow(),
                        }
                        # convert to hist.Histogram
                        h_down = h_down.to_hist()

                        # if down is present, up must be present as well
                        h_up = f_in.get(hist_name.replace("Down", "Up")).to_hist()

                        h_combined = (
                            h_nom.copy() +
                            self.prepare_cf_hist(h_down, variable_inst, shift_bin=shift_insts["down"].name) +
                            self.prepare_cf_hist(h_up, variable_inst, shift_bin=shift_insts["up"].name)
                        )

                        hists = {proc_inst: h_combined}

                        # temporarily use a merged luminostiy value, assigned to the first config
                        lumi = sum([_config_inst.x.luminosity for _config_inst in self.config_insts])
                        with law.util.patch_object(config_inst.x, "luminosity", lumi):
                            # call the plot function
                            fig, _ = self.call_plot_func(
                                self.plot_function,
                                hists=hists,
                                config_inst=config_inst,
                                category_inst=category_inst.copy_shallow(),
                                variable_insts=[variable_inst.copy_shallow()],
                                shift_insts=list(shift_insts.values()),
                                **self.get_plot_parameters(),
                            )
                            output["plots"].child(plot_name, type="f").dump(fig, formatter="mpl")
                            # save the figure to the pdf
                            pdf.savefig(fig)
                            # close the figure to avoid memory issues
                            plt.close(fig)

            logger.info(f"Finished creating plots for shifted inference model {cat_name}.")


class PrepareInferenceTaskCalls(HBWInferenceModelBase):
    """
    Simple task that produces string to run certain tasks in Inference
    """
    # upstream requirements
    reqs = Requirements(
        ModifyDatacardsFlatRebin=ModifyDatacardsFlatRebin,
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["rebinned_datacards"] = self.reqs.ModifyDatacardsFlatRebin.req(self)

        return reqs

    def requires(self):
        reqs = {
            "rebinned_datacards": self.reqs.ModifyDatacardsFlatRebin.req(self),
        }
        return reqs

    def output(self):
        return {
            "Run": self.target("Run.sh"),
            "PlotUpperLimitsAtPoint": self.target("PlotUpperLimitsAtPoint.txt"),
            "PlotUpperLimits_kl": self.target("PlotUpperLimits_kl.txt"),
            "PlotUpperLimits_c2v": self.target("PlotUpperLimits_c2v.txt"),
            "FitDiagnostics": self.target("FitDiagnostics.txt"),
            "PullsAndImpacts": self.target("PullsAndImpacts.txt"),
        }

    def run(self):
        inputs = self.input()
        output = self.output()

        # string that represents the version of datacards
        identifier_list = [*self.configs, self.selector, self.inference_model, self.version]
        identifier = "__".join(identifier_list)

        # get the datacard names from the inputs
        collection = inputs["rebinned_datacards"]["collection"]
        cards_path = {collection[key]["card"].dirname for key in collection.keys()}
        if len(cards_path) != 1:
            raise Exception("Expected only one datacard path")
        cards_path = cards_path.pop()
        # pruned_cards_path = f"{cards_path}/pruned"
        decorrelated_cards_path = f"{cards_path}/decorrelated"

        card_fns = [collection[key]["card"].basename for key in collection.keys()]

        # get the category names from the inference models
        categories = self.inference_model_inst.categories
        cat_names = [c.name for c in categories]

        # combine category names with card fn to a single string
        datacards = ",".join([f"{cat_name}=$CARDS_PATH/{card_fn}" for cat_name, card_fn in zip(cat_names, card_fns)])

        # # name of the output root file that contains the Pre+Postfit shapes
        # output_file = ""

        base_cmd = f"export BASE_CARDS_PATH={cards_path}" + "\n" + f"export CARDS_PATH={decorrelated_cards_path}" + "\n"
        full_cmd = base_cmd

        # run pruning helper on cards
        for card_fn in card_fns:
            cmd = f"prepare_cards.py $BASE_CARDS_PATH/{card_fn}"
            full_cmd += cmd + "\n"

        full_cmd += "\n\n"
        print(full_cmd)

        base_cmd = f"export CARDS_PATH={decorrelated_cards_path}" + "\n"

        lumi = sum([config_inst.x.luminosity.get("nominal") for config_inst in self.config_insts]) * 0.001
        lumi = f"'{lumi:.1f} fb^{{-1}}'"

        is_signal_region = lambda cat_name: (
            "sig_" in cat_name or cat_name == "sr__boosted" or "hh_ggf_" in cat_name or "hh_vbf_" in cat_name
        )

        # creating limits per signal region vs all 1b regions vs all 2b regions vs all regions combined
        multi_sig_cards = ":".join([
            f"{cat_name}=$CARDS_PATH/{card_fn}"
            for cat_name, card_fn in zip(cat_names, card_fns) if is_signal_region(cat_name)
        ])
        multi_sig_card_names = ",".join([
            cat_name for cat_name in cat_names if is_signal_region(cat_name)
        ])
        cards_1b = ",".join([
            f"{cat_name}=$CARDS_PATH/{card_fn}" for cat_name, card_fn in zip(cat_names, card_fns) if "1b" in cat_name
        ])
        cards_2b = ",".join([
            f"{cat_name}=$CARDS_PATH/{card_fn}" for cat_name, card_fn in zip(cat_names, card_fns) if "2b" in cat_name
        ])
        cards_boosted = ",".join([
            f"{cat_name}=$CARDS_PATH/{card_fn}"
            for cat_name, card_fn in zip(cat_names, card_fns) if "boosted" in cat_name
        ])

        multi_datacards = []
        multi_datacard_names = []
        for cards, this_identifier in [
            (multi_sig_cards, multi_sig_card_names),
            (cards_1b, "1b_combined"),
            (cards_2b, "2b_combined"),
            (cards_boosted, "boosted_combined"),
            (datacards, identifier),
        ]:
            if cards:
                multi_datacards.append(cards)
                multi_datacard_names.append(this_identifier)

        multi_datacards = ":".join(multi_datacards)
        multi_datacard_names = ",".join(multi_datacard_names)
        print("\n\n")

        # print(base_cmd)
        # for card, _ident in zip(card_fns, identifier):
        #     cmd = f"ValidateDatacard.py $CARDS_PATH/{card} --jsonFile $CARDS_PATH//validation_{_ident}.json"
        #     print(cmd)
        # print("\n\n")

        cmd = (
            f"law run PlotUpperLimitsAtPoint --version {identifier} --campaign {lumi} "
            f"--multi-datacards {multi_datacards} "
            f"--datacard-names {multi_datacard_names} "
            f"--UpperLimits-workflow htcondor "
        )
        full_cmd += cmd + "\n\n"
        print(base_cmd + cmd, "\n\n")
        output["PlotUpperLimitsAtPoint"].dump(cmd, formatter="text")

        # creating upper limits for kl=1
        cmd = (
            f"law run PlotUpperLimitsAtPoint --version {identifier} --campaign {lumi} "
            f"--multi-datacards {datacards} "
            f"--datacard-names {identifier}"
        )
        print(base_cmd + cmd, "\n\n")
        # full_cmd += cmd + "\n\n"

        # creating kl scan
        cmd = (
            f"law run PlotUpperLimits --version {identifier} --campaign {lumi} --datacards {datacards} "
            f"--xsec fb --y-log --scan-parameters kl,-20,25,46 --UpperLimits-workflow htcondor"
        )
        print(base_cmd + cmd, "\n\n")
        full_cmd += cmd + "\n\n"
        output["PlotUpperLimits_kl"].dump(cmd, formatter="text")

        # creating C2V scan
        cmd = (
            f"law run PlotUpperLimits --version {identifier} --campaign {lumi} --datacards {datacards} "
            f"--xsec fb --y-log --scan-parameters C2V,-4,6,11 --UpperLimits-workflow htcondor"
        )
        print(base_cmd + cmd, "\n\n")
        full_cmd += cmd + "\n\n"
        output["PlotUpperLimits_c2v"].dump(cmd, formatter="text")

        # running FitDiagnostics for Pre+Postfit plots
        cmd = (
            f"law run FitDiagnostics --version {identifier} --datacards {datacards} "
            f"--skip-b-only"
        )
        print(base_cmd + cmd, "\n\n")
        # full_cmd += cmd + "\n\n"
        output["FitDiagnostics"].dump(cmd, formatter="text")

        # running FitDiagnostics for Pre+Postfit plots
        cmd = (
            f"law run PlotPullsAndImpacts --version {identifier} --campaign {lumi} --datacards {datacards} "
            f"--order-by-impact --PullsAndImpacts-workflow htcondor --mc-stats --parameters-per-page 50"
        )
        pulls_and_imacts_params = "workflow=htcondor"
        if not self.unblind and not self.inference_model_inst.skip_data:
            pulls_and_imacts_params += ",custom-args='--rMax 200 --rMin -200'"
            # NOTE: the custom args do not work in combination with job submission
            cmd += " --unblinded"
        cmd += f" --PullsAndImpacts-{{{pulls_and_imacts_params}}}"

        print(base_cmd + cmd, "\n\n")
        full_cmd += cmd + "\n\n"
        output["PullsAndImpacts"].dump(cmd, formatter="text")

        # running PreAndPostfitShapes for Pre+Postfit plots
        cmd = (
            f"law run PreAndPostFitShapes --version {identifier} --datacards {datacards} "
            # f"--output-name {output_file}"
        )
        print(base_cmd + cmd, "\n\n")
        full_cmd += cmd + "\n\n"
        output["FitDiagnostics"].dump(cmd, formatter="text")

        # dump the full command to one output file
        run_script = self.create_run_script(identifier, full_cmd)
        output["Run"].dump(run_script, formatter="text")

    def create_run_script(self, identifier, full_cmd):
        full_cmd_with_fetch = full_cmd.replace("law run", f"run_and_fetch_cmd {identifier} law run")
        run_script = f"""#!/bin/bash

mkdir -p $DHI_DATA/fetched_plots/{identifier} && cd $DHI_DATA/fetched_plots/{identifier}

run_and_fetch_cmd() {{
    local folder="$1"
    cd "$DHI_DATA/fetched_plots/$folder" || exit 1
    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] ${{*:2}}"
        echo "[DRY-RUN] ${{*:2}} --fetch-output 0,a"
    else
        eval "${{@:2}}"
        eval "${{@:2}} --fetch-output 0,a"
    fi
}}

{full_cmd_with_fetch}
"""
        return run_script
