# coding: utf-8

"""
Tasks related to the creation and modification of datacards for inference purposes.
"""

from __future__ import annotations
import json
from columnflow.tasks.framework.base import AnalysisTask
import yaml
import textwrap
import os
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase1D, ProcessPlotSettingMixin, VariablePlotSettingMixin,
)

from columnflow.types import Any

from typing import Callable
from collections import defaultdict

import luigi
import law
import order as od

from columnflow.tasks.framework.base import Requirements, TaskShifts
from columnflow.tasks.framework.parameters import SettingsParameter
from columnflow.tasks.framework.mixins import (
    CalibratorClassesMixin, SelectorClassMixin, ReducerClassMixin, ProducerClassesMixin, MLModelsMixin,
    InferenceModelMixin, InferenceModelClassMixin,
    HistHookMixin, HistProducerClassMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.histograms import MergeHistograms
from columnflow.tasks.cms.inference import CreateDatacards
from columnflow.inference import InferenceModel
from columnflow.util import dev_sandbox, maybe_import, DotDict
from hbw.tasks.base import HBWTask
from hbw.hist_util import apply_rebinning_edges
import hbw.inference.constants as const

np = maybe_import("numpy")


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
        blinded_by_significance: bool = False,
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

    def max_error(value):
        return value ** 2 / min_bkg_events

    def blind_bool_func(value, bkg_value):
        return (
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
            logger.info(
                f"//////////// Bin {i} of {N_bins_input}, {N_events} events")
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
                events_per_bin = _sum / (N_bins_final - bin_count)
                logger.info(
                    f"============ Continuing with {round(events_per_bin, 3)} events per bin")
                # check if this bin should be blinded
                if blinded_by_significance:
                    should_be_blinded = blind_bool_func(N_signal, N_bkg_value)
                    if should_be_blinded:
                        logger.warning(
                            f"Blinding condition fulfilled, first bin edge is set to {this_edge}")
                        rebin_values = []
                else:
                    if this_edge >= blinding_threshold:
                        logger.warning(f"Bin edge {this_edge} is above blinding threshold {blinding_threshold}, "
                                       f"first bin edge is set to {this_edge}")
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


class HBWInferenceModelBase(
    HBWTask,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ReducerClassMixin,
    ProducerClassesMixin,
    MLModelsMixin,
    HistProducerClassMixin,
    HistHookMixin,
):
    resolution_task_cls = MergeHistograms
    single_config = False

    sandbox = dev_sandbox(law.config.get(
        "analysis", "default_columnar_sandbox"))

    # attribute to disable fully unblinding the datacard when using real data (False until pre-approval)
    unblind_step1 = True
    fully_unblind = True

    @property
    def inference_model_cls(self):
        return InferenceModel.get_cls(self.inference_model)

    @property
    def real_data(self):
        # when skip_data is used, "data_obs" is fake data, so we can "unblind"
        return not self.inference_model_cls.skip_data

    @property
    def partially_unblinded(self):
        return self.real_data and not self.inference_model_cls.unblind

    @property
    def unblinded_step1(self):
        return self.unblind_step1 and self.real_data and self.inference_model_cls.unblind

    @property
    def fully_unblinded(self):
        return self.fully_unblind and self.real_data and self.inference_model_cls.unblind

    def configs_str(self, configs):
        configs_str = "_".join(configs)
        configs_str = configs_str.replace("c22prev14_c22postv14", "2022")
        configs_str = configs_str.replace("c23prev14_c23postv14", "2023")
        return configs_str


class ModifyDatacardsFlatRebin(
    HBWInferenceModelBase,
    InferenceModelMixin,
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

    min_bkg_events = 12

    @classmethod
    def resolve_param_values(cls, params):
        params = super().resolve_param_values(params)

        config_inst = params.get("config_insts")[0]

        # resolve default and groups for `bins_per_category`
        if not params.get("bins_per_category"):
            params["bins_per_category"] = config_inst.x.default_bins_per_category
        params["bins_per_category"] = resolve_category_groups(
            params["bins_per_category"], config_inst)

        config_inst.x.inference_category_rebin_processes = resolve_category_groups(
            config_inst.x.inference_category_rebin_processes,
            config_inst,
        )
        return params

    def get_n_bins(self, DEFAULT_N_BINS=20):
        """ Method to get the requested number of bins for the current category. Defaults to *DEFAULT_N_BINS*"""
        # NOTE: we assume single config here...
        config_category = self.branch_data.inf_cat.config_data[self.config_insts[0].name].category
        n_bins = self.bins_per_category.get(config_category, None)
        if not n_bins:
            logger.warning(
                f"No number of bins setup for category {config_category}; will default to {DEFAULT_N_BINS}.")
            n_bins = DEFAULT_N_BINS
        return int(n_bins)

    def get_rebin_processes(self):
        """
        Method to resolve the requested processes on which to flatten the histograms of the current category.
        Defaults to all processes of the current category.
        """
        config_inst = self.config_insts[0]
        config_category = self.branch_data.inf_cat.config_data[self.config_insts[0].name].category
        processes = self.branch_data.inf_cat.processes.copy()

        def get_config_process(proc):
            return proc.config_data[self.config_insts[0].name].process
        rebin_process_condition = config_inst.x.inference_category_rebin_processes.get(
            config_category, None)
        if not rebin_process_condition:
            logger.warning(
                f"No rebin condition found for category {config_category}; rebinning will be flat "
                f"on all processes {[get_config_process(proc) for proc in processes]}",
            )
            return processes

        # transform `rebin_process_condition` into Callable if required
        if not isinstance(rebin_process_condition, Callable):
            _rebin_processes = list(rebin_process_condition)

            def rebin_process_condition(_proc_name):
                return _proc_name in _rebin_processes

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
            proc for proc in self.branch_data.inf_cat.processes.copy()
            if proc.name != "data_obs" and not proc.is_signal
        ]
        return background_processes

    def get_signal_processes(self, cfg):
        signal_processes = [
            proc
            for proc in self.branch_data.inf_cat.processes
            if proc.is_signal and (
                (
                    cfg.has_tag("is_hh") and (
                        proc.name.startswith("ggHH_kl_1_kt_1") or proc.name.startswith(
                            "qqHH_CV_1_C2V_1_kl_1")
                    )
                ) or (
                    cfg.has_tag("is_hhh") and proc.name.startswith("hhh_")
                )
            )
        ]
        return signal_processes

    def create_branch_map(self):
        # TODO: add config insts to branch data
        return [
            DotDict({
                "inf_cat": inf_cat,
                "configs": self.configs,
                "req_branch": i,
            })
            for i, inf_cat in enumerate(self.inference_model_inst.categories)
        ]

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["datacards"] = self.reqs.CreateDatacards.req(
            self, configs=self.configs)

        return reqs

    def requires(self):
        reqs = {
            "datacards": self.reqs.CreateDatacards.req_different_branching(
                self,
                branch=0,
                configs=self.branch_data.configs,
            ),
        }
        return reqs

    def output(self):
        configs_repr = self.configs_str(self.branch_data.configs)
        cat_obj = self.branch_data.inf_cat.config_data[self.config_insts[0].name]

        def basename(name, ext):
            return f"{name}__cfg_{configs_repr}__cat_{cat_obj.category}.{ext}"
        n_bins = self.get_n_bins()
        return {
            "card": self.target(basename(f"card_{n_bins}", "txt")),
            "shapes": self.target(basename(f"shapes_{n_bins}", "root")),
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
                rates_post = rates_post.replace(
                    number, f"-1{' ' * (len(number) - 2)}", 1)
        datacard = datacard.replace(rates, rates_post)

        n_obs = "\nobservation" + \
            datacard.split("\nobservation")[1].split("\n")[0]
        # replace with -1 and keep the same length
        datacard = datacard.replace(
            n_obs, f"\nobservation{' ' * (len(n_obs) - 2)}-1")
        return datacard

    def run(self):
        import uproot
        inputs = self.input()
        branch_inputs = inputs["datacards"].targets[self.branch_data.inf_cat.name]
        outputs = self.output()

        inp_shapes = branch_inputs["shapes"]
        inp_datacard = branch_inputs["card"]

        # create a copy of the datacard with modified name of the shape file
        datacard = inp_datacard.load(formatter="text")
        datacard = datacard.replace(
            inp_shapes.basename, outputs["shapes"].basename)

        if self.partially_unblinded:
            datacard = self.update_rates_and_obs_str(datacard)

        with uproot.open(inp_shapes.fn) as f_in:
            # determine which histograms are present
            cat_names, proc_names, syst_names = get_cat_proc_syst_names(f_in)

            if len(cat_names) != 1:
                raise Exception("Expected 1 category per file")

            cat_name = list(cat_names)[0]
            if cat_name != self.branch_data.inf_cat.name:
                raise Exception(
                    f"Category name in the histograms {cat_name} does not agree with the "
                    f"datacard category name {self.branch_data.inf_cat.name}",
                )

            # get all nominal histograms
            nominal_hists = {
                proc_name: f_in[get_hist_name(cat_name, proc_name)].to_hist()
                # proc_name: f_in[get_hist_name(cat_name, proc_name)].to_pyroot()
                for proc_name in proc_names
            }

            # determine all processes required for the category *cat_name* to determine the rebin values
            rebin_processes = self.get_rebin_processes()
            background_processes = self.get_background_processes()

            rebin_inf_proc_names = [proc.name for proc in rebin_processes]
            background_proc_names = [
                proc.name for proc in background_processes]
            # background_processes = self.get_background_processes()
            # __import__("IPython").embed()
            if diff := set(rebin_inf_proc_names).difference(nominal_hists.keys()):
                for p in diff:
                    if p not in nominal_hists.keys():
                        rebin_inf_proc_names.remove(p)
                    else:
                        raise Exception(f"Histograms {diff} requested for rebinning but no corresponding"
                                        "nominal histograms found")
            if diff := set(background_proc_names).difference(nominal_hists.keys()):
                for p in diff:
                    if p not in nominal_hists.keys():
                        background_proc_names.remove(p)
                    else:
                        raise Exception(f"Histograms {diff} requested for rebinning but no corresponding "
                                        "nominal histograms found")

            rebin_hists = [nominal_hists[proc_name]
                           for proc_name in rebin_inf_proc_names]
            # sum all histograms to get the total histogram
            rebin_hist = sum(rebin_hists[1:], rebin_hists[0])

            # background_processes = self.get_background_processes()
            # bkg_names = [proc.name for proc in background_processes]
            # for b in background_processes:
            #     print(b.name)
            #     if b.name not in nominal_hists.keys():
            #         print(b.name)
            #         # logger.warning(f"Background process {b.name} not found in nominal histograms, removing from list")  # noqa E501
            #         background_processes.remove(b)
            try:
                background_hists = [nominal_hists[proc_name]
                                    for proc_name in background_proc_names]
            except KeyError as e:
                __import__("IPython").embed()
                logger.error(
                    f"Error occurred while creating background histogram: {e}")
                raise
            background_hist = sum(background_hists[1:], background_hists[0])

            signal_processes = self.get_signal_processes(self.config_insts[0])
            signal_hists = [nominal_hists[proc.name]
                            for proc in signal_processes]
            signal_hist = sum(signal_hists[1:], signal_hists[0])

            logger.info(
                f"Finding rebin values for category {cat_name} using processes {rebin_inf_proc_names}")
            if "data_obs" in proc_names and self.partially_unblinded:
                if "sig_vbf" in cat_name:
                    # for the VBF categories, we use a blinding threshold
                    blinding_threshold = 0.004
                elif "sig_" in cat_name:
                    # for the GGF categories, we use a blinding threshold
                    blinding_threshold = 0.008
                elif "2b__ml_hhh_signal" in cat_name:
                    # for the GGF categories, we use a blinding threshold
                    # blinding_threshold = 0.000003
                    blinding_threshold = 0.9
                elif "3b__ml_hhh_signal" in cat_name:
                    # for the GGF categories, we use a blinding threshold
                    # blinding_threshold = 0.0005
                    blinding_threshold = 3.03
                elif "4b__ml_hhh_signal" in cat_name:
                    # for the GGF categories, we use a blinding threshold
                    # blinding_threshold = 0.001
                    blinding_threshold = 3.95
                elif "boosted__ml_hhh_signal" in cat_name:
                    # for the GGF categories, we use a blinding threshold
                    # blinding_threshold = 0.001
                    blinding_threshold = 2.2
                else:
                    # for all other categories, we do not blind the bins
                    blinding_threshold = 10
            else:
                # blinding_threshold = None
                blinding_threshold = 10
            rebin_values = get_rebin_values(
                rebin_hist,
                signal_hist,
                background_hist,
                N_bins_final=self.get_n_bins(),
                min_bkg_events=self.min_bkg_events,
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

                    cat_name, proc_name, syst_name = get_cat_proc_syst_name_from_key(
                        key)
                    h_rebin = apply_rebinning_edges(
                        h, h.axes[0].name, rebin_values)
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

        self.publish_message(f"datacard written to {outputs['card'].abspath}")


class PlotShiftedInferencePlots(
    HBWInferenceModelBase,
    PlotBase1D,
    ProcessPlotSettingMixin,
    VariablePlotSettingMixin,
    InferenceModelMixin,
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
        # TODO: branch data should be split into config group + category
        return list(self.inference_model_inst.categories)

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["rebinned_datacards"] = self.reqs.ModifyDatacardsFlatRebin.req(
            self)

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

    def prepare_cf_hist(self, h, variable_inst, shift_bin="nominal"):
        """
        Add a shift axis to the histogram if it does not exist.
        """
        import hist
        if len(h.axes) != 1:
            raise ValueError("Expected histogram with only one axis")
        var_ax = h.axes[0]
        out_axes = [
            hist.axis.StrCategory([shift_bin], name="shift", growth=True),
            hist.axis.Variable(
                var_ax.edges, name=variable_inst.name, label=variable_inst.x_title),
        ]
        return hist.Hist(*out_axes, storage=h.storage_type(), data=[h.view(flow=True)])

    def skip_process(self, process_inst, category_inst) -> bool:
        # skip non-SM signal processes
        if "hh_ggf" in process_inst.name and "kl1_kt1" not in process_inst.name:
            return True
        if "hh_vbf" in process_inst.name and "kv1_k2v1_kl1" not in process_inst.name:
            return True
        return False

    def get_config_shift_name(self, syst_name, analysis_tag="CMS_HIG25018_"):
        """
        Mapping of shift names to config shift names
        """
        if syst_name is None:
            config_shift_name = "dummy_down"
            return config_shift_name

        shift_source = syst_name.replace("Up", "").replace("Down", "")

        if "murf_envelope" in shift_source:
            shift_source = "murf_envelope"
        elif "pdf" in shift_source:
            shift_source = "pdf"
        elif "fsr" in shift_source:
            shift_source = "fsr"

        config_shift_source = const.rename_systematics_inverted.get(
            shift_source, shift_source)
        config_shift_source = config_shift_source.replace(analysis_tag, "")
        return f"{config_shift_source}_down"

    @law.decorator.log
    # @law.decorator.localize
    @law.decorator.safe_output
    def run(self):
        import uproot
        import matplotlib
        matplotlib.use("Agg")  # use non-interactive backend for plotting
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
        # do not try rebinning here, we already did that in the datacard modification
        variable_inst.x.rebin = 1

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
                        logger.info(
                            f"Skipping process {proc_inst.name} for category {cat_name}")
                        continue
                    h_nom = f_in[get_hist_name(cat_name, proc_name)].to_hist()
                    h_nom = self.prepare_cf_hist(
                        h_nom, variable_inst, shift_bin="nominal")
                    # TODO: when no syst_names, make nominal plot only
                    if not syst_names:
                        syst_names = [None]
                    for syst_name in sorted(syst_names):
                        if syst_name and not syst_name.endswith("Down"):
                            continue

                        hist_name = get_hist_name(
                            cat_name, proc_name, syst_name)
                        h_down = f_in.get(hist_name)
                        if not h_down:
                            continue

                        config_shift_name = self.get_config_shift_name(
                            syst_name)

                        shift_source = config_shift_name.replace("_down", "")
                        plot_name = f"{cat_name}__{proc_inst.name}__{shift_source}.pdf"
                        logger.info(f"Preparing plot {plot_name}")

                        shift_insts = {
                            "nominal": config_inst.get_shift("nominal").copy_shallow(),
                        }

                        for cfg_inst in self.config_insts:
                            shift_label_postfix = ""
                            # TODO: add cpn_tag to shift label if part of shift_source
                            if cfg_inst.x.cpn_tag in shift_source:
                                shift_label_postfix = f" ({cfg_inst.x.cpn_tag})"
                            shift_source = shift_source.replace(
                                f"_{cfg_inst.x.cpn_tag}", "")
                            if cfg_inst.has_shift(f"{shift_source}_up"):
                                # use shifts and config inst where corresponding shift is defined
                                shift_insts["up"] = cfg_inst.get_shift(
                                    f"{shift_source}_up").copy_shallow()
                                shift_insts["down"] = cfg_inst.get_shift(
                                    f"{shift_source}_down").copy_shallow()

                                # add postfix to labels
                                shift_insts["up"].label += shift_label_postfix
                                shift_insts["down"].label += shift_label_postfix
                                config_inst = cfg_inst
                                break
                        if "up" not in shift_insts.keys():
                            raise ValueError(
                                f"Shift {shift_source} not found in any of the configs {self.config_insts}",
                            )

                        # convert to hist.Histogram
                        h_down = h_down.to_hist()

                        # if down is present, up must be present as well
                        h_up = f_in.get(hist_name.replace(
                            "Down", "Up")).to_hist()

                        h_combined = (
                            h_nom.copy() +
                            self.prepare_cf_hist(h_down, variable_inst, shift_bin=shift_insts["down"].name) +
                            self.prepare_cf_hist(
                                h_up, variable_inst, shift_bin=shift_insts["up"].name)
                        )

                        hists = {proc_inst: h_combined}

                        # temporarily use a merged luminostiy value, assigned to the first config
                        lumi = sum(
                            [_config_inst.x.luminosity for _config_inst in self.config_insts])
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
                            output["plots"].child(plot_name, type="f").dump(
                                fig, formatter="mpl")
                            # save the figure to the pdf
                            pdf.savefig(fig)
                            # close the figure to avoid memory issues
                            plt.close(fig)

            self.publish_message(
                f"Finished creating plots for shifted inference model {cat_name}."
                f" Plots are stored in \n{output['plots'].abspath}",
            )


class PrepareInferenceTaskCalls(
    HBWInferenceModelBase,
    InferenceModelClassMixin,
):
    """
    Simple task that produces string to run certain tasks in Inference
    """
    # upstream requirements
    reqs = Requirements(
        ModifyDatacardsFlatRebin=ModifyDatacardsFlatRebin,
    )

    # output_collection_cls = law.NestedSiblingFileCollection
    config_groups = law.MultiCSVParameter(
        # default=(("c22prev14", "c22postv14"), ("c23prev14", "c23postv14")),
        default=(("c22prev14", "c22postv14", "c23prev14", "c23postv14"),),
        description="List of config groups to use for this task.",
        significant=False,
    )

    requested_keys = law.CSVParameter(
        default=[
            "prepare_cards",
            # "LimitsPerCategory",
            # "qqHH_LimitsPerCategory",
            # "LimitsPerCampaign",
            "pointlimits",
            "qqhh_pointlimits",
            "impacts",
            # "likelihood_r",
            "postfitshapes",
            "prefitshapes",
            "likelihood_kl",
            "likelihood_c2v",
            "likelihood_kl_c2v",
            # "likelihood_kl_kt",
            "limits_kl",
            "limits_c2v",
            "gof",
            # "multilimits_c2v",
            # "multilimits_kl",
            # "qqhh_multilimits_c2v",
            # "qqhh_multilimits_kl",
        ],
        description="List of inference task keys to prepare calls for.",
        significant=False,
    )
    rerun = luigi.BoolParameter(
        default=False,
        description="Whether to rerun the tasks even if their output exists.",
        significant=False,
    )
    # systematics that are frozen for kl and c2v scans
    frozen_for_scans = ",".join([
        "THU_HH", "pdf_Higgs_hh_vbf", "pdf_Higgs_hh_ggf", "QCDscale_hh_vbf",
        "BR_hbb", "BR_hww", "BR_hzz", "BR_htt", "BR_hgg",
    ])

    cards_version = luigi.Parameter(
        default="",
        description="Optional version string to append to the datacard output path.",
        significant=True,
    )

    # # TODO: add param to not delete existing cards each time :)
    recreate_datacards = luigi.BoolParameter(
        default=False,
        description="Whether to recreate only the Run.sh script.",
        significant=False,
    )
    run_script_name = luigi.Parameter(
        default="Run.sh",
        description="Name of the run script to create.",
        significant=True,
    )

    @classmethod
    def resolve_instances(cls, params: dict[str, Any], shifts: TaskShifts) -> dict[str, Any]:
        # params["known_shifts"] = shifts

        # TODO: freeze stuff inst dict

        for config_group in params["config_groups"]:
            _params = params.copy()
            _params["configs"] = config_group
            _params["config_insts"] = [
                config_inst for config_inst in params["config_insts"]
                if config_inst.name in config_group
            ]
            _params["inst_dict"] = {
                "configs": _params["configs"],
                "config_insts": _params["config_insts"],
            }
            cls.reqs.ModifyDatacardsFlatRebin.resolve_instances(
                _params, shifts)

        return params

    @classmethod
    def modify_param_values(cls, params: dict[str, Any]) -> dict[str, Any]:
        if params["config_groups"]:
            configs = tuple(
                config for config_group in params["config_groups"] for config in config_group)
            if configs != params["configs"]:
                logger.warning_once(
                    f"update_configs_{params['configs']}_from_groups_{params['config_groups']}",
                    f"Overwriting 'configs' parameter with values {configs} because "
                    f"'config_groups' parameter with values {params['config_groups']} will be used.",
                )
                params["configs"] = configs
        return super().modify_param_values(params)

    def workflow_requires(self):
        reqs = super().workflow_requires()
        for config_group in self.config_groups:
            config_repr = self.configs_str(config_group)
            reqs[f"rebinned_datacards__{config_repr}"] = self.reqs.ModifyDatacardsFlatRebin.req(
                self,
                configs=config_group,
                # config_insts=[config_inst for config_inst in self.config_insts if config_inst.name in config_group],
            )

        return reqs

    def requires(self):
        reqs = {}
        for config_group in self.config_groups:
            config_repr = self.configs_str(config_group)
            reqs[f"rebinned_datacards__{config_repr}"] = self.reqs.ModifyDatacardsFlatRebin.req(
                self,
                configs=config_group,
                # config_insts=[config_inst for config_inst in self.config_insts if config_inst.name in config_group],
            )
        return reqs

    @property
    def config_groups_str(self):
        return [self.configs_str(configs) for configs in self.config_groups]

    def store_parts(self):
        parts = super().store_parts()
        cards_repr = "__".join(self.config_groups_str)
        if self.cards_version:
            cards_repr += f"__{self.cards_version}"
        parts.insert_before("version", "config_group", cards_repr)
        return parts

    def output(self):
        # TODO: should add configs_str to output path
        output = {
            "Run": self.target(self.run_script_name),
        }
        self.cards_target = self.target("datacards", dir=True)
        if self.recreate_datacards:
            output["datacards"] = self.cards_target
        return output

    def run(self):
        inputs = self.input()
        output = self.output()

        card_fns = []
        for key, value in inputs.items():
            recreate_cards = self.recreate_datacards or not self.cards_target.exists()
            if not key.startswith("rebinned_datacards__"):
                continue
            for target in value.collection.targets.values():
                card_fns.append(target["card"].basename)
                if recreate_cards:
                    logger.info(
                        f"Copying datacard for target {target['card'].basename} to {self.cards_target.abspath}")
                    target["card"].copy_to(self.cards_target)
                    target["shapes"].copy_to(self.cards_target)
                    target["inspection"].copy_to(self.cards_target)

        # string that represents the version of datacards
        identifier_list = [*self.config_groups_str,
                           self.inference_model_cls.__str__()]
        if self.cards_version:
            identifier_list.append(self.cards_version)
        identifier = "__".join(identifier_list)

        # TODO: copy datacards to this output

        # TODO: merge collections from different config groups
        # TODO: get rid of inference_model_inst usage
        # get the datacard names from the inputs
        # collection = inputs["rebinned_datacards"]["collection"]
        # cards_path = {collection[key]["card"].dirname for key in collection.keys()}
        # if len(cards_path) != 1:
        #     raise Exception("Expected only one datacard path")
        # cards_path = cards_path.pop()

        cards_path = self.cards_target.abspath
        pruned_cards_path = f"{cards_path}/pruned"

        # card_fns = [collection[key]["card"].basename for key in collection.keys()]

        # get the category names from the inference models
        cat_names = self.inference_model_cls.config_categories
        cat_names = [
            f"{cat_name}__{year}" for year in self.config_groups_str for cat_name in cat_names]

        # combine category names with card fn to a single string
        datacards = ",".join(
            [f"{cat_name}=$CARDS_PATH/{card_fn}" for cat_name, card_fn in zip(cat_names, card_fns)])
        datacards_per_year = ":".join([",".join(
            [card for card in datacards.split(",") if year in card],
        ) for year in self.config_groups_str])

        # # name of the output root file that contains the Pre+Postfit shapes
        # output_file = ""

        # prepare the base command to set the environment variables
        base_cmd = f"export BASE_CARDS_PATH={cards_path}" + \
            "\n" + f"export CARDS_PATH={pruned_cards_path}" + "\n"
        export_and_prepare_cards_cmd = base_cmd
        cmd_dict = {}

        # prepare strings for campaign name and card names
        if len(self.configs) == 4:
            campaign = "run3"
        elif len(self.configs) == 1:
            campaign = self.configs[0]
        else:
            # lumi = sum([config_inst.x.luminosity.get("nominal") for config_inst in self.config_insts]) * 0.001
            campaign = "run3"

        def is_signal_region(cat_name):
            return (
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
        # cards_vbf = ",".join([
        #     f"{cat_name}=$CARDS_PATH/{card_fn}" for cat_name, card_fn in zip(cat_names, card_fns) if "ggf" not in cat_name  # noqa: E501
        # ])
        # cards_ggf = ",".join([
        #     f"{cat_name}=$CARDS_PATH/{card_fn}" for cat_name, card_fn in zip(cat_names, card_fns) if "vbf" not in cat_name  # noqa: E501
        # ])
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

        # NOTE: we could replace e.g. {datacards} with $CARDS in the commands below
        cards_dict = {
            "CARDS": datacards,
            "CARDSB": cards_1b,
            "CARDSBB": cards_2b,
            "CARDSBoost": cards_boosted,
        }
        export_cards_combinations = "\n".join(
            f"export {key}={value}" for key, value in cards_dict.items())
        cmd_dict["export"] = "\n".join([base_cmd, export_cards_combinations])

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

        # run pruning helper on cards
        prepare_cards = []
        for card_fn in card_fns:
            cmd = f"python3 prepare_cards.py $BASE_CARDS_PATH/{card_fn}"
            export_and_prepare_cards_cmd += cmd + "\n"
            prepare_cards.append(cmd)
        # cmd_dict["prepare_cards"] = "\n".joiyn(prepare_cards)
        # prepare first card to avoid runtime issues of folder creation
        cmd_dict["prepare_cards"] = prepare_cards[0] + \
            "\n" + " & ".join(prepare_cards)
        cmd_dict["prepare_cards"] += " & wait"
        print("\n\n")
        print(export_and_prepare_cards_cmd)

        base_cmd = f"export CARDS_PATH={pruned_cards_path}" + "\n"

        # print(base_cmd)
        # for card, _ident in zip(card_fns, identifier):
        #     cmd = f"ValidateDatacard.py $CARDS_PATH/{card} --jsonFile $CARDS_PATH//validation_{_ident}.json"
        #     print(cmd)
        # print("\n\n")

        # fetch card combination
        cmd = (
            f"law run CombineDatacards --version {identifier} --datacards {datacards} "
        )
        cmd_dict["combine_cards"] = cmd

        # creating upper limits for kl=1
        cmd = (
            f"law run PlotUpperLimitsAtPoint --version {identifier} --campaign {campaign} "
            f"--multi-datacards {multi_datacards} "
            f"--datacard-names {multi_datacard_names} "
            # f"--UpperLimits-workflow htcondor "
            f"--workers 10 "
        )
        if self.partially_unblinded or self.fully_unblinded:
            cmd += "--unblinded True "
        if not self.partially_unblinded:
            print(base_cmd + cmd, "\n\n")
            cmd_dict["LimitsPerCategory"] = cmd
            cmd_dict["qqHH_LimitsPerCategory"] = cmd + "--pois r_qqhh "

        # creating upper limits for kl=1 with clean signal regions only
        cmd = (
            f"law run PlotUpperLimitsAtPoint --version {identifier} --campaign {campaign} "
            f"--multi-datacards {cards_1b}:{cards_2b}:{cards_boosted}:{datacards} "
            f"--datacard-names 1b,2b,Boosted,Combined "
            # f"--UpperLimits-workflow htcondor "
            f"--workers 10 "
        )
        if self.partially_unblinded or self.fully_unblinded:
            cmd += "--unblinded True "
        if not self.partially_unblinded:
            print(base_cmd + cmd, "\n\n")
            cmd_dict["pointlimits"] = cmd
            cmd_dict["qqhh_pointlimits"] = cmd + "--pois r_qqhh "

        # datacards per config group
        if len(self.config_groups) > 1:
            cmd = (
                f"law run PlotUpperLimitsAtPoint --version {identifier} --campaign {campaign} "
                f"--multi-datacards {datacards_per_year}:{datacards} "
                f"--datacard-names {','.join(self.config_groups_str)},combined "
                # f"--UpperLimits-workflow htcondor "
                f"--workers 10 "
            )
            if self.partially_unblinded or self.fully_unblinded:
                cmd += "--unblinded True "

            if not self.partially_unblinded:
                print(base_cmd + cmd, "\n\n")
                cmd_dict["LimitsPerCampaign"] = cmd

        # # creating upper limits for kl=1
        # cmd = (
        #     f"law run PlotUpperLimitsAtPoint --version {identifier} --campaign {campaign} "
        #     f"--multi-datacards {datacards} "
        #     f"--datacard-names {identifier}"
        # )
        # if not self.partially_unblinded:
        #     print(base_cmd + cmd, "\n\n")

        # creating kl scan
        scan_kl = "kl,-20,25,46"
        # scan_kl = "kl,-15,20,36"
        # scan_c2v = "C2V,-4,6,41"
        scan_c2v = "C2V,-2,4,25"
        scan_kt = "kt,-3,3,31"
        scan_r = "r,-30,30,61"

        cmd = (
            f"law run PlotUpperLimits --version {identifier} --campaign {campaign} --datacards {datacards} "
            f"--xsec fb --y-log --scan-parameters {scan_kl} --UpperLimits-workflow htcondor "
            f"--frozen-parameters {self.frozen_for_scans} "
        )
        if self.partially_unblinded or self.fully_unblinded:
            cmd += "--unblinded True "
        if not self.partially_unblinded:
            print(base_cmd + cmd, "\n\n")
            cmd_dict["limits_kl"] = cmd
            cmd_dict["qqhh_limits_kl"] = cmd + "--pois r_qqhh "
            cmd_dict["limits_c2v"] = cmd.replace(scan_kl, scan_c2v)
            cmd_dict["qqhh_limits_c2v"] = cmd.replace(
                scan_kl, scan_c2v) + "--pois r_qqhh "

        cmd = (
            f"law run PlotMultipleUpperLimits --version {identifier} --campaign {campaign} "
            f"--multi-datacards {cards_1b}:{cards_2b}:{cards_boosted}:{datacards} "
            f"--datacard-names 1b,2b,Boosted,Combined "
            f"--xsec fb --y-log --scan-parameters {scan_c2v} --UpperLimits-workflow htcondor "
            f"--workers 4 "
            f"--frozen-parameters {self.frozen_for_scans} "
        )
        if self.partially_unblinded or self.fully_unblinded:
            cmd += "--unblinded True "
        if not self.partially_unblinded:
            print(base_cmd + cmd, "\n\n")
            cmd_dict["multilimits_c2v"] = cmd
            cmd_dict["qqhh_multilimits_c2v"] = cmd + "--pois r_qqhh "
            cmd_dict["multilimits_kl"] = cmd.replace(scan_c2v, scan_kl)
            cmd_dict["qqhh_multilimits_kl"] = cmd.replace(
                scan_c2v, scan_kl) + "--pois r_qqhh "

        # Likelihood scans
        cmd = (
            f"law run PlotMultipleLikelihoodScans --version {identifier} --campaign {campaign} "
            f"--multi-datacards $CARDS:$CARDS --datacard-names Observed,Expected --unblinded True,False "
            f"--LikelihoodScan-{{workflow=htcondor,retries=1}} --workers 4 "
        )
        for poi, scan_params in (
            ("kl", scan_kl), ("C2V", scan_c2v), ("r", scan_r),
            ("kl_C2V", f"{scan_kl}:{scan_c2v}"),
            ("kl_kt", f"{scan_kl}:{scan_kt}"),
        ):
            cmd_poi = cmd + \
                f" --scan-parameters {scan_params} --poi {poi.replace('_', ',')} "
            print(base_cmd + cmd_poi, "\n\n")
            cmd_dict[f"likelihood_{poi.lower()}"] = cmd_poi

        # running FitDiagnostics for Pre+Postfit plots
        cmd = (
            f"law run FitDiagnostics --version {identifier} --datacards {datacards} "
            f"--skip-b-only"
        )
        cmd_dict["FitDiagnostics"] = cmd

        # running Pulls and Impacts
        # impacts_order = "--order-by-impacts"
        impacts_order = ""
        cmd = (
            f"law run PlotPullsAndImpacts --version {identifier} --campaign {campaign} --datacards {datacards} "
            f"--parameters-per-page 80 --mc-stats --retry-no-analytic "
            f"{impacts_order} "
        )
        pulls_and_impacts_params = "workflow=htcondor,retries=1"
        custom_args = "--robustFit 1"
        if self.partially_unblinded:
            custom_args += " --rMin -350 --rMax 350"
        else:
            custom_args += " --rMin -32 --rMax 32"
        if self.partially_unblinded or self.fully_unblinded:
            cmd += "--unblinded True --show-best-fit True "
        elif self.unblinded_step1:
            cmd += "--unblinded True --show-best-fit False "
            # cmd += "--unblinded True "  # do this only after all unblinding steps!
        retry_pulls_and_impacts_params = pulls_and_impacts_params + f",custom-args='{custom_args} --X-rtd MINIMIZER_no_analytic'"  # noqa: E501
        pulls_and_impacts_params += f",custom-args='{custom_args}'"

        cmd += f"--PullsAndImpacts-{{{pulls_and_impacts_params}}} "

        print(base_cmd + cmd, "\n\n")
        cmd_dict["impacts"] = cmd
        cmd_dict["impacts_retry"] = cmd.replace(
            pulls_and_impacts_params, retry_pulls_and_impacts_params)

        # MultiplePullsAndImpacts
        cmd = (
            f"law run PlotMultiplePullsAndImpacts --version {identifier} --campaign {campaign} "
            f"--multi-datacards {cards_1b}:{cards_2b}:{cards_boosted}:{datacards} "
            f"--datacard-names 1b,2b,Boosted,Combined "
            # f"--parameters-per-page 80 --mc-stats "
            "--retry-no-analytic "
            f"{impacts_order} "
            f"--workers 4 "
        )
        cmd += f"--PullsAndImpacts-{{{pulls_and_impacts_params}}} "
        cmd_dict["multiimpacts"] = cmd
        cmd_dict["multiimpacts_retry"] = cmd.replace(
            pulls_and_impacts_params, retry_pulls_and_impacts_params)

        # running GoodnessOfFit
        # freeze_signal = "--freezeParameters r,r_gghh,r_qqhh,kl,kt,CV,C2V --setParameters r=0.0,r_gghh=0.0,r_qqhh=0.0,kl=1.0,kt=1.0,CV=1.0,C2V=1.0"  # noqa: E501
        parameter_values = "r=0.0:r_gghh=0.0:r_qqhh=0.0:kl=1.0:kt=1.0:CV=1.0:C2V=1.0"
        if self.partially_unblinded or self.unblinded_step1 or self.fully_unblinded:
            snapshot_custom_args = custom_args
            # if self.partially_unblinded:
            #     snapshot_custom_args = " ".join([custom_args, freeze_signal])
            cmd = (
                f"law run PlotMultipleGoodnessOfFits --version {identifier} --campaign {campaign} "
                f"--multi-datacards {cards_1b}:{cards_2b}:{cards_boosted}:{datacards} "
                f"--datacard-names 1b,2b,Boosted,Combined "
                f"--toys 1000 --toys-per-branch 20 --frequentist-toys --use-snapshot "
            )
            cmd += f"--Snapshot-{{unblinded,custom-args='{snapshot_custom_args}'}} "
            if self.partially_unblinded:
                gof_custom_args = " --rMin -350 --rMax 350"
                cmd += f"--GoodnessOfFit-{{workflow=htcondor,custom-args='{gof_custom_args}'}} "
                cmd += f"--parameter-values {parameter_values} "
            else:
                cmd += "--GoodnessOfFit-workflow htcondor "
            cmd += "--workers 4 "

            cmd_dict["gof"] = cmd

        # running PreAndPostfitShapes for Pre+Postfit plots
        cmd = (
            f"law run MergePreAndPostFitShapes --version {identifier} "
            f"--datacards {datacards} "
            f"--apply-fit-datacards {datacards.replace(',', ':')} "
            "--PreAndPostFitShapes-workflow htcondor --workers 5 "
        )
        if self.partially_unblinded or self.unblinded_step1 and not self.fully_unblinded:
            # cmd += f"--FitParameters-custom-args='{freeze_signal}' "
            cmd += f"--parameter-values {parameter_values} "
        if self.partially_unblinded or self.unblinded_step1 or self.fully_unblinded:
            cmd += "--unblinded True "
        print(base_cmd + cmd, "\n\n")
        cmd_dict["postfitshapes"] = cmd
        cmd += "--prefit"
        cmd_dict["prefitshapes"] = cmd

        if self.rerun:
            for key, cmd in cmd_dict.items():
                if key in ("prepare_cards", "export"):
                    continue
                cmd_dict[key] = cmd + " --remove-output 0,a,y"

        # dump all the commands to one output file
        functions_cmd = "\n\n".join([
            f"{key}() {{\n    {cmd_dict[key]}\n}}" for key in cmd_dict.keys() if key != "export"
        ])
        if missing_keys := (set(self.requested_keys) - set(cmd_dict.keys())):
            logger.warning(
                f"Requested keys {missing_keys} are not available in the command dictionary.")
            self.requested_keys = [
                key for key in self.requested_keys if key not in missing_keys]
        run_script = self.create_run_script(
            identifier=identifier,
            export_cmd=cmd_dict["export"],
            functions_cmd=functions_cmd,
        )
        output["Run"].dump(run_script, formatter="text")

        print("# combined task calls")
        print(f"bash {output['Run'].abspath}")

    def create_run_script(self, identifier, export_cmd, functions_cmd):
        # TODO: I would like to rewrite the script such that each command of the cmd_dict is in one function
        def fetch_func(cmd, multi_thread: bool = False):
            cmd = cmd.replace(
                "law run", f"run_and_fetch_cmd {identifier} law run")
            if multi_thread:
                cmd = cmd.replace("\n", " & \n") + " & wait"
            return cmd
        newline_with_indent = "\n    "

        run_script = f"""#!/bin/bash
# AUTO-GENERATED FILE! DO NOT EDIT!

dry_run=${{dry_run:-false}}
retry_impacts=${{retry_impacts:-false}}

{export_cmd}

mkdir -p $DHI_DATA/fetched_plots/{identifier} && cd $DHI_DATA/fetched_plots/{identifier}

run_and_fetch_cmd() {{
    local folder="$1"
    cd "$DHI_DATA/fetched_plots/$folder" || exit 1
    if [[ "$dry_run" != "false" ]]; then
        echo "[DRY-RUN] ${{*:2}}"
        echo "[DRY-RUN] ${{*:2}} --fetch-output 0,a"
    else
        echo "[RUNNING] ${{*:2}}"
        "${{@:2}}"
        "${{@:2}}" --fetch-output 0,a
    fi
}}

{fetch_func(functions_cmd)}

run_all() {{
    {newline_with_indent.join([key for key in self.requested_keys])}
}}

# Only run if script is executed, not sourced
if [[ "${{BASH_SOURCE[0]}}" == "${{0}}" ]]; then
    run_all
fi
"""
        return run_script


class MultiDatacards(
    HBWTask,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ReducerClassMixin,
    ProducerClassesMixin,
    MLModelsMixin,
    HistProducerClassMixin,
    HistHookMixin,
):
    resolution_task_cls = MergeHistograms
    single_config = False

    sandbox = dev_sandbox(law.config.get(
        "analysis", "default_columnar_sandbox"))

    inference_models = law.CSVParameter(
        default=[],
        description="List of inference model classes to produce datacards for.",
        significant=False,
    )

    # upstream requirements
    reqs = Requirements(
        # RemoteWorkflow.reqs,
        CreateDatacards=CreateDatacards,
    )

    @property
    def inference_model_clses(self):
        return [InferenceModel.get_cls(inference_model) for inference_model in self.inference_models]

    @classmethod
    def resolve_instances(cls, params: dict[str, Any], shifts: TaskShifts) -> dict[str, Any]:
        for inference_model in params["inference_models"]:
            _params = params.copy()
            _params["inference_model"] = inference_model
            cls.reqs.CreateDatacards.resolve_instances(_params, shifts)
        return params

    def requires(self):
        reqs = {
            "datacards": [self.reqs.CreateDatacards.req_different_branching(
                self,
                branch=0,
                configs=self.configs,
                inference_model=inference_model,
            ) for inference_model in self.inference_models],
        }
        return reqs

    def output(self):
        output = {
            "calls": self.target("MultiDatacards_Calls.txt"),
        }
        return output

    def run(self):
        inputs = self.input()
        output = self.output()

        apply_fit_datacards = []
        variables = []
        for _input, inference_model_cls in zip(inputs["datacards"], self.inference_model_clses):
            datacards = []
            variable = ""
            for category, target in _input.targets.items():
                if not variable:
                    variable = inference_model_cls.config_variable(
                        inference_model_cls, category)
                elif variable != inference_model_cls.config_variable(inference_model_cls, category):
                    raise ValueError(
                        f"Variable {variable} from category {category} does not agree with previous "
                        f"variable {inference_model_cls.config_variable(inference_model_cls, category)}",
                    )
                abspath = target["card"].abspath
                datacards.append(f"{category}={abspath}")
            variables.append(variable)
            apply_fit_datacards.append(",".join(datacards))

        cmd = (
            "law run MergePreAndPostFitShapes "
            "--FitParameters-workflow local "
            "--PreAndPostFitShapes-workflow htcondor --workers 5 --unblinded True --prefit "
            "--version shapes --datacards $CARDS "
            f"--apply-fit-datacards {':'.join(apply_fit_datacards)} "
            f"--fit-datacard-variables {','.join(variables)} "
        )
        print("\n\n" + cmd + "\n\n")

        output["calls"].dump(cmd, formatter="text")

        return
# ---------------------------------------------------------------------------------------------------------------------
# CWS inference helpers + tasks (paste into hbw/tasks/inference.py, below PrepareInferenceTaskCalls).
# Reuses law, luigi, HBWInferenceModelBase, InferenceModelClassMixin, ModifyDatacardsFlatRebin, Requirements,
# TaskShifts, HBWTask and logger from that module.
#
#   1. hbw.PrepareCWSDatacards  (one law call per inference model)  -> datacards/, manifest.json
#   2. hbw.PrepareCWSTaskCalls  (combine, strings only)             -> inference.yaml, inference_dummy.yaml,
#                                                                       cards/ (rename/remove rules), result_plots/,
#                                                                       run.sh
# prepare_cws.sh runs 1. per model and then 2.
#
# run.sh has two card modes (card_mode=standard|dummy, default from --card-mode):
#   standard  cws combines the pruned per-category cards itself (inference.yaml)
#   dummy     the combined cards per group (3b, 4b, combined) are turned into dummy cards for the multiH model,
#             checked and scaled (inference_dummy.yaml points to the scaled dummy cards)
# In both modes prepare_cards.py already removes the non-SM HHH signals and renames the rest to the multiH names
# (remove_processes_list and rename_processes_dict in prepare_cards.py).
# ---------------------------------------------------------------------------------------------------------------------


STAGE_ASIMOV = "asimov"
STAGE_PART_UNBLINDED = "part_unblinded"
STAGES = (STAGE_ASIMOV, STAGE_PART_UNBLINDED)
CARD_MODES = ("standard", "dummy")


# ---------------------------------------------------------------------------------------------------------------------
# pure helpers (no law/columnflow dependency, so they can be tested standalone)
# ---------------------------------------------------------------------------------------------------------------------

def channel_name(category: str, config_group: str, label: str, taken: list[str]) -> str:
    """
    Name of a channel (= bin name in the combined datacard). combine requires bin names to start with a letter, so
    the category comes first (e.g. sr__3b__ml_tth__c24v15_c25v15); the model label is only appended when two models
    share a category name.
    """
    name = f"{category}__{config_group}"
    if not name[0].isalpha():
        name = f"ch_{name}"
    if name in taken:
        name = f"{name}__{label}"
    return name


def _yaml(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False, default_flow_style=False, width=10_000)


def build_task_configs(datacards: str) -> list[tuple[str, str, str, dict]]:
    """
    Returns the cws task configs as a list of (name, stage, comment, config).
    *datacards* is the colon-separated list of datacard groups, e.g. "3b:4b:combined".
    Edit this function to add, remove or change tasks; run.sh picks up the stages automatically.
    """
    # background-only setup: signal strength frozen to 0, couplings at their SM values
    freeze_signal = (
        "--freezeParameters r,k4,kl,kt,CV,C2V "
        "--setParameters r=0.0,k4=1.0,kl=1.0,CV=1.0,kt=1.0,C2V=1.0"
    )
    pulls_custom_args = f"--robustFit 1 --rMin -10000 --rMax 100000 {freeze_signal}"

    def limit_at_point(output_name: str, poi: str, xsec: str | None = None) -> dict:
        cfg = {
            "task": "PlotUpperLimitsAtPoint",
            "output_name": output_name,
            "datacards": datacards,
            "pois": poi,
        }
        if xsec:
            cfg["xsec"] = xsec
        cfg.update({"style": "cms", "unblinded": False})
        return cfg

    def limit_scan(output_name: str, poi: str, param: str, min_: float, max_: float, points: int) -> dict:
        return {
            "task": "PlotUpperLimits",
            "output_name": output_name,
            "datacards": datacards,
            "pois": poi,
            "scan_parameters": [{"name": param, "min": min_, "max": max_, "points": points}],
            "xsec": "fb",
            "y_log": True,
            "style": "cms",
            "unblinded": False,
        }

    def likelihood_scan(output_name: str, poi: str, param: str, min_: float, max_: float, points: int) -> dict:
        return {
            "task": "PlotMultipleLikelihoodScans",
            "output_name": output_name,
            "datacards": datacards,
            "pois": poi,
            "scan_parameters": [{"name": param, "min": min_, "max": max_, "points": points}],
            "style": "cms",
            "unblinded": False,
        }

    def pulls_impacts(output_name: str, mc_stats: bool, unblinded: bool) -> dict:
        return {
            "task": "PlotPullsAndImpacts",
            "output_name": output_name,
            "datacards": datacards,
            "pois": "r_gghhh",
            "show_parameters": "kl,k4",
            "mc_stats": mc_stats,
            "show_best_fit": False,
            "order_by_impact": True,
            "parameters_per_page": 40,
            "method": "robust",
            "retry_no_analytic": True,
            "custom_args": pulls_custom_args,
            "collapse_impacts_and_pulls": True,
            "frozen_parameters": "kl,kt,CV,C2V",
            "skip_parameters": "kt,CV,C2V",
            "style": "cms",
            "unblinded": unblinded,
            "workflow": "htcondor",
        }

    A, U = STAGE_ASIMOV, STAGE_PART_UNBLINDED
    return [
        # ---- stage 1: Asimov -------------------------------------------------------------------------------------
        (
            "create_workspace", A,
            "build the workspaces first (fails fast if the cards and the physics model do not fit together)",
            {"task": "CreateWorkspace", "datacards": "combined"},
        ),
        (
            "limits_at_point", A,
            "expected (Asimov) upper limit at the SM point on the HHH signal strength (ttHH stays at r_tthh = 1)",
            limit_at_point("asimov_limit", "r_gghhh"),
        ),
        (
            "limits_at_point_fb", A,
            "the same limit as a cross section, in fb",
            limit_at_point("asimov_limit_xsec", "r_gghhh", xsec="fb"),
        ),
        (
            "limits_at_point_r_gghh", A,
            "expected (Asimov) upper limit at the SM point on the HH signal strength r_gghh",
            limit_at_point("asimov_limit_r_gghh", "r_gghh"),
        ),
        (
            "limits_at_point_r_gghh_fb", A,
            "the same r_gghh limit as a cross section, in fb",
            limit_at_point("asimov_limit_r_gghh_xsec", "r_gghh", xsec="fb"),
        ),
        (
            "limits_k4", A,
            "expected upper limit on the HHH cross section as a function of k4 (steps of 10)",
            limit_scan("asimov_limits_k4", "r_gghhh", "k4", -300.0, 300.0, 61),
        ),
        (
            "limits_kl", A,
            "expected upper limit on the HHH cross section as a function of kl (steps of 1)",
            limit_scan("asimov_limits_kl", "r_gghh", "kl", -30.0, 30.0, 61),
        ),
        (
            "likelihood_k4", A,
            "expected upper limit on the HHH cross section as a function of k4 (steps of 10)",
            likelihood_scan("asimov_likelihood_k4", "k4",
                            "k4", -300.0, 300.0, 61),
        ),
        (
            "likelihood_kl", A,
            "expected upper limit on the HHH cross section as a function of kl (steps of 1)",
            likelihood_scan("asimov_likelihood_kl", "kl",
                            "kl", -30.0, 30.0, 61),
        ),
        (
            "asimov_pulls_impact_no_mc_stats", A,
            "Asimov pulls and impacts, without MC-stat nuisances",
            pulls_impacts("asimov_pulls", mc_stats=False, unblinded=False),
        ),
        (
            "asimov_pulls_impact", A,
            "Asimov pulls and impacts, with MC-stat nuisances",
            pulls_impacts("asimov_pulls_mc_stats",
                          mc_stats=True, unblinded=False),
        ),
        (
            "nll2d_k4_kl", A,
            "2D likelihood scan in k4 vs kl (49 x 49 points -> needs htcondor)",
            {
                "task": "PlotLikelihoodScan",
                "output_name": "nll2d_k4_kl",
                "datacards": datacards,
                "pois": "k4,kl",
                "scan_parameters": [
                    {"name": "k4", "min": -300.0, "max": 300.0, "points": 49},
                    {"name": "kl", "min": -30.0, "max": 30.0, "points": 49},
                ],
                # sigma contours instead of the default 0.68,0.95
                "show_significances": [1, 2],
                "interpolate_nans": True,  # fill the points where a fit did not converge
                "smooth_contour": "k5a,2",  # two 5x5 passes before the contours are found
                "show_sm": True,
                "style": "cms",
                "unblinded": False,
                "workflow": "htcondor",
            },
        ),
        # ---- stage 2: partially unblinded (background-only fits on data) ------------------------------------------
        (
            "part_unblind_pulls_impact_no_mc_stats", U,
            "partially unblinded pulls and impacts (signal frozen to 0), without MC-stat nuisances",
            pulls_impacts("part_unblind_pulls",
                          mc_stats=False, unblinded=True),
        ),
        (
            "part_unblind_pulls_impact", U,
            "partially unblinded pulls and impacts (signal frozen to 0), with MC-stat nuisances",
            pulls_impacts("part_unblind_pulls_mc_stats",
                          mc_stats=True, unblinded=True),
        ),
    ]


def build_yaml_text(
    parameters: dict,
    channels: list[tuple[str, str, dict]],
    groups: dict[str, dict],
    tasks: list[tuple[str, str, str, dict]],
    tasks_key: str = "tasks",
    header: str = "",
) -> str:
    """
    *channels*: list of (section label, channel key, {"path": ..., "name": ...}).
    *groups*: {group key: {"cards": "ch1,ch2", "name": ...}}.
    """
    def indent(text):
        return textwrap.indent(text, "  ")

    lines = [header.rstrip(), "", "parameters:", indent(
        _yaml(parameters)).rstrip(), "", "datacards:"]

    current_section = None
    for section, key, channel in channels:
        if section != current_section:
            lines.append(f"  # --- {section} ---")
            current_section = section
        lines.append(indent(_yaml({key: channel})).rstrip())
    lines.append("")
    lines.append(
        "  # --- datacard groups (use these in the 'datacards' field of the tasks) ---")
    lines.append(indent(_yaml(groups)).rstrip())
    lines.append("")

    lines.append(f"{tasks_key}:")
    current_stage = None
    for name, stage, comment, cfg in tasks:
        if stage != current_stage:
            lines.append(f"  # ===== stage: {stage} =====")
            current_stage = stage
        lines.append(f"  # {comment}")
        lines.append(indent(_yaml({name: cfg})).rstrip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


RUN_SCRIPT_TEMPLATE = r"""#!/usr/bin/env bash
# AUTO-GENERATED by __TASK__ -- DO NOT EDIT, rerun the law task instead.
#
# Runs the inference in stages:
#   asimov          expected (Asimov) limits (r_gghhh, r_gghh, vs k4, vs kl), pulls/impacts, 2D k4-kl scan
#   part_unblinded  pulls/impacts from a partially unblinded (background-only) fit
#   all             both stages, in that order (default)
# Before every stage the cards are prepared (each step is skipped when its outputs exist):
#   prepare_cards   prepare_cards.py: pruned per-category cards, non-SM HHH signals removed and
#                   the remaining signals renamed to the multiH model names
#   combine_cards   combineCards.py per group (__GROUPS__)        -> $COMBINED_DIR
#   dummy_cards     only for card_mode=dummy: dummy card -> check -> scale  -> $DUMMY_DIR/<group>/
#
# usage:
#   bash run.sh [asimov|part_unblinded|all]           # default: all
#   bash run.sh limits_kl nll2d_k4_kl                 # single task configs
#   bash run.sh prepare_cards|combine_cards|dummy_cards
#   card_mode=dummy bash run.sh asimov                # run on the dummy cards (inference_dummy.yaml)
#   source run.sh                                     # only define the variables and functions
#
# switches (environment variables):
#   card_mode=standard|dummy  which cards / yaml to use                      (default: __CARD_MODE__)
#   dry_run=true              only print the commands
#   rerun=true                remove existing outputs before running (law --remove-output 0,a,y)
#   fetch_output=false        do not copy the plots into $RESULT_PLOTS after each task
#   force_prepare=true        redo prepare/combine/dummy cards even if they exist (prepare_cards.py --force)
#   strip_rate_params=false   keep the rateParam lines when making the dummy cards  (default: true)
#   ASIMOV_WORKERS=N          local law workers in the asimov stage          (default: __ASIMOV_WORKERS__)
#   UNBLINDED_WORKERS=N       local law workers in the part_unblinded stage  (default: __UNBLINDED_WORKERS__)
#   MULTIH_DIR, SCALE_SCRIPT, PREPARE_CARDS_SCRIPT   tool locations (defaults below)

card_mode=${card_mode:-__CARD_MODE__}
dry_run=${dry_run:-false}
rerun=${rerun:-false}
fetch_output=${fetch_output:-true}
force_prepare=${force_prepare:-false}
strip_rate_params=${strip_rate_params:-true}
ASIMOV_WORKERS=${ASIMOV_WORKERS:-__ASIMOV_WORKERS__}
UNBLINDED_WORKERS=${UNBLINDED_WORKERS:-__UNBLINDED_WORKERS__}

__EXPORTS__

case "$card_mode" in
    standard) export CFG="$CFG_STANDARD" ;;
    dummy) export CFG="$CFG_DUMMY" ;;
    *) echo "[ERROR] card_mode must be 'standard' or 'dummy', got '$card_mode'"; exit 2 ;;
esac

GROUPS_=(__GROUPS__)
ASIMOV_TASKS=(__ASIMOV_TASKS__)
PART_UNBLINDED_TASKS=(__PART_UNBLINDED_TASKS__)
declare -A TASK_STAGE=(__TASK_STAGE__)

FAILED=()
_cards_ready=false
_combined_ready=false
_dummy_ready=false

_run() {
    if [[ "$dry_run" != "false" ]]; then
        echo "[DRY-RUN] $*"
    else
        echo "[RUNNING] $*"
        "$@"
    fi
}

# _run_in <dir> <cmd...>: run a command inside a directory
_run_in() {
    local dir="$1"
    shift
    if [[ "$dry_run" != "false" ]]; then
        echo "[DRY-RUN] (cd $dir) $*"
    else
        echo "[RUNNING] (cd $dir) $*"
        ( cd "$dir" && "$@" )
    fi
}

# _all_exist <file...>: true when all files exist and force_prepare is not set
_all_exist() {
    [[ "$force_prepare" == "true" ]] && return 1
    local f
    for f in "$@"; do
        [[ -f "$f" ]] || return 1
    done
    return 0
}

# ---- step 1: pruned per-category cards ------------------------------------------------------------------------------

prepare_cards() {
    [[ "$_cards_ready" == "true" ]] && return 0
    if _all_exist __PRUNED_CARDS__; then
        echo "[SKIP] pruned datacards exist"
        _cards_ready=true
        return 0
    fi

    local rc=0 pid pids=()
    local prep_args=()
    [[ "$force_prepare" == "true" ]] && prep_args+=(--force)
    # the first card of each model runs alone so that its output folder exists before the parallel jobs start
__PREPARE_CARDS__
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
    [[ "$rc" -eq 0 ]] && _cards_ready=true
    return "$rc"
}

# ---- step 2: one combined card per group ----------------------------------------------------------------------------

# combine_group <group> <name=card ...>
combine_group() {
    local group="$1"
    shift
    local out="$COMBINED_DIR/datacard_${group}.txt"
    if [[ "$dry_run" != "false" ]]; then
        echo "[DRY-RUN] (cd $COMBINED_DIR) combineCards.py $* > $out"
        return 0
    fi
    mkdir -p "$COMBINED_DIR"
    echo "[RUNNING] combineCards.py ($# cards) > $out"
    ( cd "$COMBINED_DIR" && combineCards.py "$@" > "$out.tmp" ) && mv "$out.tmp" "$out"
}

combine_cards() {
    [[ "$_combined_ready" == "true" ]] && return 0
    prepare_cards || return 1
    if _all_exist __COMBINED_CARDS__; then
        echo "[SKIP] combined datacards exist in $COMBINED_DIR"
        _combined_ready=true
        return 0
    fi
__COMBINE_CARDS__
    _combined_ready=true
}

# ---- step 3: dummy cards for the multiH model -----------------------------------------------------------------------   # noqa: E501

# dummy_group <group>
dummy_group() {
    local group="$1"
    local card="datacard_${group}.txt"
    local work="$DUMMY_DIR/$group"
    local dummy="datacard_dummy_${group}.txt"
    local scaled="datacard_dummy_${group}_scaled.txt"

    echo ""
    echo "----- [dummy] $group -----"
    [[ "$dry_run" == "false" ]] && mkdir -p "$work/input" "$work/dummy"

    # 0. input: the combined card, optionally without rateParam lines
    if [[ "$strip_rate_params" == "true" ]]; then
        if [[ "$dry_run" != "false" ]]; then
            echo "[DRY-RUN] grep -vE '^[^#[:space:]]+[[:space:]]+rateParam[[:space:]]' $COMBINED_DIR/$card > $work/input/$card"  # noqa: E501
        else
            grep -vE '^[^#[:space:]]+[[:space:]]+rateParam[[:space:]]' "$COMBINED_DIR/$card" > "$work/input/$card"
            echo "[INFO] removed $(grep -cE '^[^#[:space:]]+[[:space:]]+rateParam[[:space:]]' "$COMBINED_DIR/$card") rateParam lines"  # noqa: E501
        fi
    else
        _run cp "$COMBINED_DIR/$card" "$work/input/$card" || return 1
    fi

    # (non-SM removal and renaming already happened in prepare_cards.py)
    # 1. dummy card for the physics model
    _run_in "$MULTIH_DIR" python3 scripts/make_dummy_card.py \
        --model "$DUMMY_MODEL" \
        --input-card "$work/input/$card" \
        --output-dir "$work/dummy" \
        --energy "$DUMMY_ENERGY" \
        --output-name "$dummy" || return 1
    # 2. check it against the physics model
    _run_in "$(dirname "$MULTIH_DIR")" python3 multih_models/scripts/check_card.py \
        "$work/dummy/$dummy" "$PHYSICS_MODEL" || return 1
    # 3. scale the dummy templates
    _run_in "$work/dummy" python3 "$SCALE_SCRIPT" "$dummy" -o "$scaled" || return 1
}

dummy_cards() {
    [[ "$_dummy_ready" == "true" ]] && return 0
    combine_cards || return 1
    if _all_exist __DUMMY_CARDS__; then
        echo "[SKIP] dummy datacards exist in $DUMMY_DIR"
        _dummy_ready=true
        return 0
    fi
    local group
    for group in "${GROUPS_[@]}"; do
        dummy_group "$group" || { echo "[FAILED] dummy cards for $group"; return 1; }
    done
    _dummy_ready=true
}

# all inputs needed for the current card_mode
prepare_inputs() {
    if [[ "$card_mode" == "dummy" ]]; then
        dummy_cards
    else
        combine_cards
    fi
}

# ---- inference ------------------------------------------------------------------------------------------------------  # noqa: E501

# run_task <task config> <stage> <workers>
run_task() {
    local task="$1" stage="$2" workers="$3"
    local outdir="$RESULT_PLOTS/$card_mode/$stage/$task"
    local cmd=(cws run --analysis "$CFG" --task-config "$task" --workers "$workers")
    [[ "$rerun" == "true" ]] && cmd+=(--remove-output 0,a,y)
    local fetch_cmd=(cws run --analysis "$CFG" --task-config "$task" --fetch-output 0,a)

    echo ""
    echo "----- [$card_mode/$stage] $task -----"
    if [[ "$dry_run" != "false" ]]; then
        echo "[DRY-RUN] (cd $outdir) ${cmd[*]}"
        [[ "$fetch_output" == "true" ]] && echo "[DRY-RUN] (cd $outdir) ${fetch_cmd[*]}"
        return 0
    fi

    mkdir -p "$outdir"
    echo "[RUNNING] ${cmd[*]}  (log: $outdir/run.log)"
    ( cd "$outdir" && "${cmd[@]}" ) 2>&1 | tee "$outdir/run.log"
    local rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        echo "[FAILED] $stage/$task (exit code $rc)"
        FAILED+=("$card_mode/$stage/$task")
        return "$rc"
    fi

    if [[ "$fetch_output" == "true" ]]; then
        ( cd "$outdir" && "${fetch_cmd[@]}" ) >> "$outdir/run.log" 2>&1 \
            || echo "[WARNING] could not fetch the outputs of $task into $outdir"
    fi
    return 0
}

# run_stage <stage>
run_stage() {
    local stage="$1" workers task tasks=()
    case "$stage" in
        __STAGE_ASIMOV__) workers="$ASIMOV_WORKERS"; tasks=("${ASIMOV_TASKS[@]}") ;;
        __STAGE_PART_UNBLINDED__) workers="$UNBLINDED_WORKERS"; tasks=("${PART_UNBLINDED_TASKS[@]}") ;;
        *) echo "[ERROR] unknown stage '$stage'"; return 2 ;;
    esac

    echo ""
    echo "========== stage: $stage, cards: $card_mode (${#tasks[@]} tasks, $workers workers) =========="
    prepare_inputs || { FAILED+=("inputs ($card_mode)"); return 1; }
    for task in "${tasks[@]}"; do
        run_task "$task" "$stage" "$workers" || true
    done
}

summary() {
    echo ""
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo "[SUMMARY] ${#FAILED[@]} failed: ${FAILED[*]}"
        return 1
    fi
    echo "[SUMMARY] all done, plots in $RESULT_PLOTS/$card_mode"
}

main() {
    [[ $# -eq 0 ]] && set -- all
    mkdir -p "$RESULT_PLOTS"
    echo "[INFO] card_mode=$card_mode, CFG=$CFG"
    local arg
    for arg in "$@"; do
        case "$arg" in
            all)
                run_stage __STAGE_ASIMOV__
                run_stage __STAGE_PART_UNBLINDED__
                ;;
            __STAGE_ASIMOV__|__STAGE_PART_UNBLINDED__)
                run_stage "$arg"
                ;;
            prepare_cards|combine_cards|dummy_cards)
                "$arg" || FAILED+=("$arg")
                ;;
            *)
                if [[ -z "${TASK_STAGE[$arg]:-}" ]]; then
                    echo "[ERROR] unknown stage or task config '$arg', choose from: all __STAGE_ASIMOV__ __STAGE_PART_UNBLINDED__ prepare_cards combine_cards dummy_cards ${!TASK_STAGE[*]}"  # noqa: E501
                    exit 2
                fi
                local stage="${TASK_STAGE[$arg]}" workers="$ASIMOV_WORKERS"
                [[ "$stage" == "__STAGE_PART_UNBLINDED__" ]] && workers="$UNBLINDED_WORKERS"
                if prepare_inputs; then
                    run_task "$arg" "$stage" "$workers" || true
                else
                    FAILED+=("inputs ($card_mode)")
                fi
                ;;
        esac
    done
    summary
}

# only run if executed, not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
"""


def build_run_script(
    task_family: str,
    exports: dict[str, str],
    cards: list[tuple[str, list[str]]],
    groups: dict[str, list[tuple[str, str]]],
    tasks: list[tuple[str, str, str, dict]],
    card_mode: str = "standard",
    asimov_workers: int = 50,
    unblinded_workers: int = 20,
) -> str:
    """
    *exports*: ordered env variables written at the top of the script; tool paths are written as overridable
    defaults (VAR=${VAR:-value}), everything else is fixed.
    *cards*: per model (label, card file names); BASE_CARDS_<label> and CARDS_<label> must be in *exports*.
    *groups*: per group (e.g. 3b, 4b, combined) the list of (channel name, card path) to combine.
    """
    overridable = {"MULTIH_DIR", "SCALE_SCRIPT", "PREPARE_CARDS_SCRIPT"}
    export_lines = "\n".join(
        f'export {key}="${{{key}:-{value}}}"' if key in overridable else f'export {key}="{value}"'
        for key, value in exports.items()
    )

    pruned = " ".join(
        f'"$CARDS_{label}/{fn}"' for label, fns in cards for fn in fns)

    prepare_lines = []
    for label, fns in cards:
        if fns:
            prepare_lines.append(
                f'    _run python3 "$PREPARE_CARDS_SCRIPT" "$BASE_CARDS_{label}/{fns[0]}" "${{prep_args[@]}}" || return 1')  # noqa: E501
    for label, fns in cards:
        for fn in fns[1:]:
            prepare_lines.append(
                f'    _run python3 "$PREPARE_CARDS_SCRIPT" "$BASE_CARDS_{label}/{fn}" "${{prep_args[@]}}" & pids+=($!)')

    combine_lines = []
    for group, entries in groups.items():
        args = " \\\n        ".join(
            f'"{name}={path}"' for name, path in entries)
        combine_lines.append(
            f"    combine_group {group} \\\n        {args} || return 1")

    combined = " ".join(f'"$COMBINED_DIR/datacard_{g}.txt"' for g in groups)
    dummies = " ".join(
        f'"$DUMMY_DIR/{g}/dummy/datacard_dummy_{g}_scaled.txt"' for g in groups)

    stage_tasks = {stage: [name for name, s, _,
                           _ in tasks if s == stage] for stage in STAGES}
    task_stage = " ".join(f"[{name}]={stage}" for name, stage, _, _ in tasks)

    replacements = {
        "__TASK__": task_family,
        "__CARD_MODE__": card_mode,
        "__ASIMOV_WORKERS__": str(asimov_workers),
        "__UNBLINDED_WORKERS__": str(unblinded_workers),
        "__EXPORTS__": export_lines,
        "__GROUPS__": " ".join(groups),
        "__ASIMOV_TASKS__": " ".join(stage_tasks[STAGE_ASIMOV]),
        "__PART_UNBLINDED_TASKS__": " ".join(stage_tasks[STAGE_PART_UNBLINDED]),
        "__TASK_STAGE__": task_stage,
        "__PRUNED_CARDS__": pruned,
        "__PREPARE_CARDS__": "\n".join(prepare_lines),
        "__COMBINED_CARDS__": combined,
        "__COMBINE_CARDS__": "\n".join(combine_lines),
        "__DUMMY_CARDS__": dummies,
        "__STAGE_ASIMOV__": STAGE_ASIMOV,
        "__STAGE_PART_UNBLINDED__": STAGE_PART_UNBLINDED,
    }
    script = RUN_SCRIPT_TEMPLATE
    for key, value in replacements.items():
        script = script.replace(key, value)
    return script


# ---------------------------------------------------------------------------------------------------------------------
# the law tasks
#
# Why two tasks: the ML categories are added to the (process-global) config objects by add_categories_ml, which is
# decorated with @call_once_on_config, so the first ML model wins. On top of that Cat_3b and Cat_4b give the same
# node names different ml_ids (tth: 2 vs 1, tthh: 1 vs 2, ttbb_custom: 4 vs 3), so their categories cannot live in
# one config anyway. Hence every inference model has to be resolved in its own law process.
# ---------------------------------------------------------------------------------------------------------------------


class PrepareCWSDatacards(
    HBWInferenceModelBase,
    InferenceModelClassMixin,
):
    """
    Collects the rebinned datacards of ONE inference model over all config groups and writes a manifest.json that
    PrepareCWSTaskCalls reads. Run it once per inference model, each in its own law call.
    """

    reqs = Requirements(
        ModifyDatacardsFlatRebin=ModifyDatacardsFlatRebin,
    )

    config_groups = law.MultiCSVParameter(
        default=(("c22prev14", "c22postv14", "c23prev14", "c23postv14"),),
        description="List of config groups to use for this task.",
        significant=False,
    )
    cards_version = luigi.Parameter(
        default="",
        description="Optional version string appended to the output path.",
        significant=True,
    )

    @classmethod
    def modify_param_values(cls, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("config_groups"):
            configs = tuple(
                config for config_group in params["config_groups"] for config in config_group)
            if configs != params.get("configs"):
                logger.warning_once(
                    f"update_configs_{params.get('configs')}_from_groups_{params['config_groups']}",
                    f"Overwriting 'configs' parameter with values {configs} because "
                    f"'config_groups' parameter with values {params['config_groups']} will be used.",
                )
                params["configs"] = configs
        return super().modify_param_values(params)

    @classmethod
    def resolve_instances(cls, params: dict[str, Any], shifts: TaskShifts) -> dict[str, Any]:
        for config_group in params["config_groups"]:
            _params = params.copy()
            _params["configs"] = config_group
            _params["config_insts"] = [
                config_inst for config_inst in params["config_insts"]
                if config_inst.name in config_group
            ]
            _params["inst_dict"] = {
                "configs": _params["configs"],
                "config_insts": _params["config_insts"],
            }
            cls.reqs.ModifyDatacardsFlatRebin.resolve_instances(
                _params, shifts)
        return params

    def _req_key(self, config_group) -> str:
        return f"rebinned_datacards__{self.configs_str(config_group)}"

    def _datacard_reqs(self) -> dict:
        return {
            self._req_key(config_group): self.reqs.ModifyDatacardsFlatRebin.req(self, configs=config_group)
            for config_group in self.config_groups
        }

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs.update(self._datacard_reqs())
        return reqs

    def requires(self):
        return self._datacard_reqs()

    @property
    def config_groups_str(self) -> list[str]:
        return [self.configs_str(configs) for configs in self.config_groups]

    def store_parts(self):
        parts = super().store_parts()
        cards_repr = "__".join(self.config_groups_str)
        if self.cards_version:
            cards_repr += f"__{self.cards_version}"
        parts.insert_before("version", "config_group", cards_repr)
        return parts

    def output(self):
        return {
            "datacards": self.target("datacards", dir=True),
            "manifest": self.target("manifest.json"),
        }

    def run(self):
        inputs = self.input()
        output = self.output()
        cards_target = output["datacards"]
        cards_target.touch()

        categories = list(self.inference_model_cls.config_categories)
        channels = []
        for config_group, group_str in zip(self.config_groups, self.config_groups_str):
            targets = list(inputs[self._req_key(
                config_group)].collection.targets.values())
            if len(targets) != len(categories):
                raise Exception(
                    f"inference model {self.inference_model} has {len(categories)} config_categories but "
                    f"{len(targets)} datacards for config group {group_str}; cannot match them",
                )
            for cat_name, target in zip(categories, targets):
                logger.info(
                    f"copying datacard {target['card'].basename} to {cards_target.abspath}")
                target["card"].copy_to(cards_target)
                target["shapes"].copy_to(cards_target)
                target["inspection"].copy_to(cards_target)
                channels.append(
                    {"category": cat_name, "config_group": group_str, "card": target["card"].basename})

        lumi = sum(config_inst.x.luminosity.get("nominal")
                   for config_inst in self.config_insts) * 0.001
        manifest = {
            "inference_model": self.inference_model,
            "ml_models": list(getattr(self, "ml_models", ()) or ()),
            "configs": list(self.configs),
            "config_groups": self.config_groups_str,
            "lumi_fb": float(lumi),
            "cards_dir": cards_target.abspath,
            "channels": channels,
        }
        output["manifest"].dump(manifest, formatter="json", indent=4)
        print(f"\n# manifest\n{output['manifest'].abspath}")


class PrepareCWSTaskCalls(HBWTask, AnalysisTask):
    """
    Combines the manifests of several PrepareCWSDatacards calls (default labels: 3b, 4b) into the YAML task configs
    for ``cws run`` (standard and dummy cards), the result_plots folder and run.sh. No inference or ML model is set up here. Use prepare_cws.sh to run both steps.  # noqa: E501
    """

    manifests = law.CSVParameter(
        default=(),
        description="manifest.json files written by PrepareCWSDatacards, one per inference model.",
    )
    model_labels = law.CSVParameter(
        default=("3b", "4b"),
        description="Short label per manifest (same order). Used as datacard group names in the YAML and as env "
        "variable suffix (CARDS_<label>). Default: 3b,4b",
    )
    card_mode = luigi.ChoiceParameter(
        default="standard",
        choices=CARD_MODES,
        description="Default card mode of run.sh: 'standard' (pruned cards as they are) or 'dummy' (combined, "
        "renamed, non-SM removed, dummy card for the multiH model). Both YAMLs are always written; run.sh can "
        "switch with card_mode=... Default: standard",
        significant=False,
    )
    cws_version = luigi.Parameter(
        default="",
        description="'version' written into the YAML parameters ('__dummy' is appended for the dummy YAML). "
        "Default: built from the config groups and inference models.",
        significant=False,
    )
    multih_dir = luigi.Parameter(
        default="/data/dust/user/markusla/analysis/hhh_inference/multih_models",
        description="multih_models checkout (contains scripts/make_dummy_card.py and scripts/check_card.py).",
        significant=False,
    )
    scale_script = luigi.Parameter(
        default="/data/dust/user/markusla/analysis/hhh_inference/cws/data/renamed_cards/tech_test/"
        "scale_dummy_templates.py",
        description="Script that scales the dummy templates.",
        significant=False,
    )
    prepare_cards_script = luigi.Parameter(
        default="prepare_cards.py",
        description="prepare_cards.py to run on every card; relative paths are taken from where run.sh is started.",
        significant=False,
    )
    run_script_name = luigi.Parameter(
        default="run.sh", description="Name of the run script.")
    yaml_name = luigi.Parameter(
        default="inference.yaml", description="Name of the YAML task config.")
    asimov_workers = luigi.IntParameter(
        default=50,
        description="Default number of law workers in the asimov stage of run.sh (overridable via ASIMOV_WORKERS).",
        significant=False,
    )
    unblinded_workers = luigi.IntParameter(
        default=20,
        description="Default number of law workers in the part_unblinded stage of run.sh (UNBLINDED_WORKERS).",
        significant=False,
    )

    # physics model settings written into the YAML parameters
    physics_model = "multih_models.kappa_framework.multiH.hh_model:model_hhh_tthh"
    model_options = ["poiRanges=k4=-300,300:kl=-30,30"]
    parameter_ranges = [
        {"name": "k4", "min": -300, "max": 300},
        {"name": "kl", "min": -30, "max": 30},
        {"name": "kt", "min": 0, "max": 2},
    ]
    # model spec and energy for make_dummy_card.py (relative to multih_dir)
    dummy_model = "kappa_framework/multiH/model.yaml:model_hhh_tthh"
    dummy_energy = "13p6TeV"

    # top-level YAML key the task configs are written under; change it if cws expects another one
    yaml_tasks_key = "tasks"

    @classmethod
    def modify_param_values(cls, params: dict[str, Any]) -> dict[str, Any]:
        params = super().modify_param_values(params)
        manifests, labels = tuple(params.get("manifests") or ()), tuple(
            params.get("model_labels") or ())
        if manifests and len(manifests) != len(labels):
            raise ValueError(
                f"--model-labels {labels} needs one label per file in --manifests {manifests}")
        return params

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "models", "__".join(self.model_labels))
        return parts

    def output(self):
        dummy_yaml = self.yaml_name.replace(".yaml", "_dummy.yaml") if self.yaml_name.endswith(".yaml") \
            else f"{self.yaml_name}_dummy"
        return {
            "run": self.target(self.run_script_name),
            "yaml": self.target(self.yaml_name),
            "yaml_dummy": self.target(dummy_yaml),
            "result_plots": self.target("result_plots", dir=True),
        }

    def load_manifests(self) -> list[dict]:
        if not self.manifests:
            raise ValueError(
                "no --manifests given; run PrepareCWSDatacards per inference model first (prepare_cws.sh)")
        loaded = []
        for path in self.manifests:
            path = os.path.expandvars(os.path.expanduser(path))
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"manifest {path} does not exist; run PrepareCWSDatacards first")
            with open(path) as f:
                loaded.append(json.load(f))
        return loaded

    def run(self):
        output = self.output()
        manifests = self.load_manifests()

        base_dir = os.path.dirname(output["run"].abspath)
        cards_dir_out = os.path.join(base_dir, "cards")
        combined_dir = os.path.join(cards_dir_out, "combined")
        dummy_dir = os.path.join(cards_dir_out, "dummy")
        result_plots_dir = output["result_plots"].abspath

        exports = {
            "BASE": base_dir,
            "CFG_STANDARD": output["yaml"].abspath,
            "CFG_DUMMY": output["yaml_dummy"].abspath,
            "RESULT_PLOTS": result_plots_dir,
            "COMBINED_DIR": combined_dir,
            "DUMMY_DIR": dummy_dir,
            "PHYSICS_MODEL": self.physics_model,
            "DUMMY_MODEL": self.dummy_model,
            "DUMMY_ENERGY": self.dummy_energy,
            "MULTIH_DIR": self.multih_dir,
            "SCALE_SCRIPT": self.scale_script,
            "PREPARE_CARDS_SCRIPT": self.prepare_cards_script,
        }

        # standard channels: one per pruned per-category card
        channels, groups, cards_per_model, all_keys = [], {}, [], []
        # group -> [(channel name, card path)] for combineCards.py
        combine_groups = {}
        for manifest, label in zip(manifests, self.model_labels):
            cards_dir = manifest["cards_dir"]
            pruned_dir = os.path.join(cards_dir, "pruned")
            exports[f"BASE_CARDS_{label}"] = cards_dir
            exports[f"CARDS_{label}"] = pruned_dir

            keys = []
            for ch in manifest["channels"]:
                key = channel_name(
                    ch["category"], ch["config_group"], label, all_keys + keys)
                channels.append((
                    f"{label} ({manifest['inference_model']})",
                    key,
                    {
                        "path": os.path.join(pruned_dir, ch["card"]),
                        "name": f"{label} {ch['category']} ({ch['config_group']})",
                    },
                ))
                keys.append(key)
            groups[label] = {"cards": ",".join(keys), "name": label}
            all_keys.extend(keys)
            cards_per_model.append(
                (label, [ch["card"] for ch in manifest["channels"]]))
            combine_groups[label] = [
                (key, f"$CARDS_{label}/{ch['card']}") for key, ch in zip(keys, manifest["channels"])
            ]

        groups["combined"] = {"cards": ",".join(
            all_keys), "name": "+".join(self.model_labels)}
        combine_groups["combined"] = [
            entry for label in self.model_labels for entry in combine_groups[label]]
        datacard_groups = ":".join(groups)

        # dummy channels: one scaled dummy card per group
        dummy_channels, dummy_groups = [], {}
        for group, group_cfg in groups.items():
            key = f"dummy_{group}"
            dummy_channels.append((
                "dummy cards (made by run.sh: card_mode=dummy)",
                key,
                {
                    "path": os.path.join(dummy_dir, group, "dummy", f"datacard_dummy_{group}_scaled.txt"),
                    "name": f"{group_cfg['name']} (dummy)",
                },
            ))
            dummy_groups[group] = {"cards": key, "name": group_cfg["name"]}

        # lumi / campaign: the models should be built from the same configs
        lumis = {round(m["lumi_fb"], 3) for m in manifests}
        configs = manifests[0]["configs"]
        if len(lumis) > 1 or any(m["configs"] != configs for m in manifests):
            logger.warning(
                f"the inference models use different configs/luminosities: "
                f"{[(m['inference_model'], m['configs'], m['lumi_fb']) for m in manifests]}; using the first one",
            )
        lumi = manifests[0]["lumi_fb"]
        campaign = configs[0] if len(configs) == 1 else "run3"

        identifier = "__".join(
            [*manifests[0]["config_groups"], *(m["inference_model"] for m in manifests)])
        version = self.cws_version or identifier
        tasks = build_task_configs(datacard_groups)

        def parameters(mode: str) -> dict:
            return {
                "version": version if mode == "standard" else f"{version}__dummy",
                "lumi": f"{lumi:.1f} fb^{{-1}}",
                "target_luminosity": round(lumi),
                "campaign": campaign,
                "parameter_ranges": self.parameter_ranges,
                "model": self.physics_model,
                "model_options": self.model_options,
                "BASE": base_dir,
                "CFG": exports["CFG_STANDARD"] if mode == "standard" else exports["CFG_DUMMY"],
                **{k: v for k, v in exports.items() if k.startswith(("BASE_CARDS_", "CARDS_"))},
                "output_dir": os.path.join(result_plots_dir, mode),
            }

        def header(mode: str) -> str:
            return "\n".join([
                f"# AUTO-GENERATED by {self.task_family} -- DO NOT EDIT, rerun the law task instead.",
                f"# card mode: {mode}",
                "# inference models: " + ", ".join(
                    f"{label}={m['inference_model']} (ML: {'+'.join(m['ml_models']) or '?'})"
                    for m, label in zip(manifests, self.model_labels)
                ),
                "#",
                f"# run everything in stages:   card_mode={mode} bash $BASE/run.sh [asimov|part_unblinded|all]",
                "# or a single task config:    cws run --analysis $CFG --task-config <name> --workers 10",
            ])

        output["yaml"].dump(
            build_yaml_text(parameters("standard"), channels, groups,
                            tasks, self.yaml_tasks_key, header("standard")),
            formatter="text",
        )
        output["yaml_dummy"].dump(
            build_yaml_text(parameters("dummy"), dummy_channels, dummy_groups, tasks, self.yaml_tasks_key,
                            header("dummy")),
            formatter="text",
        )
        output["result_plots"].touch()

        output["run"].dump(
            build_run_script(
                task_family=self.task_family,
                exports=exports,
                cards=cards_per_model,
                groups=combine_groups,
                tasks=tasks,
                card_mode=self.card_mode,
                asimov_workers=self.asimov_workers,
                unblinded_workers=self.unblinded_workers,
            ),
            formatter="text",
        )
        os.chmod(output["run"].abspath, 0o755)

        print(
            f"\n# task configs\n{output['yaml'].abspath}\n{output['yaml_dummy'].abspath}")
        print("\n# run the inference; dry_run=true only prints the commands")
        print(
            f"card_mode={self.card_mode} bash {output['run'].abspath} [asimov|part_unblinded|all]")
