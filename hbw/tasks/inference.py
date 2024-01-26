# coding: utf-8

"""
Tasks related to the creation and modification of datacards for inference purposes.
"""

from __future__ import annotations

from typing import Callable

# import luigi
import law
import order as od

from columnflow.tasks.framework.base import Requirements, RESOLVE_DEFAULT
from columnflow.tasks.framework.parameters import SettingsParameter
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin, InferenceModelMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.cms.inference import CreateDatacards
from columnflow.util import dev_sandbox, maybe_import
from hbw.tasks.base import HBWTask

array = maybe_import("array")


logger = law.logger.get_logger(__name__)


def get_hist_name(cat_name: str, proc_name: str, syst_name: str | None = None) -> str:
    hist_name = f"{cat_name}/{proc_name}"
    if syst_name:
        hist_name += f"__{syst_name}"
    return hist_name


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


def get_rebin_values(hist, N_bins_final: int = 10):
    """
    Function that determines how to rebin a hist to *N_bins_final* bins such that
    the resulting histogram is flat
    """
    N_bins_input = hist.GetNbinsX()

    # replace empty bin values (TODO: remove as soon as we can disable the empty bin filling)
    EMPTY_BIN_VALUE = 1e-5
    for i in range(1, N_bins_input + 1):
        if hist.GetBinContent(i) == EMPTY_BIN_VALUE:
            hist.SetBinContent(i, 0)

    # determine events per bin the final histogram should have
    events_per_bin = hist.Integral() / N_bins_final
    logger.info(f"============ {round(events_per_bin, 3)} events per bin")

    # bookkeeping number of bins and number of events
    bin_count = 1
    N_events = 0
    rebin_values = [hist.GetBinLowEdge(1)]

    # starting loop at 1 to exclude underflow
    # ending at N_bins_input + 1 excludes overflow
    for i in range(1, N_bins_input + 1):
        if bin_count == N_bins_final:
            # break as soon as N-1 bin edges have been determined --> final bin is x_max
            break

        N_events += hist.GetBinContent(i)
        if i % 100 == 0:
            logger.info(f"========== Bin {i} of {N_bins_input}, {N_events} events")
        if N_events >= events_per_bin * bin_count:
            # when *N_events* surpasses threshold, append the corresponding bin edge and count
            logger.info(f"++++++++++ Append bin edge {bin_count} of {N_bins_final} at edge {hist.GetBinLowEdge(i)}")
            rebin_values.append(hist.GetBinLowEdge(i + 1))
            bin_count += 1

    # final bin is x_max
    x_max = hist.GetBinLowEdge(N_bins_input + 1)
    rebin_values.append(x_max)
    logger.info(f"final bin edges: {rebin_values}")
    return rebin_values


def apply_binning(hist, rebin_values: list):
    N_bins = len(rebin_values) - 1
    rebin_values_ptr = array.array("d", rebin_values)

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


class ModifyDatacardsFlatRebin(
    HBWTask,
    InferenceModelMixin,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/cmssw_default.sh")

    bins_per_category = SettingsParameter(
        default=(RESOLVE_DEFAULT,),
        description="Number of bins per category in the format `cat_name=n_bins,...`; ",
    )

    inference_category_rebin_processes = SettingsParameter(
        default=(RESOLVE_DEFAULT,),
        significant=False,
        description="Dummy Parameter; only used via config_inst default",
    )

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateDatacards=CreateDatacards,
    )

    @classmethod
    def resolve_param_values(cls, params):
        params = super().resolve_param_values(params)

        if config_inst := params.get("config_inst"):
            def resolve_category_groups(param):
                outp_param = {}
                for cat_name in list(param.keys()):
                    resolved_cats = cls.find_config_objects(
                        (cat_name,), config_inst, od.Category,
                        object_groups=config_inst.x.category_groups, deep=True,
                    )
                    if resolved_cats:
                        for resolved_cat in law.util.make_tuple(resolved_cats):
                            outp_param[resolved_cat] = param[cat_name]
                return outp_param

            # resolve default and groups for `bins_per_category`
            params["bins_per_category"] = cls.resolve_config_default(
                params,
                params.get("bins_per_category"),
                container=config_inst,
                default_str="default_bins_per_category",
            )
            params["bins_per_category"] = resolve_category_groups(params["bins_per_category"])

            # set `inference_category_rebin_processes` as parameter and resolve groups
            params["inference_category_rebin_processes"] = cls.resolve_config_default(
                params,
                RESOLVE_DEFAULT,
                container=config_inst,
                default_str="inference_category_rebin_processes",
            )
            params["inference_category_rebin_processes"] = resolve_category_groups(
                params["inference_category_rebin_processes"],
            )
        return params

    def get_n_bins(self, DEFAULT_N_BINS=8):
        """ Method to get the requested number of bins for the current category. Defaults to *DEFAULT_N_BINS*"""
        config_category = self.branch_data.config_category
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
        config_category = self.branch_data.config_category
        proc_names = [proc.name for proc in self.branch_data.processes]

        rebin_process_condition = self.inference_category_rebin_processes.get(config_category, None)
        if not rebin_process_condition:
            logger.warning(
                f"No rebin condition found for category {config_category}; rebinning will be flat "
                f"on all processes {proc_names}",
            )
            return proc_names

        # transform `rebin_process_condition` into Callable if required
        if not isinstance(rebin_process_condition, Callable):
            _rebin_processes = law.util.make_tuple(rebin_process_condition)
            rebin_process_condition = lambda _proc_name: _proc_name in _rebin_processes

        for proc_name in proc_names.copy():
            # check for each process if the *rebin_process_condition*  is fulfilled
            if not rebin_process_condition(proc_name):
                proc_names.remove(proc_name)
        logger.info(f"Category {config_category} will be rebinned flat in processes {proc_names}")
        return proc_names

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
        cat_obj = self.branch_data
        basename = lambda name, ext: f"{name}__cat_{cat_obj.config_category}__var_{cat_obj.config_variable}.{ext}"
        n_bins = self.get_n_bins()
        return {
            "card": self.target(basename(f"datacard_rebin_{n_bins}", "txt")),
            "shapes": self.target(basename(f"shapes_rebin_{n_bins}", "root")),
            "edges": self.target(basename(f"edges_{n_bins}", "json")),
        }

    def run(self):
        import uproot
        from ROOT import TH1

        # config_inst = self.config_inst
        # inference_model = self.inference_model_inst

        inputs = self.input()
        outputs = self.output()

        inp_shapes = inputs["datacards"]["shapes"]
        inp_datacard = inputs["datacards"]["card"]

        # create a copy of the datacard with modified name of the shape file
        datacard = inp_datacard.load(formatter="text")
        datacard = datacard.replace(inp_shapes.basename, outputs["shapes"].basename)
        outputs["card"].dump(datacard, formatter="text")

        with uproot.open(inp_shapes.fn) as file:
            logger.info(f"File keys: {file.keys()}")
            # determine which histograms are present
            cat_names, proc_names, syst_names = get_cat_proc_syst_names(file)

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
                proc_name: file[get_hist_name(cat_name, proc_name)].to_pyroot()
                for proc_name in proc_names
            }

            # determine all processes required for the category *cat_name* to determine the rebin values
            rebin_processes = self.get_rebin_processes()

            if diff := set(rebin_processes).difference(nominal_hists.keys()):
                raise Exception(f"Histograms {diff} requested for rebinning but no corresponding "
                "nominal histograms found")

            hists = [nominal_hists[proc_name] for proc_name in rebin_processes]

            hist = hists[0]
            for h in hists[1:]:
                hist += h

            logger.info(f"Finding rebin values for category {cat_name} using processes {rebin_processes}")
            rebin_values = get_rebin_values(hist, self.get_n_bins())
            outputs["edges"].dump(rebin_values, formatter="json")

            # apply rebinning on all histograms and store resulting hists in a ROOT file
            out_file = uproot.recreate(outputs["shapes"].fn)
            for key, h in file.items():
                # remove ";1" appendix
                key = key.split(";")[0]
                try:
                    # convert histograms to pyroot
                    h = h.to_pyroot()
                except AttributeError:
                    # skip non-histograms
                    continue
                if not isinstance(h, TH1):
                    raise Exception(f"{h} is not a TH1 histogram")

                h_rebin = apply_binning(h, rebin_values)
                problematic_bin_count = check_empty_bins(h_rebin)  # noqa
                logger.info(f"Inserting histogram with name {key}")
                out_file[key] = uproot.from_pyroot(h_rebin)
