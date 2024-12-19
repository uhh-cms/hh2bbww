# coding: utf-8

"""
Tasks related to the creation and modification of datacards for inference purposes.
"""

from __future__ import annotations

from typing import Callable

# import luigi
import law
import order as od

from columnflow.tasks.framework.base import Requirements, RESOLVE_DEFAULT, AnalysisTask, ConfigTask
from columnflow.tasks.framework.parameters import SettingsParameter
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin, InferenceModelMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.cms.inference import CreateDatacards
from columnflow.util import dev_sandbox, maybe_import  # import_ROOT
from hbw.tasks.base import HBWTask
# from hbw.tasks.postfit_plots import load_hists_uproot

# import hbw.inference.constants as const

array = maybe_import("array")
uproot = maybe_import("uproot")
np = maybe_import("numpy")
hist = maybe_import("hist")


logger = law.logger.get_logger(__name__)


# Function copied from Mathis Hist hook commit
def apply_rebinning_edges(h: hist.Histogram, axis_name: str, edges: list):
    """
    Generalized rebinning of a single axis from a hist.Histogram, using predefined edges.

    :param h: histogram to rebin
    :param axis_name: string representing the axis to rebin
    :param edges: list of floats representing the new bin edges. Must be a subset of the original edges.
    :return: rebinned histogram
    """
    if isinstance(edges, int):
        return h[{axis_name: hist.rebin(edges)}]

    ax = h.axes[axis_name]
    ax_idx = [a.name for a in h.axes].index(axis_name)
    if not all([np.isclose(x, ax.edges).any() for x in edges]):
        raise ValueError(
            f"Cannot rebin histogram due to incompatible edges for axis '{ax.name}'\n"
            f"Edges of histogram are {ax.edges}, requested rebinning to {edges}",
        )

    # If you rebin to a subset of initial range, keep the overflow and underflow
    overflow = ax.traits.overflow or (edges[-1] < ax.edges[-1] and not np.isclose(edges[-1], ax.edges[-1]))
    underflow = ax.traits.underflow or (edges[0] > ax.edges[0] and not np.isclose(edges[0], ax.edges[0]))
    flow = overflow or underflow
    new_ax = hist.axis.Variable(edges, name=ax.name, overflow=overflow, underflow=underflow)
    axes = list(h.axes)
    axes[ax_idx] = new_ax

    hnew = hist.Hist(*axes, name=h.name, storage=h._storage_type())

    # Offset from bin edge to avoid numeric issues
    offset = 0.5 * np.min(ax.edges[1:] - ax.edges[:-1])
    edges_eval = edges + offset
    edge_idx = ax.index(edges_eval)
    # Avoid going outside the range, reduceat will add the last index anyway
    if edge_idx[-1] == ax.size + ax.traits.overflow:
        edge_idx = edge_idx[:-1]

    if underflow:
        # Only if the original axis had an underflow should you offset
        if ax.traits.underflow:
            edge_idx += 1
        edge_idx = np.insert(edge_idx, 0, 0)

    # Take is used because reduceat sums i:len(array) for the last entry, in the case
    # where the final bin isn't the same between the initial and rebinned histogram, you
    # want to drop this value. Add tolerance of 1/2 min bin width to avoid numeric issues
    hnew.values(flow=flow)[...] = np.add.reduceat(h.values(flow=flow), edge_idx,
            axis=ax_idx).take(indices=range(new_ax.size + underflow + overflow), axis=ax_idx)
    if hnew._storage_type() == hist.storage.Weight():
        hnew.variances(flow=flow)[...] = np.add.reduceat(h.variances(flow=flow), edge_idx,
                axis=ax_idx).take(indices=range(new_ax.size + underflow + overflow), axis=ax_idx)
    return hnew


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
    N_bins_input = hist.axes[0].size
    # N_bins_input = hist.GetNbinsX()

    # # replace empty bin values (TODO: remove as soon as we can disable the empty bin filling)
    # EMPTY_BIN_VALUE = 1e-5
    # for i in range(1, N_bins_input + 1):
    #     if hist.GetBinContent(i) == EMPTY_BIN_VALUE:
    #         hist.SetBinContent(i, 0)

    # determine events per bin the final histogram should have
    events_per_bin = hist.sum().value / N_bins_final
    # events_per_bin = hist.Integral() / N_bins_final
    logger.info(f"============ {round(events_per_bin, 3)} events per bin")

    # bookkeeping number of bins and number of events
    bin_count = 1
    N_events = 0
    # rebin_values = [hist.GetBinLowEdge(1)]
    rebin_values = [int(hist.axes[0].edges[0])]

    # starting loop at 1 to exclude underflow
    # ending at N_bins_input + 1 excludes overflow
    # for i in range(1, N_bins_input + 1):
    for i in range(1, N_bins_input):
        if bin_count == N_bins_final:
            # break as soon as N-1 bin edges have been determined --> final bin is x_max
            break
        N_events += hist.view()["value"][i]
        # N_events += hist.GetBinContent(i)
        if i % 100 == 0:
            logger.info(f"========== Bin {i} of {N_bins_input}, {N_events} events")
        if N_events >= events_per_bin * bin_count:
            # when *N_events* surpasses threshold, append the corresponding bin edge and count
            # logger.info(f"++++++++++ Append bin edge {bin_count} of {N_bins_final} at edge {hist.GetBinLowEdge(i)}")
            logger.info(f"++++++++++ Append bin edge {bin_count} of {N_bins_final} at edge {hist.axes[0].edges[i]}")
            # rebin_values.append(hist.GetBinLowEdge(i + 1))
            rebin_values.append(hist.axes[0].edges[i])
            bin_count += 1

    # final bin is x_max
    x_max = hist.axes[0].edges[N_bins_input]
    # x_max = hist.GetBinLowEdge(N_bins_input + 1)
    rebin_values.append(x_max)
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


class ModifyDatacardsFlatRebin(
    HBWTask,
    InferenceModelMixin,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    ConfigTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))
    # sandbox = dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_plotting.sh")
    # sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/cmssw_default.sh")

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

            # resolve default and groups for `bins_per_category`
            params["bins_per_category"] = cls.resolve_config_default(
                params,
                params.get("bins_per_category"),
                container=config_inst,
                default_str="default_bins_per_category",
            )
            params["bins_per_category"] = resolve_category_groups(params["bins_per_category"], config_inst)

            # set `inference_category_rebin_processes` as parameter and resolve groups
            params["inference_category_rebin_processes"] = cls.resolve_config_default(
                params,
                RESOLVE_DEFAULT,
                container=config_inst,
                default_str="inference_category_rebin_processes",
            )
            params["inference_category_rebin_processes"] = resolve_category_groups(
                params["inference_category_rebin_processes"],
                config_inst,
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
        processes = self.branch_data.processes.copy()
        # inf_proc_names = [self.inference_model_inst.inf_proc(proc.name) for proc in self.branch_data.processes]

        rebin_process_condition = self.inference_category_rebin_processes.get(config_category, None)
        if not rebin_process_condition:
            if "ggf" in config_category:
                for proc in processes.copy():
                    proc_name = proc.config_process

            #         if not "ggf" in proc_name:
            #             processes.remove(proc)
            #         elif proc_name == 'h_ggf': # This is quiet custom made
            #             processes.remove(proc)
            #         elif "kl5" in proc_name:
            #             processes.remove(proc)
            #     return processes
            # elif 'vbf' in config_category:
            #     for proc in processes.copy():
            #         proc_name = proc.config_process
            #         if not 'vbf' in proc_name:
            #             processes.remove(proc)
            #         if proc_name == 'h_vbf':
            #             processes.remove(proc)
            #     return processes
            # else:
                logger.warning(
                    f"No rebin condition found for category {config_category}; rebinning will be flat "
                    f"on all processes {[proc.config_process for proc in processes]}",
                )
                return processes

        # transform `rebin_process_condition` into Callable if required
        if not isinstance(rebin_process_condition, Callable):
            _rebin_processes = list(rebin_process_condition)
            rebin_process_condition = lambda _proc_name: _proc_name in _rebin_processes

        for proc in processes.copy():
            proc_name = proc.config_process
            # check for each process if the *rebin_process_condition*  is fulfilled
            if not rebin_process_condition(proc_name):
                processes.remove(proc)
        logger.info(
            f"Category {config_category} will be rebinned flat in processes "
            f"{[proc.config_process for proc in processes]}",
        )
        return processes

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
        # import uproot
        # from ROOT import TH1
        # import_ROOT()

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
            # logger.info(f"File keys: {file.keys()}")
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
                proc_name: file[get_hist_name(cat_name, proc_name)].to_hist()
                # proc_name: file[get_hist_name(cat_name, proc_name)].to_pyroot()
                for proc_name in proc_names
            }

            # determine all processes required for the category *cat_name* to determine the rebin values
            rebin_processes = self.get_rebin_processes()
            rebin_inf_proc_names = [proc.name for proc in rebin_processes]

            if diff := set(rebin_inf_proc_names).difference(nominal_hists.keys()):
                raise Exception(f"Histograms {diff} requested for rebinning but no corresponding "
                "nominal histograms found")

            hists = [nominal_hists[proc_name] for proc_name in rebin_inf_proc_names]

            hist = hists[0]
            for h in hists[1:]:
                hist += h

            logger.info(f"Finding rebin values for category {cat_name} using processes {rebin_processes}")
            rebin_values = get_rebin_values(hist, self.get_n_bins())
            outputs["edges"].dump(rebin_values, formatter="json")

            # apply rebinning on all histograms and store resulting hists in a ROOT file
            out_file = uproot.recreate(outputs["shapes"].fn)
            rebinned_histograms = {}
            for key, h in file.items():

                key = key.split(";")[0]
                if "TH1" in str(h):
                    try:
                        h = file[key].to_hist()
                    except AttributeError:
                        continue
                    h_rebin = apply_rebinning_edges(h, h.axes[0].name, rebin_values)
                    # bin_edges = h_rebin.axes[0].edges  # Bin edges (first axis)
                    # bin_values = h_rebin.values()
                    # bin_variance = h_rebin.variances()  # Bin values
                    # h_rebin_root = uproot.newhist.Histogram(
                    #     bins=(len(h_rebin.edges)-1),
                    #     x_low=h_rebin.edges[0],
                    #     x_high=h_rebin.edges[-1],
                    #     values=h_rebin.values(),
                    # )
                    rebinned_histograms[key] = h_rebin

                    logger.info(f"Inserting histogram with name {key}")
                    out_file[key] = h_rebin

            # for key, h in file.items():
            #     # remove ";1" appendix
            #     key = key.split(";")[0]
            #     __import__("IPython").embed()
            #     # try:
            #         # convert histograms to pyroot
            #         # h = h.to_pyroot()
            #     # except AttributeError:
            #         # skip non-histograms
            #         # continue
            #     # if not isinstance(h, TH1):
            #     #     raise Exception(f"{h} is not a TH1 histogram")

            #     # h_rebin = apply_binning(h, rebin_values)
            #     h_rebin = apply_rebinning_edges(h, "0", rebin_values)
            #     # problematic_bin_count = check_empty_bins(h_rebin)
            #     logger.info(f"Inserting histogram with name {key}")
            #     out_file[key] = uproot.recreate(h_rebin)


class PrepareInferenceTaskCalls(
    HBWTask,
    InferenceModelMixin,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    ConfigTask,
):
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
            "PlotUpperLimitsAtPoint": self.target("PlotUpperLimitsAtPoint.txt"),
            "PlotUpperLimitsPoint": self.target("PlotUpperLimitsPoint.txt"),
            "FitDiagnostics": self.target("FitDiagnostics.txt"),
        }

    def run(self):
        inputs = self.input()
        output = self.output()

        # string that represents the version of datacards
        identifier = "__".join([self.config, self.selector, self.inference_model, self.version])

        # get the datacard names from the inputs
        collection = inputs["rebinned_datacards"]["collection"]
        cards_path = {collection[key]["card"].dirname for key in collection.keys()}
        if len(cards_path) != 1:
            raise Exception("Expected only one datacard path")
        cards_path = cards_path.pop()

        card_fns = [collection[key]["card"].basename for key in collection.keys()]

        # get the category names from the inference models
        categories = self.inference_model_inst.categories
        cat_names = [c.name for c in categories]

        # combine category names with card fn to a single string
        datacards = ",".join([f"{cat_name}=$CARDS_PATH/{card_fn}" for cat_name, card_fn in zip(cat_names, card_fns)])

        base_cmd = f"export CARDS_PATH={cards_path}" + "\n"

        print("\n\n")
        # creating upper limits for kl=1
        cmd = base_cmd + (
            f"law run PlotUpperLimitsAtPoint --version {identifier} --multi-datacards {datacards} "
            f"--datacard-names {identifier}"
        )
        print(cmd, "\n\n")
        output["PlotUpperLimitsAtPoint"].dump(cmd, formatter="text")

        # creating kl scan
        cmd = base_cmd + (
            f"law run PlotUpperLimits --version {identifier} --datacards {datacards} "
            f"--xsec fb --y-log"
        )
        print(cmd, "\n\n")
        output["PlotUpperLimitsPoint"].dump(cmd, formatter="text")

        # running FitDiagnostics for Pre+Postfit plots
        cmd = base_cmd + (
            f"law run FitDiagnostics --version {identifier} --datacards {datacards} "
            f"--skip-b-only"
        )
        print(cmd, "\n\n")
        output["FitDiagnostics"].dump(cmd, formatter="text")

        # running FitDiagnostics for Pre+Postfit plots
        cmd = base_cmd + (
            f"law run PlotPullsAndImpacts --version {identifier} --datacards {datacards} "
            f"--order-by-impact"
        )
        print(cmd, "\n\n")
        output["FitDiagnostics"].dump(cmd, formatter="text")
