# coding: utf-8

"""
Tasks related to the creation and modification of datacards for inference purposes.
"""

from __future__ import annotations

import luigi
import law

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin, InferenceModelMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.cms.inference import CreateDatacards
from columnflow.util import dev_sandbox, maybe_import

array = maybe_import("array")


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

    # determine events per bin the final histogram should have
    events_per_bin = hist.Integral() / N_bins_final
    print(f" ==== {events_per_bin} events per bin")

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
            print(f" ========== Bin {i} of {N_bins_input}, {N_events} events")
        if N_events >= events_per_bin * bin_count:
            # when *N_events* surpasses threshold, append the corresponding bin edge and count
            print(f" ++++++++++ Append bin edge {bin_count} of {N_bins_final} at edge {hist.GetBinLowEdge(i)}")
            rebin_values.append(hist.GetBinLowEdge(i))
            bin_count += 1

    # final bin is x_max
    x_max = hist.GetBinLowEdge(N_bins_input + 1)
    rebin_values.append(x_max)
    print(f"final bin edges: {rebin_values}")
    return rebin_values


def apply_binning(hist, rebin_values: list):
    N_bins = len(rebin_values) - 1
    rebin_values_ptr = array.array("d", rebin_values)

    h_out = hist.Rebin(N_bins, hist.GetName(), rebin_values_ptr)
    return h_out


def check_empty_bins(hist) -> int:
    """ Checks for empty bins, negative bin content, or bins with less than 3 entries """
    print(f"Checking histogram {hist.GetName()}")
    count = 0
    for i in range(1, hist.GetNbinsX() + 1):
        value = hist.GetBinContent(i)
        error = hist.GetBinError(i)
        if value <= 0 or error > 0.57 * value:
            # error above 57% means that we have less than 3 MC events
            # (assuming each MC event has the same weight)
            count += 1
            print(f"==== Found issue with bin {i}, (value: {value}, error: {error})")

    return count


def print_hist(hist, max_bins: int = 20):
    print("Printing bin number, lower edge and bin content")
    for i in range(0, hist.GetNbinsX() + 2):
        if i > max_bins:
            return

        print(f"{i} \t {hist.GetBinLowEdge(i)} \t {hist.GetBinContent(i)}")


class ModifyDatacards(
    InferenceModelMixin,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    # NOTE: probably needs PyRoot
    # sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))
    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/cmssw_default.sh")

    n_bins_sr = luigi.IntParameter(
        default=10,
        description="Number of bins in the Signal Region histograms",
    )
    n_bins_vbfsr = luigi.IntParameter(
        default=5,
        description="Number of bins in the VBF Signal Region histograms",
    )
    n_bins_br = luigi.IntParameter(
        default=3,
        description="Number of bins in the Background Region histograms",
    )

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateDatacards=CreateDatacards,
    )

    def get_category_type(self):
        cat_name = self.branch_data.name

        if "ggHH" in cat_name:
            return "SR"
        elif "qqHH" in cat_name:
            return "vbfSR"
        else:
            return "BR"

    def n_bins(self):
        """ Helper to determine the requested number of bins for the current category """
        cat_type = self.get_category_type()

        n_bins = {
            "SR": self.n_bins_sr,
            "vbfSR": self.n_bins_vbfsr,
            "BR": self.n_bins_br,
        }[cat_type]
        return n_bins

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

        n_bins = self.n_bins()
        return {
            "card": self.target(basename(f"datacard_rebin_{n_bins}", "txt")),
            "shapes": self.target(basename(f"shapes_rebin_{n_bins}", "root")),
        }

    def run(self):
        import uproot
        from ROOT import TH1

        inputs = self.input()
        outputs = self.output()

        inp_shapes = inputs["datacards"]["shapes"]
        inp_datacard = inputs["datacards"]["card"]

        # create a copy of the datacard with modified name of the shape file
        datacard = inp_datacard.load(formatter="text")
        datacard = datacard.replace(inp_shapes.basename, outputs["shapes"].basename)
        outputs["card"].dump(datacard, formatter="text")

        with uproot.open(inp_shapes.fn) as file:
            print(f"File keys: {file.keys()}")
            # determine which histograms are present
            cat_names, proc_names, syst_names = get_cat_proc_syst_names(file)

            if len(cat_names) != 1:
                raise Exception("Expected 1 category per file")

            cat_name = list(cat_names)[0]
            if cat_name != self.branch_data.name:
                raise Exception(
                    f"Category name in the histograms {cat_name} does not agree with the"
                    f"datacard category name {self.branch_data.name}",
                )

            # get all histograms relevant for determining the rebin values
            nominal_hists = {
                proc_name: file[f"{cat_name}/{proc_name}"].to_pyroot()
                for proc_name in proc_names
            }

            cat_type = self.get_category_type()

            if "SR" in cat_type:
                # HH signal region --> flat in signal
                hist = nominal_hists["ggHH_kl_1_kt_1_sl_hbbhww"]
            else:
                # background region --> flat in all backgrounds
                hists = [
                    nominal_hists[proc_name]
                    for proc_name in nominal_hists.keys()
                    if "ggHH" not in proc_name
                ]
                hist = hists[0]
                for h in hists[1:]:
                    hist += h

            print(f"Finding rebin values for category {cat_name}")
            rebin_values = get_rebin_values(hist, self.n_bins())

            # apply rebinning on all histograms and store resulting hists in a ROOT file
            out_file = uproot.recreate(outputs["shapes"].fn)
            for key, h in file.items():
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

                out_file[key] = uproot.from_pyroot(h_rebin)
