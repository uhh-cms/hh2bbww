# coding: utf-8

"""
Tasks related to the creation and modification of datacards for inference purposes.
"""

import law

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin, InferenceModelMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.cms.inference import CreateDatacards
from columnflow.util import dev_sandbox, maybe_import

array = maybe_import("array")


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


def print_if(msg, i, N=100):
    if i % N == 0:
        print(msg)


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
        print_if(f" ========== Bin {i} of {N_bins_input}, {N_events} events", i)
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


def print_hist(hist, max_bins: int = 20):
    print("Printing bin number, lower edge and bin content")
    for i in range(0, hist.GetNbinsX() + 2):
        if i > 20:
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

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateDatacards=CreateDatacards,
    )

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

        return {
            "card": self.target(basename("datacard_rebin", "txt")),
            "shapes": self.target(basename("shapes_rebin", "root")),
        }

    def run(self):
        inputs = self.input()
        outputs = self.output()

        import uproot
        from ROOT import TH1

        root_filename = inputs["datacards"]["shapes"].fn

        # copy the datacard (NOTE: we might want to modify the name of the shapes file)
        inputs["datacards"]["card"].copy_to(outputs["card"])
        # datacard = inputs["datacards"]["card"].fn

        with uproot.open(root_filename) as file:
            print(f"File keys: {file.keys()}")
            # determine which histograms are present
            cat_names, proc_names, syst_names = get_cat_proc_syst_names(file)

            if len(cat_names) != 1:
                raise Exception("Expected 1 category per file")

            cat_name = list(cat_names)[0]

            nominal_hists = {
                proc_name: file[f"{cat_name}/{proc_name}"].to_pyroot()
                for proc_name in proc_names
            }

            # now determine the rebinning
            if "ggHH_kl_1_kt_1_sl_hbbhww" in cat_name:
                # HH signal region --> flat in signal
                N_bins = 10
                hist = nominal_hists["ggHH_kl_1_kt_1_sl_hbbhww"]
            else:
                # background region --> flat in all backgrounds
                N_bins = 3
                hists = [
                    nominal_hists[proc_name]
                    for proc_name in nominal_hists.keys()
                    if "ggHH" not in proc_name
                ]
                hist = hists[0]
                for h in hists[1:]:
                    hist += h

            print(f"Finding rebin values for category {cat_name}")
            rebin_values = get_rebin_values(hist, N_bins)

            # loop over all histograms and store in a ROOT file
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
                out_file[key] = uproot.from_pyroot(h_rebin)
