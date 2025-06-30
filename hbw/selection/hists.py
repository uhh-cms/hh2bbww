# coding: utf-8

"""
Stat-related methods.
"""

import law
import order as od

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.production.cms.btag import btag_weights
from hbw.production.weights import event_weights_to_normalize
from columnflow.columnar_util import set_ak_column

from columnflow.util import maybe_import
from hbw.util import has_tag, IF_MC
from columnflow.hist_util import create_hist_from_variables

np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")


@selector(
    uses={increment_stats, event_weights_to_normalize, IF_MC("Jet.hadronFlavour")},
    produces=IF_MC({"ht", "njet", "nhf"}),
)
def hbw_selection_hists(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    hists: dict,
    **kwargs,
) -> ak.Array:
    """
    Main selector to create and fill histograms for weight normalization.
    """
    # collect important information from the results
    no_weights = ak.values_astype(ak.Array(np.ones(len(events))), np.int64)
    event_masks = {
        "Initial": results.steps.no_sel_mask,
        "selected_no_bjet": results.steps.all_but_bjet,
        "selected": results.event,
    }
    njet = results.x.n_central_jets
    ht = results.x.ht

    # store ht, njet, and nhf for consistency checks
    events = set_ak_column(events, "ht", ht)
    events = set_ak_column(events, "njet", njet)

    if self.dataset_inst.is_mc:
        hadron_flavour = events.Jet[results.objects.Jet.Jet].hadronFlavour
        nhf = ak.sum(hadron_flavour == 5, axis=1) + ak.sum(hadron_flavour == 4, axis=1)
        events = set_ak_column(events, "nhf", nhf)

    # weight map definition
    weight_map = {
        # "num" operations
        "num_events": no_weights,
    }

    if self.dataset_inst.is_mc:
        # "sum" operations
        weight_map["sum_mc_weight"] = events.mc_weight  # weights of all events

        weight_columns = set(self[event_weights_to_normalize].produced_columns)
        if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
            # btag_weights are not produced and therefore need some manual care
            weight_columns |= set(self[btag_weights].produced_columns)

        weight_columns = sorted([col.string_nano_column for col in weight_columns])

        # mc weight times correction weight (with variations) without any selection
        for name in weight_columns:
            if "weight" not in name:
                # skip non-weight columns here
                continue

            # TODO: decide whether to keep mc_weight * weight or just weight
            weight_map[f"sum_mc_weight_{name}"] = events.mc_weight * events[name]
            # weight_map[f"sum_{name}"] = events[name]

    # initialize histograms if not already done
    # (NOTE: this only works as long as this is the only selector that adds histograms)
    if not hists:
        for key, weight in weight_map.items():
            if "btag_weight" not in key:
                hists[key] = create_hist_from_variables(self.steps_variable)
                hists[f"{key}_per_process"] = create_hist_from_variables(self.steps_variable, self.process_variable)
            if key == "sum_mc_weight" or "btag_weight" in key:
                hists[f"{key}_per_process_ht_njet_nhf"] = create_hist_from_variables(
                    self.steps_variable,
                    self.process_variable,
                    self.ht_variable,
                    self.njet_variable,
                    self.nhf_variable,
                )

    # fill histograms
    for key, weight in weight_map.items():
        for step, mask in event_masks.items():
            # TODO: can I fill with single value instead of array of strings?
            step_arr = np.array([step] * ak.sum(mask))
            if "btag_weight" not in key:
                hists[key].fill(steps=step_arr, weight=weight[mask])
                hists[f"{key}_per_process"].fill(steps=step_arr, process=events.process_id[mask], weight=weight[mask])
            if step == "selected_no_bjet" and (key == "sum_mc_weight" or "btag_weight" in key):
                # to reduce computing time, only fill the selected_no_bjet mask for btag weights
                hists[f"{key}_per_process_ht_njet_nhf"].fill(
                    steps=step_arr,
                    process=events.process_id[mask],
                    ht=ht[mask],
                    njet=njet[mask],
                    nhf=nhf[mask],
                    weight=weight[mask],
                )

    return events


@hbw_selection_hists.setup
def hbw_selection_hists_setup(
    self: Selector, task: law.Task, reqs: dict, inputs: dict, reader_targets: dict,
) -> None:
    self.process_variable = od.Variable(
        name="process",
        expression="process_id",
        aux={"axis_type": "intcategory"},
    )
    self.steps_variable = od.Variable(
        name="steps",
        aux={"axis_type": "strcategory"},
    )
    self.ht_variable = od.Variable(
        name="ht",
        binning=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1450, 1700, 2400],
        aux={"axis_type": "variable"},
    )
    self.njet_variable = od.Variable(
        name="njet",
        binning=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        aux={
            "axis_type": "integer",
            "axis_kwargs": {"growth": True},
        },
    )
    self.nhf_variable = od.Variable(
        name="nhf",
        binning=[0, 1, 2, 3, 4],
        aux={
            "axis_type": "integer",
            "axis_kwargs": {"growth": True},
        },
    )


@hbw_selection_hists.init
def hbw_selection_hists_init(self: Selector) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    if self.dataset_inst.is_data:
        return

    if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {btag_weights}

    if self.dataset_inst.is_mc:
        self.uses |= {"mc_weight", "Jet.hadronFlavour"}
        self.produces |= {"ht", "njet", "nhf"}
