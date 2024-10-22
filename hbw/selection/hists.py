# coding: utf-8

"""
Stat-related methods.
"""

import order as od

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.production.cms.btag import btag_weights
from hbw.production.weights import event_weights_to_normalize
from columnflow.columnar_util import set_ak_column

from columnflow.util import maybe_import
from hbw.util import has_tag
from hbw.hist_util import create_columnflow_hist

np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")


@selector(
    uses={increment_stats, event_weights_to_normalize, "Jet.hadronFlavour"},
    produces={"ht", "n_jets", "n_heavy_flavour"},
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
    no_weights = ak.Array(np.ones(len(events)))
    event_mask = results.event
    event_mask_no_bjet = results.steps.all_but_bjet
    n_jets = results.x.n_central_jets
    ht = results.x.ht

    hadron_flavour = events.Jet[results.objects.Jet.Jet].hadronFlavour
    n_heavy_flavour = ak.sum(hadron_flavour == 5, axis=1) + ak.sum(hadron_flavour == 4, axis=1)

    # store ht and n_jets for consistency checks
    events = set_ak_column(events, "ht", ht)
    events = set_ak_column(events, "n_jets", n_jets)
    events = set_ak_column(events, "n_heavy_flavour", n_heavy_flavour)

    # weight map definition
    weight_map = {
        # "num" operations
        "num_events": no_weights,
    }

    if self.dataset_inst.is_mc:
        weight_map["num_negative_weights"] = (events.mc_weight < 0)
        weight_map["num_pu_0"] = (events.pu_weight == 0)
        weight_map["num_pu_100"] = (events.pu_weight >= 100)
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
            weight_map[f"sum_mc_weight_{name}"] = events.mc_weight * events[name]
            weight_map[f"sum_{name}"] = events[name]

    for key, weight in weight_map.items():
        if key not in hists:
            hists[key] = create_columnflow_hist(self.steps_variable)
            hists[f"{key}_per_process"] = create_columnflow_hist(self.steps_variable, self.process_variable)
            if key == "sum_mc_weight" or "btag_weight" in key:
                hists[f"{key}_per_process_ht_njet"] = create_columnflow_hist(
                    self.steps_variable,
                    self.process_variable,
                    self.ht_variable,
                    self.n_jet_variable,
                )

                hists[f"{key}_per_process_ht_njet_nhf"] = create_columnflow_hist(
                    self.steps_variable,
                    self.process_variable,
                    self.ht_variable,
                    self.n_jet_variable,
                    self.n_heavy_flavour_variable,
                )

        for step, mask in (
            ("all", ak.ones_like(event_mask)),
            ("selected_no_bjet", event_mask_no_bjet),
            ("selected", event_mask),
        ):
            # TODO: how can I fill with single value instead of array?
            step_arr = [step] * ak.sum(mask)
            hists[key].fill(steps=step_arr, weight=weight[mask])
            hists[f"{key}_per_process"].fill(steps=step_arr, process=events.process_id[mask], weight=weight[mask])
            if key == "sum_mc_weight" or "btag_weight" in key:
                hists[f"{key}_per_process_ht_njet"].fill(
                    steps=step_arr,
                    process=events.process_id[mask],
                    ht=ht[mask],
                    n_jets=n_jets[mask],
                    weight=weight[mask],
                )

                hists[f"{key}_per_process_ht_njet_nhf"].fill(
                    steps=step_arr,
                    process=events.process_id[mask],
                    ht=ht[mask],
                    n_jets=n_jets[mask],
                    n_heavy_flavour=n_heavy_flavour[mask],
                    weight=weight[mask],
                )
    #TODO: check on limited stats that we can use this as it is
    return events


@hbw_selection_hists.setup
def hbw_selection_hists_setup(self: Selector, reqs: dict, inputs: dict, reader_targets: dict) -> None:
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
    self.n_jet_variable = od.Variable(
        name="n_jets",
        binning=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        aux={
            "axis_type": "integer",
            "axis_kwargs": {"growth": True},
        },
    )
    self.n_heavy_flavour_variable = od.Variable(
        name="n_heavy_flavour",
        binning=[0, 1, 2, 3, 4, 5, 6],
        aux={
            "axis_type": "integer",
            "axis_kwargs": {"growth": True},
        },
    )


@hbw_selection_hists.init
def hbw_selection_hists_init(self: Selector) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {btag_weights}

    if self.dataset_inst.is_mc:
        self.uses |= {"mc_weight"}
