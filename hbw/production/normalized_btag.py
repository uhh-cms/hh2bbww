# coding: utf-8

"""
Producers for phase-space normalized btag scale factor weights.
"""

from __future__ import annotations

from columnflow.production import Producer, producer
from columnflow.production.cms.btag import btag_weights
from columnflow.util import maybe_import, safe_div, InsertableDict
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")


@producer(
    uses={
        btag_weights.PRODUCES, "process_id", "Jet.pt",
    },
    # produced columns are defined in the init function below
    mc_only=True,
)
def normalized_btag_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    variable_map = {
        # NOTE: might be cleaner to use the ht and njet reconstructed during the selection (and also compare?)
        "ht": ak.sum(events.Jet.pt, axis=1),
        "n_jets": ak.num(events.Jet.pt, axis=1),
    }

    for weight_name in self[btag_weights].produces:
        if not weight_name.startswith("btag_weight"):
            continue

        if weight_name not in self.sf_map:
            raise KeyError(f"Missing scale factor for {weight_name}")

        sf = self.sf_map[weight_name]
        inputs = [variable_map[inp.name] for inp in sf.inputs]

        norm_weight = sf.evaluate(*inputs)
        norm_weight = norm_weight * events[weight_name]
        events = set_ak_column(events, f"normalized_{weight_name}", norm_weight)

    return events


@normalized_btag_weights.init
def normalized_btag_weights_init(self: Producer) -> None:
    for weight_name in self[btag_weights].produces:
        if not weight_name.startswith("btag_weight"):
            continue
        self.produces.add(f"normalized_{weight_name}")


@normalized_btag_weights.requires
def normalized_btag_weights_requires(self: Producer, reqs: dict) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req(
        self.task,
        tree_index=0,
        branch=-1,
        _exclude=MergeSelectionStats.exclude_params_forest_merge,
    )


@normalized_btag_weights.setup
def normalized_btag_weights_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    """
    Setup function for the normalized btag weights producer, which extracts the normalization factors
    from the selection histograms. Histograms are identified via key following the naming format
    `sum_mc_weight_btag_weight_{weight_name}_per_process_ht_njet`.
    """
    # load the selection hists
    hists = inputs["selection_stats"]["collection"][0]["hists"].load(formatter="pickle")

    # initialize the scale factor map
    self.sf_map = {}

    den = hists["sum_mc_weight_per_process_ht_njet"][{"process": sum, "steps": "selected_no_bjet"}].values()

    for key in hists.keys():
        if not key.startswith("sum_mc_weight_btag_weight") or not key.endswith("_per_process_ht_njet"):
            continue

        # extract the weight name
        weight_name = key.replace("sum_mc_weight_", "").replace("_per_process_ht_njet", "")

        # create the scale factor histogram
        h = hists[key][{"process": sum}]

        num = h[{"steps": "selected_no_bjet"}].values()

        # calculate the scale factor and store it as a correctionlib evaluator
        sf = np.where(
            (num > 0) & (den > 0),
            num / den,
            1.0,
        )

        sfhist = hist.Hist(*h.axes[1:], data=sf)
        sfhist.name = f"{weight_name}_renormalization"
        sfhist.label = "out"

        import correctionlib.convert
        btag_renormalization = correctionlib.convert.from_histogram(sfhist)
        btag_renormalization.description = f"{weight_name} re-normalization"

        # set overflow bins behavior (default is to raise an error when out of bounds)
        # NOTE: claming seems to not work for int axes. Hopefully the number of jets considered to
        # create these SFs is always large enough to not hit the overflow bin.
        btag_renormalization.data.flow = "clamp"

        # store the evaluator
        self.sf_map[weight_name] = btag_renormalization.to_evaluator()


@producer(
    uses={
        btag_weights.PRODUCES, "process_id", "Jet.pt",
    },
    # produced columns are defined in the init function below
    mc_only=True,
)
def normalized_btag_weights_from_json(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    for weight_name in self[btag_weights].produces:
        if not weight_name.startswith("btag_weight"):
            continue

        # create a weight vectors starting with ones for both weight variations, i.e.,
        # nomalization per pid and normalization per pid and jet multiplicity
        norm_weight_per_pid = np.ones(len(events), dtype=np.float32)
        norm_weight_per_pid_njet = np.ones(len(events), dtype=np.float32)

        # fill weights with a new mask per unique process id (mostly just one)
        for pid in self.unique_process_ids:
            pid_mask = events.process_id == pid
            # single value
            norm_weight_per_pid[pid_mask] = self.ratio_per_pid[weight_name][pid]
            # lookup table
            n_jets = ak.num(events[pid_mask].Jet.pt, axis=1)
            norm_weight_per_pid_njet[pid_mask] = self.ratio_per_pid_njet[weight_name][pid][n_jets]

        # multiply with actual weight
        norm_weight_per_pid = norm_weight_per_pid * events[weight_name]
        norm_weight_per_pid_njet = norm_weight_per_pid_njet * events[weight_name]

        # store them
        events = set_ak_column(events, f"normalized_{weight_name}", norm_weight_per_pid)
        events = set_ak_column(events, f"normalized_njet_{weight_name}", norm_weight_per_pid_njet)

    return events


@normalized_btag_weights_from_json.init
def normalized_btag_weights_from_json_init(self: Producer) -> None:
    for weight_name in self[btag_weights].produces:
        if not weight_name.startswith("btag_weight"):
            continue

        self.produces.add(f"normalized_{weight_name}")
        self.produces.add(f"normalized_njet_{weight_name}")


@normalized_btag_weights_from_json.requires
def normalized_btag_weights_from_json_requires(self: Producer, reqs: dict) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req(
        self.task,
        tree_index=0,
        branch=-1,
        _exclude=MergeSelectionStats.exclude_params_forest_merge,
    )


@normalized_btag_weights_from_json.setup
def normalized_btag_weights_from_json_setup(
    self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict,
) -> None:
    # load the selection stats
    stats = inputs["selection_stats"]["collection"][0]["stats"].load(formatter="json")

    # get the unique process ids in that dataset
    key = "sum_mc_weight_selected_no_bjet_per_process_and_njet"
    self.unique_process_ids = list(map(int, stats[key].keys()))

    # get the maximum numbers of jets
    max_n_jets = max(map(int, sum((list(d.keys()) for d in stats[key].values()), [])))

    # helper to get numerators and denominators
    def numerator_per_pid(pid):
        key = "sum_mc_weight_selected_no_bjet_per_process"
        return stats[key].get(str(pid), 0.0)

    def denominator_per_pid(weight_name, pid):
        key = f"sum_mc_weight_{weight_name}_selected_no_bjet_per_process"
        return stats[key].get(str(pid), 0.0)

    def numerator_per_pid_njet(pid, n_jets):
        key = "sum_mc_weight_selected_no_bjet_per_process_and_njet"
        d = stats[key].get(str(pid), {})
        return d.get(str(n_jets), 0.0)

    def denominator_per_pid_njet(weight_name, pid, n_jets):
        key = f"sum_mc_weight_{weight_name}_selected_no_bjet_per_process_and_njet"
        d = stats[key].get(str(pid), {})
        return d.get(str(n_jets), 0.0)

    # extract the ratio per weight and pid
    self.ratio_per_pid = {
        weight_name: {
            pid: safe_div(numerator_per_pid(pid), denominator_per_pid(weight_name, pid))
            for pid in self.unique_process_ids
        }
        for weight_name in self[btag_weights].produces
        if weight_name.startswith("btag_weight")
    }

    # extract the ratio per weight, pid and also the jet multiplicity, using the latter as in index
    # for a lookup table (since it naturally starts at 0)
    self.ratio_per_pid_njet = {
        weight_name: {
            pid: np.array([
                safe_div(numerator_per_pid_njet(pid, n_jets), denominator_per_pid_njet(weight_name, pid, n_jets))
                for n_jets in range(max_n_jets + 1)
            ])
            for pid in self.unique_process_ids
        }
        for weight_name in self[btag_weights].produces
        if weight_name.startswith("btag_weight")
    }
