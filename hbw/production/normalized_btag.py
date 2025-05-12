# coding: utf-8

"""
Producers for phase-space normalized btag scale factor weights.
"""

from __future__ import annotations

import law

from columnflow.production import Producer, producer
from columnflow.production.cms.btag import btag_weights
from columnflow.util import maybe_import, safe_div
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")


@producer(
    uses={
        btag_weights.PRODUCES, "process_id", "Jet.{pt,eta,phi}", "njet", "ht", "nhf",
    },
    # produced columns are defined in the init function below
    mc_only=True,
    modes=["ht_njet_nhf"],
    # modes=["ht_njet_nhf", "ht_njet", "njet", "ht"],
    from_file=False,
)
def normalized_btag_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    variable_map = {
        # NOTE: might be cleaner to use the ht and njet reconstructed during the selection (and also compare?)
        "ht": ak.sum(events.Jet.pt, axis=1),
        "njet": ak.num(events.Jet.pt, axis=1),
        "nhf": events.nhf,
    }

    # sanity check
    for var in ("ht", "njet"):
        consistency_check = np.isclose(events[var], variable_map[var], rtol=0.0001)
        if not ak.all(consistency_check):
            raise ValueError(f"Variable {var} is not consistent between before and after event selection")

    for mode in self.modes:
        if mode not in ("ht_njet_nhf", "ht_njet", "njet", "ht"):
            raise NotImplementedError(
                f"Normalization mode {mode} not implemented (see hbw.tasks.corrections.GetBtagNormalizationSF)",
            )
        for weight_route in self[btag_weights].produced_columns:
            weight_name = weight_route.string_column
            if not weight_name.startswith("btag_weight"):
                continue

            correction_key = f"{mode}_{weight_name}"
            if correction_key not in set(self.correction_set.keys()):
                raise KeyError(f"Missing scale factor for {correction_key}")

            sf = self.correction_set[correction_key]
            inputs = [variable_map[inp.name] for inp in sf.inputs]

            norm_weight = sf.evaluate(*inputs)
            norm_weight = norm_weight * events[weight_name]
            events = set_ak_column(events, f"normalized_{mode}_{weight_name}", norm_weight, value_type=np.float32)

    return events


@normalized_btag_weights.post_init
def normalized_btag_weights_post_init(self: Producer, task: law.Task) -> None:
    # NOTE: self[btag_weights].produced_columns is empty during the `init`, therefore changed to `post_init`
    # this means that running this Producer directly on command line would not be triggered due to empty produces
    # during task initialization
    for weight_route in self[btag_weights].produced_columns:
        weight_name = weight_route.string_column
        if not weight_name.startswith("btag_weight"):
            continue
        for mode in self.modes:
            self.produces.add(f"normalized_{mode}_{weight_name}")


@normalized_btag_weights.requires
def normalized_btag_weights_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    from hbw.tasks.corrections import GetBtagNormalizationSF
    reqs["btag_renormalization_sf"] = GetBtagNormalizationSF.req(task)


normalized_btag_weights_full = normalized_btag_weights.derive("normalized_btag_weights_full", cls_dict=dict(
    modes=["ht_njet_nhf", "ht_njet", "njet", "ht"],
))


@normalized_btag_weights.setup
def normalized_btag_weights_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
) -> None:
    # create the corrector
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    if self.from_file:
        # used when the correction is stored as a JSON dict
        self.correction_set = correctionlib.CorrectionSet.from_file(
            inputs["btag_renormalization_sf"]["btag_renormalization_sf"].fn,
        )
    else:
        # used when correction is stored as a JSON string
        self.correction_set = correctionlib.CorrectionSet.from_string(
            inputs["btag_renormalization_sf"]["btag_renormalization_sf"].load(formatter="json"),
        )


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
def normalized_btag_weights_from_json_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req(
        task,
        branch=-1,
    )


@normalized_btag_weights_from_json.setup
def normalized_btag_weights_from_json_setup(
    self: Producer, task: law.Task, reqs: dict, inputs: dict, reader_targets: law.util.InsertableDict,
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
