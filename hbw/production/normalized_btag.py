# coding: utf-8

"""
Producers for phase-space normalized btag scale factor weights.
"""

from __future__ import annotations

import law

from columnflow.production import Producer, producer
from hbw.production.btag import HBWTMP_btag_weights
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")


@producer(
    uses={
        HBWTMP_btag_weights.PRODUCES, "process_id", "Jet.{pt,eta,phi}", "njet", "ht", "nhf",
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
        for weight_route in self[HBWTMP_btag_weights].produced_columns:
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
    # NOTE: self[HBWTMP_btag_weights].produced_columns is empty during the `init`, therefore changed to `post_init`
    # this means that running this Producer directly on command line would not be triggered due to empty produces
    # during task initialization
    for weight_route in self[HBWTMP_btag_weights].produced_columns:
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
