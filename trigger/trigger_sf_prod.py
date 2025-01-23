# coding: utf-8

"""
Producers to load trigger scale factors, wip
"""

from __future__ import annotations

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, InsertableDict
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")


@producer(
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_trigger_file=(lambda self, external_files: external_files.ele_trigger_sf),
    version=3,
)
def trigger_sf_weights(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Creates trigger scale factor weights using correctionlib. Requires external file.
    """

    var_map = {
        "ht": events.ht,
        "electron_pt": events.Electron.pt,
    }

    sf = self.trigger_sf_corrector

    inputs = [var_map[inp.name] for inp in sf.inputs]
    sf_weights = sf.evaluate(*inputs)

    events = set_ak_column(events, "trigger_sf_weights", sf_weights)

    return events


@trigger_sf_weights.init
def trigger_sf_weights_init(self: Producer) -> None:

    if hasattr(self, "dataset_inst") and self.dataset_inst.is_data:
        return

    self.produces = {"trigger_sf_weights"}
    self.uses = {"Electron.pt", "ht", "HLT.*"}


@trigger_sf_weights.requires
def trigger_sf_weights_requires(self: Producer, reqs: dict) -> None:

    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@trigger_sf_weights.setup
def trigger_sf_weights_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:

    bundle = reqs["external_files"]

    # create the corrector
    import correctionlib
    import json
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    correction_set = correctionlib.CorrectionSet.from_string(
        json.loads(self.get_trigger_file(bundle.files).load(formatter="gzip")),
    )
    self.trigger_sf_corrector = correction_set["sf_Ele30_WPTight_Gsf_electron_pt-trig_bits_e"]
