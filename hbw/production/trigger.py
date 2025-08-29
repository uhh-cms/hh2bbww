# coding: utf-8

"""
Trigger related event weights.
"""

from __future__ import annotations

import functools
import law

# from dataclasses import dataclass

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, fill_at
from columnflow.production.cms.muon import muon_weights, MuonSFConfig

from hbw.production.prepare_objects import prepare_objects

np = maybe_import("numpy")
ak = maybe_import("awkward")


set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
fill_at_f32 = functools.partial(fill_at, value_type=np.float32)


# @dataclass
# class TriggerSFConfig:


from hbw.categorization.categories import catid_2e, catid_2mu, catid_emu


# NOTE: dummy up/down variation at the moment
trigger_sf_config = {
    "trigger_sf_ee": {
        "corr_keys": {
            "nominal": "sf_ee_trg_lepton0_pt-trg_lepton1_pt-trig_ids",
            "up": "sf_ee_trg_lepton0_pt-trg_lepton1_pt-trig_ids_up",
            "down": "sf_ee_trg_lepton0_pt-trg_lepton1_pt-trig_ids_down",
        },
        "category": catid_2e,
    },
    "trigger_sf_mm": {
        "corr_keys": {
            "nominal": "sf_mm_trg_lepton0_pt-trg_lepton1_pt-trig_ids",
            "up": "sf_mm_trg_lepton0_pt-trg_lepton1_pt-trig_ids_up",
            "down": "sf_mm_trg_lepton0_pt-trg_lepton1_pt-trig_ids_down",
        },
        "category": catid_2mu,
    },
    "trigger_sf_mixed": {
        "corr_keys": {
            "nominal": "sf_mixed_trg_lepton0_pt-trg_lepton1_pt-trig_ids",
            "up": "sf_mixed_trg_lepton0_pt-trg_lepton1_pt-trig_ids_up",
            "down": "sf_mixed_trg_lepton0_pt-trg_lepton1_pt-trig_ids_down",
        },
        "category": catid_emu,
    },
}


@producer(
    uses={"{Electron,Muon}.{pt,eta,phi,mass}", prepare_objects},
    # produces in the init
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    trigger_sf_config=trigger_sf_config,
    weight_name="trigger_weight",
)
def dl_trigger_weights(
    self: Producer,
    events: ak.Array,
    trigger_mask: ak.Array | type(Ellipsis) = Ellipsis,
    **kwargs,
) -> ak.Array:
    """
    Creates trigger weights using custom trigger SF jsons.
    """

    events = self[prepare_objects](events, **kwargs)

    variable_map = {
        "mli_lep_pt": events.Lepton[:, 0].pt,
        "mli_lep2_pt": events.Lepton[:, 1].pt,
        "trg_lepton0_pt": events.Lepton[:, 0].pt,
        "trg_lepton1_pt": events.Lepton[:, 1].pt,
    }

    full_mask = ak.zeros_like(events.event, dtype=bool)

    for key, corr_set in self.correction_sets.items():
        sf_config = self.trigger_sf_config[key]

        categorizer = sf_config["category"]
        events, mask = self[categorizer](events, **kwargs)

        # ensure that no event is assigned to multiple categories
        if ak.any(mask & full_mask):
            raise Exception(f"Overlapping categories in {dl_trigger_weights.cls_name}")
        full_mask = mask | full_mask

        for sys, corr_key in sf_config["corr_keys"].items():
            sysfix = "" if sys == "nominal" else f"_{sys}"
            col_name = f"{self.weight_name}{sysfix}"
            if col_name not in events.fields:
                events = set_ak_column_f32(events, col_name, ak.ones_like(events.event))
            corr = corr_set[corr_key]
            inputs = [variable_map[inp.name] for inp in corr.inputs]

            _sf = corr.evaluate(*inputs)

            events = fill_at_f32(
                ak_array=events,
                where=mask,
                route=col_name,
                value=_sf,
            )

    return events


@dl_trigger_weights.requires
def dl_trigger_weights_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@dl_trigger_weights.setup
def dl_trigger_weights_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
) -> None:
    bundle_files = reqs["external_files"].files

    # create the corrector
    import correctionlib
    import json
    self.correction_sets = {}
    for key, sf_config in self.trigger_sf_config.items():
        target = bundle_files[key]
        correction_set = correctionlib.CorrectionSet.from_string(
            json.loads(target.load(formatter="gzip")),
        )
        self.correction_sets[key] = correction_set


@dl_trigger_weights.init
def dl_trigger_weights_init(self: Producer, **kwargs) -> None:
    weight_name = self.weight_name
    self.produces |= {weight_name, f"{weight_name}_up", f"{weight_name}_down"}

    for key, sf_config in self.trigger_sf_config.items():
        self.uses.add(sf_config["category"])


muon_trigger_weights = muon_weights.derive("muon_trigger_weights", cls_dict={
    "weight_name": "muon_trigger_weight",
    "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_trigger_sf_names)),
})


@producer(
    uses={muon_trigger_weights, "Muon.{pt,eta,phi,mass}", prepare_objects},
)
def sl_trigger_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer that calculates the single lepton trigger weights.
    NOTE: this only includes the trigger weights from the muon channel. They should be combined with
    the electron trigger weights in this producer.
    """
    if not self.config_inst.has_aux("muon_trigger_sf_names"):
        raise Exception(f"In {sl_trigger_weights.__name__}: missing 'muon_trigger_sf_names' in config")

    # compute muon trigger SF weights (NOTE: trigger SFs are only defined for muons with
    # pt > 26 GeV, so create a copy of the events array with with all muon pt < 26 GeV set to 26 GeV)
    trigger_sf_events = set_ak_column_f32(events, "Muon.pt", ak.where(events.Muon.pt > 26., events.Muon.pt, 26.))
    trigger_sf_events = self[muon_trigger_weights](trigger_sf_events, **kwargs)
    for route in self[muon_trigger_weights].produced_columns:
        events = set_ak_column_f32(events, route, route.apply(trigger_sf_events))
    # memory cleanup
    del trigger_sf_events

    return events
