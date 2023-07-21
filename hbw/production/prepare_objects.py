# coding: utf-8

"""
Column production method
"""

import law

from columnflow.production import Producer, producer
from columnflow.selection import SelectionResult
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, optional_column
from columnflow.production.util import attach_coffea_behavior
# from columnflow.production.util import attach_coffea_behavior

from hbw.util import four_vec

ak = maybe_import("awkward")
coffea = maybe_import("coffea")
np = maybe_import("numpy")
maybe_import("coffea.nanoevents.methods.nanoaod")

logger = law.logger.get_logger(__name__)

custom_collections = {
    "Bjet": {
        "type_name": "Jet",
        "check_attr": "metric_table",
        "skip_fields": "*Idx*G",
    },
    "Lightjet": {
        "type_name": "Jet",
        "check_attr": "metric_table",
        "skip_fields": "*Idx*G",
    },
    "VBFJet": {
        "type_name": "Jet",
        "check_attr": "metric_table",
        "skip_fields": "*Idx*G",
    },
    "HbbSubJet": {
        "type_name": "Jet",
        "check_attr": "metric_table",
        "skip_fields": "*Idx*G",
    },
    "HbbJet": {
        "type_name": "FatJet",
        "check_attr": "metric_table",
        "skip_fields": "*Idx*G",
    },
    "Lepton": {
        "type_name": "Muon",  # is there some other collection?
        "check_attr": "metric_table",
        "skip_fields": "*Idx*G",
    },
    "VetoLepton": {
        "type_name": "Muon",  # is there some other collection?
        "check_attr": "metric_table",
        "skip_fields": "*Idx*G",
    },
}


def apply_object_results(events: ak.Array, results: SelectionResult = None):
    """
    Small helper function to apply all object masks to clean collections or create new ones;
    Does not apply any event masks.
    """
    if not results:
        return events

    # loop through all object selection, go through their masks and create new collections if required
    for src_name in results.objects.keys():
        # get all destination collections, handling those named identically to the source collection last
        dst_names = list(results.objects[src_name].keys())
        if src_name in dst_names:
            dst_names.remove(src_name)
            dst_names.append(src_name)
        for dst_name in dst_names:
            object_mask = results.objects[src_name][dst_name]
            dst_collection = events[src_name][object_mask]
            events = set_ak_column(events, dst_name, dst_collection)

    return events


@producer(
    # This producer only requires 4-vector properties,
    # but all columns required by the main Selector/Producer will be considered
    uses=(
        {attach_coffea_behavior} | four_vec({"Electron", "Muon", "MET"}) |
        optional_column(four_vec("Jet", "btagDeepFlavB"))
    ),
    # no produces since we do not want to permanently produce columns
)
def prepare_objects(self: Producer, events: ak.Array, results: SelectionResult = None, **kwargs) -> ak.Array:
    """
    Producer that defines objects in a convenient way.
    When used as part of `SelectEvents`, be careful since it may override the original NanoAOD columns.
    """
    # apply results if given to create new collections
    events = apply_object_results(events, results)

    if "Bjet" not in events.fields and "Jet" in events.fields:
        logger.warning("Bjet collection is missing: will be defined using the Jet collection")
        # define b-jets as the two b-score leading jets, b-score sorted
        bjet_indices = ak.argsort(events.Jet.btagDeepFlavB, axis=-1, ascending=False)
        events = set_ak_column(events, "Bjet", events.Jet[bjet_indices[:, :2]])

    if "Lightjet" not in events.fields and "Jet" in events.fields:
        logger.warning("Lightjet collection is missing: will be defined using the Jet collection")
        # define lightjets as all non b-jets, pt-sorted
        bjet_indices = ak.argsort(events.Jet.btagDeepFlavB, axis=-1, ascending=False)
        lightjets = events.Jet[bjet_indices[:, 2:]]
        lightjets = lightjets[ak.argsort(lightjets.pt, axis=-1, ascending=False)]
        events = set_ak_column(events, "Lightjet", lightjets)

    if "VetoLepton" not in events.fields and "VetoElectron" in events.fields and "VetoMuon" in events.fields:
        # combine VetoElectron and VetoMuon into a single object (VetoLepton)
        lepton_fields = set(events.VetoMuon.fields).intersection(events.VetoElectron.fields)
        veto_lepton = ak.concatenate([
            ak.zip({f: events.VetoMuon[f] for f in lepton_fields}),
            ak.zip({f: events.VetoElectron[f] for f in lepton_fields}),
        ], axis=-1)
        events = set_ak_column(events, "VetoLepton", veto_lepton[ak.argsort(veto_lepton.pt, ascending=False)])

    if "Lepton" not in events.fields and "Electron" in events.fields and "Muon" in events.fields:
        # combine Electron and Muon into a single object (Lepton)
        lepton_fields = set(events.Muon.fields).intersection(events.Electron.fields)
        lepton = ak.concatenate([
            ak.zip({f: events.Muon[f] for f in lepton_fields}),
            ak.zip({f: events.Electron[f] for f in lepton_fields}),
        ], axis=-1)
        events = set_ak_column(events, "Lepton", lepton[ak.argsort(lepton.pt, ascending=False)])

    # coffea behavior for relevant objects
    events = self[attach_coffea_behavior](events, collections=custom_collections, **kwargs)

    # transform MET into 4-vector
    events["MET"] = set_ak_column(events.MET, "mass", 0)
    events["MET"] = set_ak_column(events.MET, "eta", 0)
    events["MET"] = ak.with_name(events["MET"], "PtEtaPhiMLorentzVector")

    return events
