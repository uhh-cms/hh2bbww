# coding: utf-8

"""
Column production method
"""

from columnflow.production import Producer, producer
from columnflow.selection import SelectionResult
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
# from columnflow.production.util import attach_coffea_behavior

ak = maybe_import("awkward")
coffea = maybe_import("coffea")
np = maybe_import("numpy")
maybe_import("coffea.nanoevents.methods.nanoaod")


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
        {
            # attach_coffea_behavior,  # TODO use
            "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB",
            "MET.pt", "MET.phi",
        } |
        set(
            f"{obj}.{var}"
            for obj in ("Electron", "Muon", "FatJet")
            for var in ("pt", "eta", "phi", "mass")
        )
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

    # check that main objects are present
    if not {"Jet", "Muon", "Electron", "MET"}.intersection(events.fields):
        raise Exception(f"Missing object in event fields {events.fields}")

    if "Bjet" not in events.fields:
        # define b-jets as the two b-score leading jets, b-score sorted
        bjet_indices = ak.argsort(events.Jet.btagDeepFlavB, axis=-1, ascending=False)
        events = set_ak_column(events, "Bjet", events.Jet[bjet_indices[:, :2]])

    if "Lightjet" not in events.fields:
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

    # transform MET into 4-vector
    events["MET"] = set_ak_column(events.MET, "mass", 0)
    events["MET"] = set_ak_column(events.MET, "eta", 0)

    # 4-vector behavior for relevant objects
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    for obj in ["Jet", "Bjet", "Lightjet", "FatJet", "Lepton", "VetoLepton", "MET"]:
        if obj in events.fields:
            events[obj] = ak.with_name(events[obj], "PtEtaPhiMLorentzVector")

    return events
