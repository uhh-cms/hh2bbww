# coding: utf-8

"""
Column production method
"""

import law

from columnflow.production import Producer, producer
from columnflow.selection import SelectionResult
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
# from columnflow.production.util import attach_coffea_behavior
from hbw.util import has_four_vec

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
    "Forwardjet": {
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
    # "gen_hbw_decay": {
    #     "type_name": "Muon",  # is there some other collection?
    #     "check_attr": "metric_table",
    #     "skip_fields": "pdgId",
    # },
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
    # collections are only created when needed by someone else
    uses={attach_coffea_behavior},
    # no produces since we do not want to permanently produce columns
)
def prepare_objects(self: Producer, events: ak.Array, results: SelectionResult = None, **kwargs) -> ak.Array:
    """
    Producer that defines objects in a convenient way.
    When used as part of `SelectEvents`, be careful since it may override the original NanoAOD columns.
    """
    # apply results if given to create new collections
    events = apply_object_results(events, results)

    # coffea behavior for relevant objects
    events = self[attach_coffea_behavior](events, collections=custom_collections, **kwargs)

    if (
        "Lepton" not in events.fields and
        has_four_vec(events, "Muon") and
        has_four_vec(events, "Electron")
    ):
        # combine Electron and Muon into a single object (Lepton)
        lepton = ak.with_name(ak.concatenate([events.Muon, events.Electron], axis=-1), "PtEtaPhiMLorentzVector")
        events = set_ak_column(events, "Lepton", lepton[ak.argsort(lepton.pt, ascending=False)])

    # transform MET into 4-vector
    met_name = self.config_inst.x.met_name
    if met_name in events.fields:
        events[met_name] = set_ak_column(events[met_name], "mass", 0)
        events[met_name] = set_ak_column(events[met_name], "eta", 0)
        events[met_name] = ak.with_name(events[met_name], "PtEtaPhiMLorentzVector")

    return events
