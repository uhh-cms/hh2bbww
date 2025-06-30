# coding: utf-8

"""
Column production method
"""

import law

from columnflow.production import Producer, producer
from columnflow.selection import SelectionResult
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, has_ak_column
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


def combine_collections(events: ak.Array, src_names: list[str], dst_name: str, sort_by: str = "pt") -> ak.Array:
    """
    Combine multiple collections into a single collection, sorting by `sort_by`.
    :param events: ak.Array
    :param src_names: list of strings with the names of the source collections to combine
    :param dst_name: string with the name of the destination collection
    :param sort_by: string with the name of the field to sort by
    :return: events with the combined collection added
    """
    if dst_name in events.fields:
        logger.warning(f"Collection {dst_name} already exists, skipping combination")
        return events
    if all(has_four_vec(events, src_name) for src_name in src_names):
        combined = ak.with_name(
            ak.concatenate([events[name] for name in src_names], axis=-1),
            "PtEtaPhiMLorentzVector",
        )
        if sort_by:
            combined = combined[ak.argsort(combined[sort_by], ascending=False)]
        events = set_ak_column(events, dst_name, combined)
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

    if has_four_vec(events, "ForwardJet"):
        # apply pt cut to forward jets
        # TODO: decide on forward jet cuts
        forward_jet_mask = ak.where(
            (abs(events["ForwardJet"].eta) > 2.5) & (abs(events["ForwardJet"].eta) < 3.0),
            (events["ForwardJet"].pt > 50),
            (events["ForwardJet"].pt > 30),
        )
        events = set_ak_column(events, "ForwardJet", events["ForwardJet"][forward_jet_mask])

    elif "ForwardJet" in events.fields:
        raise ValueError("ForwardJet arguments incomplete.")

    # combine collections if necessary and possible
    events = combine_collections(events, ["Muon", "Electron"], "Lepton", sort_by="pt")
    events = combine_collections(events, ["ForwardJet", "Lightjet"], "VBFCandidateJet", sort_by="pt")
    events = combine_collections(events, ["Jet", "ForwardJet"], "InclJet", sort_by="pt")

    if has_ak_column(events, "FatJet.particleNet_XbbVsQCD"):
        # TODO: change to particleNetWithMass_HbbvsQCD after next selection run
        # NOTE: this sometimes produces errors due to NaN values in the column
        events = combine_collections(events, ["FatJet"], "FatBjet", sort_by="particleNet_XbbVsQCD")

    # transform MET into 4-vector
    met_name = self.config_inst.x.met_name
    if met_name in events.fields:
        events[met_name] = set_ak_column(events[met_name], "mass", 0)
        events[met_name] = set_ak_column(events[met_name], "eta", 0)
        events[met_name] = ak.with_name(events[met_name], "PtEtaPhiMLorentzVector")

    if "RecoilCorrMET" in events.fields:
        # transform RecoilCorrMET into 4-vector
        events["RecoilCorrMET"] = set_ak_column(events["RecoilCorrMET"], "mass", 0)
        events["RecoilCorrMET"] = set_ak_column(events["RecoilCorrMET"], "eta", 0)
        events["RecoilCorrMET"] = ak.with_name(events["RecoilCorrMET"], "PtEtaPhiMLorentzVector")

    return events
