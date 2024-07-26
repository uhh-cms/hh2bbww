# coding: utf-8

"""
Column production methods related to generic event weights.
"""

import functools

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.selection import SelectionResult
from columnflow.production import Producer, producer
from columnflow.production.cms.pileup import pu_weight
from columnflow.production.normalization import (
    normalization_weights,
    stitched_normalization_weights,
    stitched_normalization_weights_brs_from_processes,
)
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.production.cms.btag import btag_weights
from columnflow.production.cms.scale import murmuf_weights, murmuf_envelope_weights
from columnflow.production.cms.pdf import pdf_weights
from columnflow.production.cms.top_pt_weight import gen_parton_top, top_pt_weight
from hbw.production.gen_v import gen_v_boson, vjets_weight
from hbw.production.normalized_weights import normalized_weight_factory
from hbw.production.normalized_btag import normalized_btag_weights
from hbw.util import has_tag


np = maybe_import("numpy")
ak = maybe_import("awkward")


# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={gen_parton_top, gen_v_boson, pu_weight},
    produces={gen_parton_top, gen_v_boson, pu_weight},
    mc_only=True,
)
def event_weights_to_normalize(self: Producer, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers that are typically called as part of SelectEvents
    since it is required to normalize them before applying certain event selections.
    """

    # compute gen information that will later be needed for top pt reweighting
    if self.dataset_inst.has_tag("has_top"):
        events = self[gen_parton_top](events, **kwargs)

    # compute gen information that will later be needed for vector boson pt reweighting
    if self.dataset_inst.has_tag("is_v_jets"):
        events = self[gen_v_boson](events, **kwargs)

    # compute pu weights
    events = self[pu_weight](events, **kwargs)

    if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
        # compute btag SF weights (for renormalization tasks)
        events = self[btag_weights](events, jet_mask=results.aux["jet_mask"], **kwargs)

    # skip scale/pdf weights for some datasets (missing columns)
    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
        # compute scale weights
        events = self[murmuf_envelope_weights](events, **kwargs)

        # read out mur and weights
        events = self[murmuf_weights](events, **kwargs)

    if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
        # compute pdf weights
        events = self[pdf_weights](
            events,
            outlier_threshold=0.5,
            outlier_action="ignore",
            outlier_log_mode="warning",
            **kwargs,
        )

    return events


@event_weights_to_normalize.init
def event_weights_to_normalize_init(self) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {btag_weights}
        # dont store most btag_weights to save storage space, since we can reproduce them in ProduceColumns
        # but keep nominal one for checks/synchronization
        if hasattr(self, "local_shift_inst") and self.local_shift_inst.name == "nominal":
            self.produces |= {"btag_weight"}

    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {murmuf_envelope_weights, murmuf_weights}
        self.produces |= {murmuf_envelope_weights, murmuf_weights}

    if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {pdf_weights}
        self.produces |= {pdf_weights}


normalized_scale_weights = normalized_weight_factory(
    producer_name="normalized_scale_weights",
    weight_producers={murmuf_envelope_weights, murmuf_weights},
)

normalized_pdf_weights = normalized_weight_factory(
    producer_name="normalized_pdf_weights",
    weight_producers={pdf_weights},
)

normalized_pu_weights = normalized_weight_factory(
    producer_name="normalized_pu_weights",
    weight_producers={pu_weight},
)

muon_id_weights = muon_weights.derive("muon_id_weights", cls_dict={
    "weight_name": "muon_id_weight",
    "get_muon_config": (lambda self: self.config_inst.x.muon_iso_sf_names),
})
muon_iso_weights = muon_weights.derive("muon_iso_weights", cls_dict={
    "weight_name": "muon_iso_weight",
    "get_muon_config": (lambda self: self.config_inst.x.muon_id_sf_names),
})
muon_trigger_weights = muon_weights.derive("muon_trigger_weights", cls_dict={
    "weight_name": "muon_trigger_weight",
    "get_muon_config": (lambda self: self.config_inst.x.muon_trigger_sf_names),
})


@producer(
    uses={muon_id_weights, muon_iso_weights},
    produces={muon_id_weights, muon_iso_weights},
    mc_only=True,
)
def muon_id_iso_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer that calculates the muon id and iso weights.
    """
    # run muon id and iso weights
    events = self[muon_id_weights](events, **kwargs)
    events = self[muon_iso_weights](events, **kwargs)

    return events


@producer(
    uses={muon_trigger_weights},
)
def sl_trigger_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer that calculates the single lepton trigger weights.
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


def sl_trigger_weights_skip_func(self: Producer) -> bool:
    if not getattr(self, "config_inst", None) or not getattr(self, "dataset_inst", None):
        # do not skip when config or dataset is not set
        return False

    if self.config_inst.x.lepton_tag == "sl":
        # do not skip when lepton tag is single lepton
        return False
    else:
        return True


sl_trigger_weights.skip_func = sl_trigger_weights_skip_func


@producer(
    uses={
        normalization_weights,
        stitched_normalization_weights,
        stitched_normalization_weights_brs_from_processes,
    },
    produces={
        normalization_weights,
        stitched_normalization_weights,
        stitched_normalization_weights_brs_from_processes,
    },
    mc_only=True,
)
def all_normalization_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer that calculates each type of normalization weight separately
    """
    events = self[normalization_weights](events, **kwargs)
    events = self[stitched_normalization_weights](events, **kwargs)
    events = self[stitched_normalization_weights_brs_from_processes](events, **kwargs)

    return events


@all_normalization_weights.init
def all_normalization_weights_init(self: Producer) -> None:
    self[stitched_normalization_weights].weight_name = "stitched_normalization_weight"
    self[stitched_normalization_weights_brs_from_processes].weight_name = "stitched_normalization_weight_brs_from_processes"  # noqa: E501


@producer
def combined_normalization_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer that decides on a single type of stiched normalization weight per dataset.
    Reason for implementing this Producer is that we always want to rely on the CMSDB cross sections
    when stitching our signal samples, but we want to calculate the BRs ourselved for other
    types of sample stitching (e.g. DY).
    """
    events = self[normalization_weights](events, **kwargs)
    events = self[self.norm_weights_producer](events, **kwargs)
    return events


@combined_normalization_weights.init
def combined_normalization_weights_init(self: Producer) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    self[stitched_normalization_weights].weight_name = "stitched_normalization_weight"
    self[stitched_normalization_weights_brs_from_processes].weight_name = "stitched_normalization_weight"  # noqa: E501

    if self.dataset_inst.has_tag("is_hbv"):
        self.norm_weights_producer = stitched_normalization_weights_brs_from_processes
    else:
        self.norm_weights_producer = stitched_normalization_weights

    self.uses |= {self.norm_weights_producer, normalization_weights}
    self.produces |= {self.norm_weights_producer, normalization_weights}


@producer(
    uses={
        all_normalization_weights,
        top_pt_weight,
        vjets_weight,
        normalized_pu_weights,
    },
    produces={
        all_normalization_weights,
        top_pt_weight,
        vjets_weight,
        normalized_pu_weights,
    },
    mc_only=True,
    version=1,
)
def event_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers that are typically called in ProduceColumns.
    """

    # compute normalization weights
    events = self[all_normalization_weights](events, **kwargs)

    # compute gen top pt weights
    if self.dataset_inst.has_tag("is_ttbar"):
        events = self[top_pt_weight](events, **kwargs)

    # compute gen vjet pt weights
    if self.dataset_inst.has_tag("is_v_jets"):
        events = self[vjets_weight](events, **kwargs)

    if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
        # compute and normalize btag SF weights
        events = self[btag_weights](events, **kwargs)
        events = self[normalized_btag_weights](events, **kwargs)

    # compute electron and muon SF weights
    if not has_tag("skip_electron_weights", self.config_inst, self.dataset_inst, operator=any):
        events = self[electron_weights](events, **kwargs)
    if not has_tag("skip_muon_weights", self.config_inst, self.dataset_inst, operator=any):
        events = self[muon_id_iso_weights](events, **kwargs)

        if self.config_inst.x.lepton_tag == "sl":
            # compute single lepton trigger SF weights
            events = self[sl_trigger_weights](events, **kwargs)

    # normalize event weights using stats
    events = self[normalized_pu_weights](events, **kwargs)

    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
        events = self[normalized_scale_weights](events, **kwargs)

    if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
        events = self[normalized_pdf_weights](events, **kwargs)

    return events


@event_weights.init
def event_weights_init(self: Producer) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    if not has_tag("skip_electron_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {electron_weights}
        self.produces |= {electron_weights}

    if not has_tag("skip_muon_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {muon_id_iso_weights}
        self.produces |= {muon_id_iso_weights}

        self.uses |= {sl_trigger_weights}
        self.produces |= {sl_trigger_weights}

    if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {btag_weights, normalized_btag_weights}
        self.produces |= {normalized_btag_weights}

    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {normalized_scale_weights}
        self.produces |= {normalized_scale_weights}

    if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {normalized_pdf_weights}
        self.produces |= {normalized_pdf_weights}


@producer(
    uses={"mc_weight"},
    produces={"mc_weight"},
    mc_only=True,
)
def large_weights_killer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Simple producer that sets eventweights to 0 when too large.
    """
    if self.dataset_inst.is_data:
        raise Exception("large_weights_killer is only callable for MC")

    # TODO: figure out a good threshold when events are considered unphysical
    median_weight = ak.sort(abs(events.mc_weight))[int(len(events) / 2)]
    weight_too_large = abs(events.mc_weight) > 1000 * median_weight
    events = set_ak_column(events, "mc_weight", ak.where(weight_too_large, 0, events.mc_weight))

    return events
