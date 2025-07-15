# coding: utf-8

"""
Column production methods related to generic event weights.
"""

import functools
import law

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, fill_at
from columnflow.selection import SelectionResult
from columnflow.production import Producer, producer
from columnflow.production.cms.pileup import pu_weight
from columnflow.production.cms.parton_shower import ps_weights
from columnflow.production.normalization import (
    normalization_weights,
    stitched_normalization_weights,
    stitched_normalization_weights_brs_from_processes,
)
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights, MuonSFConfig
from columnflow.production.cms.btag import btag_weights
from columnflow.production.cms.scale import murmuf_weights, murmuf_envelope_weights
from columnflow.production.cms.pdf import pdf_weights
from columnflow.production.cms.top_pt_weight import top_pt_weight
from hbw.production.top_pt_theory import top_pt_theory_weight
from columnflow.production.cms.dy import dy_weights
from hbw.production.gen_v import vjets_weight
from hbw.production.normalized_weights import normalized_weight_factory
from hbw.production.normalized_btag import normalized_btag_weights
from hbw.production.dataset_normalization import dataset_normalization_weight
from hbw.production.trigger import sl_trigger_weights, dl_trigger_weights
from hbw.util import has_tag, IF_DY

np = maybe_import("numpy")
ak = maybe_import("awkward")


# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
logger = law.logger.get_logger(__name__)


@producer(
    uses={
        pu_weight,
    },
    # produces={
    #     pu_weight,
    # },
    mc_only=True,
)
def event_weights_to_normalize(self: Producer, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers that are typically called as part of SelectEvents
    since it is required to normalize them before applying certain event selections.
    """

    # compute pu weights
    events = self[pu_weight](events, **kwargs)
    if self.has_dep(ps_weights):
        events = self[ps_weights](events, **kwargs)

    if self.has_dep(btag_weights):
        # compute btag SF weights (for renormalization tasks)
        events = self[btag_weights](
            events,
            jet_mask=results.aux["jet_mask"],
            negative_b_score_action="ignore",
            negative_b_score_log_mode="debug",
            **kwargs,
        )

    # skip scale/pdf weights for some datasets (missing columns)
    if self.has_dep(murmuf_envelope_weights):
        # compute scale weights
        events = self[murmuf_envelope_weights](events, **kwargs)

    if self.has_dep(murmuf_weights):
        # read out mur and weights
        events = self[murmuf_weights](events, **kwargs)

    if self.has_dep(pdf_weights):
        # compute pdf weights
        events = self[pdf_weights](
            events,
            outlier_threshold=0.99,
            outlier_action="remove",
            outlier_log_mode="debug",
            invalid_weights_action="ignore" if self.dataset_inst.has_tag("partial_lhe_weights") else "raise",
            **kwargs,
        )

    return events


@event_weights_to_normalize.init
def event_weights_to_normalize_init(self) -> None:
    # used Producers need to be set in the init or decorator
    if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {btag_weights}

    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {murmuf_envelope_weights, murmuf_weights}

    if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {pdf_weights}

    if not has_tag("no_ps_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {ps_weights}


@event_weights_to_normalize.post_init
def event_weights_to_normalize_post_init(self, task: law.Task) -> None:
    # produced columns can be set in post_init to choose stored columns based on the shift
    for _cls in self.uses:
        if _cls == btag_weights and task.shift == "nominal":
            self.produces |= {btag_weights}
        elif _cls == btag_weights:
            self.produces |= self.deps[btag_weights].produced_columns
        elif task.shift == "nominal":
            self.produces |= self.deps[_cls].produced_columns
        else:
            self.produces |= {
                route for route in self.deps[_cls].produced_columns
                if not route.nano_column.endswith("_up") and not route.nano_column.endswith("_down")
            }


# renormalized weights
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
normalized_ps_weights = normalized_weight_factory(
    producer_name="normalized_ps_weights",
    weight_producers={ps_weights},
)

# muon weights
muon_id_weights = muon_weights.derive("muon_id_weights", cls_dict={
    "weight_name": "muon_id_weight",
    "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_iso_sf_names)),
})
muon_iso_weights = muon_weights.derive("muon_iso_weights", cls_dict={
    "weight_name": "muon_iso_weight",
    "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_id_sf_names)),
})

# electron weights
electron_reco_weights = electron_weights.derive("electron_reco_weights", cls_dict={
    "weight_name": "electron_reco_weight",
    "get_electron_config": (lambda self: self.config_inst.x.electron_reco_sf_names),
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
    events = self[self.norm_weights_producer](events, **kwargs)

    # very simple Producer that creates normalization weight without any stitching
    # (can only be used when there is a one-to-one mapping between datasets and processes)
    events = self[dataset_normalization_weight](events, **kwargs)
    return events


@combined_normalization_weights.init
def combined_normalization_weights_init(self: Producer) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    if self.dataset_inst.has_tag("is_hbv"):
        self.norm_weights_producer = stitched_normalization_weights_brs_from_processes
    elif "dy_" in self.dataset_inst.name:
        self.norm_weights_producer = stitched_normalization_weights
    elif self.dataset_inst.name.startswith("w_lnu") and self.dataset_inst.name.endswith("_amcatnlo"):
        self.norm_weights_producer = stitched_normalization_weights
    else:
        self.norm_weights_producer = normalization_weights

    self.norm_weights_producer.weight_name = "stitched_normalization_weight"

    self.uses |= {self.norm_weights_producer, dataset_normalization_weight}
    self.produces |= {self.norm_weights_producer, dataset_normalization_weight}


@producer(
    uses={
        combined_normalization_weights,
        IF_DY(dy_weights),
        top_pt_weight,
        top_pt_theory_weight,
        vjets_weight,
        IF_DY(dy_weights),
        normalized_pu_weights,
    },
    produces={
        combined_normalization_weights,
        IF_DY(dy_weights),
        top_pt_theory_weight,
        vjets_weight,
        IF_DY(dy_weights),
        normalized_pu_weights,
    },
    mc_only=True,
    version=law.config.get_expanded("analysis", "event_weights_version", 2),
)
def event_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers that are typically called in ProduceColumns.
    """

    # compute normalization weights
    events = self[combined_normalization_weights](events, **kwargs)

    # compute gen top pt weights
    if self.dataset_inst.has_tag("is_ttbar"):
        events = self[top_pt_weight](events, **kwargs)
        events = self[top_pt_theory_weight](events, **kwargs)

    if self.dataset_inst.has_tag("is_dy"):
        events = self[dy_weights](events, **kwargs)

    # compute gen vjet pt weights
    if self.dataset_inst.has_tag("is_v_jets"):
        events = self[vjets_weight](events, **kwargs)

    if self.dataset_inst.has_tag("is_dy"):
        events = self[dy_weights](events, **kwargs)

    if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
        # compute and normalize btag SF weights
        events = self[btag_weights](
            events,
            negative_b_score_action="ignore",
            negative_b_score_log_mode="debug",
            **kwargs,
        )
        events = self[normalized_btag_weights](events, **kwargs)

    # compute electron and muon SF weights
    if not has_tag("skip_electron_weights", self.config_inst, self.dataset_inst, operator=any):
        events = self[electron_weights](events, **kwargs)
        events = self[electron_reco_weights](events, **kwargs)

    if not has_tag("skip_muon_weights", self.config_inst, self.dataset_inst, operator=any):
        events = self[muon_id_iso_weights](events, **kwargs)

    if not has_tag("skip_trigger_weights", self.config_inst, self.dataset_inst, operator=any):
        events = self[self.trigger_weights_producer](events, **kwargs)

    # normalize event weights using stats
    events = self[normalized_pu_weights](events, **kwargs)

    if not has_tag("no_ps_weights", self.config_inst, self.dataset_inst, operator=any):
        events = self[normalized_ps_weights](events, **kwargs)

    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
        events = self[normalized_scale_weights](events, **kwargs)

    if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
        events = self[normalized_pdf_weights](events, **kwargs)

    return events


@event_weights.init
def event_weights_init(self: Producer) -> None:
    logger.debug(f"checking selector tag: {self.config_inst.has_tag('selector_init')}")

    if not getattr(self, "dataset_inst", None):
        return

    if not has_tag("skip_electron_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {electron_weights, electron_reco_weights}
        self.produces |= {electron_weights, electron_reco_weights}

    if not has_tag("skip_muon_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {muon_id_iso_weights}
        self.produces |= {muon_id_iso_weights}

    if not has_tag("skip_trigger_weights", self.config_inst, self.dataset_inst, operator=any):
        self.trigger_weights_producer = (
            sl_trigger_weights if self.config_inst.x.lepton_tag == "sl"
            else dl_trigger_weights
        )
        self.uses |= {self.trigger_weights_producer}
        self.produces |= {self.trigger_weights_producer}

    if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {btag_weights, normalized_btag_weights}
        self.produces |= {normalized_btag_weights}

    if not has_tag("no_ps_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {normalized_ps_weights}
        self.produces |= {normalized_ps_weights}

    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {normalized_scale_weights}
        self.produces |= {normalized_scale_weights}

    if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {normalized_pdf_weights}
        self.produces |= {normalized_pdf_weights}


@producer(
    uses={"mc_weight", "genWeight"},
    produces={"mc_weight", "genWeight"},
    mc_only=True,
)
def large_weights_killer(self: Producer, events: ak.Array, stats: dict, **kwargs) -> ak.Array:
    """
    Simple producer that sets eventweights to 0 when too large.
    """
    if self.dataset_inst.is_data:
        raise Exception("large_weights_killer is only callable for MC")

    # set mc_weight to zero when genWeight is > 0.5 for powheg HH events
    if self.dataset_inst.has_tag("is_hh") and self.dataset_inst.name.endswith("powheg"):
        # TODO: this feels very unsafe because genWeight can also be just 1 for all events. To be revisited
        weight_too_large = abs(events.genWeight) > 0.5
        logger.warning(f"found {ak.sum(weight_too_large)} HH events with genWeight > 0.5")

        events = fill_at(events, weight_too_large, "mc_weight", 0.0, value_type=np.float32)

    # check for anomalous weights and store in stats
    median_weight = ak.sort(abs(events.mc_weight))[int(len(events) / 2)]
    anomalous_weights_mask = abs(events.mc_weight) > 1000 * median_weight
    if ak.any(anomalous_weights_mask):
        logger.warning(f"found {ak.sum(anomalous_weights_mask)} events with weights > 1000 * median weight")
        stats["num_events_anomalous_weights"] += ak.sum(anomalous_weights_mask)

    return events


# for testing
test_event_weights = event_weights.derive("test_event_weights")
