# coding: utf-8

"""
Column production methods related to generic event weights.
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, has_ak_column, Route
from columnflow.selection import SelectionResult
from columnflow.production import Producer, producer
from columnflow.production.cms.pileup import pu_weight
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from hbw.production.btag import btag_weights
from columnflow.production.cms.scale import murmuf_weights, murmuf_envelope_weights
from columnflow.production.cms.pdf import pdf_weights
from columnflow.production.cms.top_pt_weight import gen_parton_top, top_pt_weight
from hbw.production.gen_v import gen_v_boson, vjets_weight
from hbw.production.normalized_weights import normalized_weight_factory
from hbw.production.normalized_btag import normalized_btag_weights
from hbw.util import has_tag


np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    produces={"event_weight"},
    mc_only=True,
)
def event_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer that calculates the 'final' event weight (as done in cf.CreateHistograms)
    """
    weight = ak.Array(np.ones(len(events)))
    if self.dataset_inst.is_mc:
        for column in self.config_inst.x.event_weights:
            weight = weight * Route(column).apply(events)
        for column in self.dataset_inst.x("event_weights", []):
            if has_ak_column(events, column):
                weight = weight * Route(column).apply(events)
            else:
                self.logger.warning_once(
                    f"missing_dataset_weight_{column}",
                    f"weight '{column}' for dataset {self.dataset_inst.name} not found",
                )

    events = set_ak_column(events, "event_weight", weight)

    return events


@event_weight.init
def event_weight_init(self: Producer) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    self.uses |= set(self.config_inst.x.event_weights.keys())
    self.uses |= set(self.dataset_inst.x("event_weights", {}).keys())


# copy of the btag_weights setup, removing the version check
# TODO: should be done in columnflow
@btag_weights.setup
def btag_weights_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets,
) -> None:
    bundle = reqs["external_files"]
    # create the btag sf corrector
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    correction_set = correctionlib.CorrectionSet.from_string(
        self.get_btag_file(bundle.files).load(formatter="gzip").decode("utf-8"),
    )
    corrector_name = self.get_btag_config()[0]
    self.btag_sf_corrector = correction_set[corrector_name]


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
            outlier_action="remove",
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


@producer(
    uses={
        normalization_weights,
        top_pt_weight,
        vjets_weight,
        normalized_pu_weights,
        event_weight,
    },
    produces={
        normalization_weights,
        top_pt_weight,
        vjets_weight,
        normalized_pu_weights,
        event_weight,
    },
    mc_only=True,
)
def event_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers that are typically called in ProduceColumns.
    """

    # compute normalization weights
    events = self[normalization_weights](events, **kwargs)

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
        events = self[muon_weights](events, **kwargs)

    # normalize event weights using stats
    events = self[normalized_pu_weights](events, **kwargs)

    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
        events = self[normalized_scale_weights](events, **kwargs)

    if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
        events = self[normalized_pdf_weights](events, **kwargs)

    # calculate the full event weight for plotting purposes
    events = self[event_weight](events, **kwargs)

    return events


@event_weights.init
def event_weights_init(self: Producer) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    if not has_tag("skip_electron_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {electron_weights}
        self.produces |= {electron_weights}

    if not has_tag("skip_muon_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {muon_weights}
        self.produces |= {muon_weights}

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
