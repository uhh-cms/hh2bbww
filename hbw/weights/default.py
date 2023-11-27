# coding: utf-8

"""
Event weight producer.
"""

import law

from columnflow.util import maybe_import
from columnflow.weight import WeightProducer, weight_producer
from columnflow.config_util import get_shifts_from_sources
from columnflow.columnar_util import Route

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@weight_producer(uses={"normalization_weight"}, mc_only=True)
def normalization_only(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events.normalization_weight


@weight_producer(mc_only=True)
def no_weights(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    return ak.Array(np.ones(len(events), dtype=np.float32))


@weight_producer(
    # both used columns and dependent shifts are defined in init below
    weight_columns=None,
    # only run on mc
    mc_only=True,
)
def default_weight_producer(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    # build the full event weight
    weight = ak.Array(np.ones(len(events), dtype=np.float32))
    for column in self.weight_columns.keys():
        weight = weight * Route(column).apply(events)

    return weight


@default_weight_producer.init
def default_weight_producer_init(self: WeightProducer) -> None:
    # set the default weight_columns
    if not self.weight_columns:
        self.weight_columns = {
            "normalization_weight": [],
            "normalized_pu_weight": ["minbias_xs"],
            "muon_weight": ["mu_sf"],
            "electron_weight": ["e_sf"],
            "normalized_btag_weight": [f"btag_{unc}" for unc in self.config_inst.x("btag_uncs")],
            "normalized_murf_envelope_weight": ["murf_envelope"],
            "normalized_mur_weight": ["mur"],
            "normalized_muf_weight": ["muf"],
            "normalized_pdf_weight": ["pdf"],
        }

    if self.dataset_inst.has_tag("skip_scale"):
        # remove dependency towards mur/muf weights
        for column in [
            "normalized_mur_weight", "normalized_muf_weight", "normalized_murf_envelope_weight",
            "mur_weight", "muf_weight", "murf_envelope_weight",
        ]:
            self.weight_columns.pop(column, None)

    if self.dataset_inst.has_tag("skip_pdf"):
        # remove dependency towards pdf weights
        for column in ["pdf_weight", "normalized_pdf_weight"]:
            self.weight_columns.pop(column, None)

    self.shifts = set()
    for weight_column, shift_sources in self.weight_columns.items():
        shift_sources = law.util.make_list(shift_sources)
        shifts = get_shifts_from_sources(self.config_inst, *shift_sources)
        for shift in shifts:
            if weight_column not in shift.x("column_aliases").keys():
                # make sure that column aliases are implemented
                raise Exception(
                    f"Weight column {weight_column} implements shift {shift}, but does not use it"
                    f"in 'column_aliases' aux {shift.x('column_aliases')}",
                )

            # declare shifts that the produced event weight depends on
            self.shifts |= set(shifts)

    # store column names referring to weights to multiply
    self.uses |= self.weight_columns.keys()
