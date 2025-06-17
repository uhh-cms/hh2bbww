# coding: utf-8

"""
Event weight producer.
"""

import law

from columnflow.util import maybe_import
from columnflow.histogramming import HistProducer
from columnflow.histogramming.default import cf_default
from columnflow.config_util import get_shifts_from_sources
from columnflow.columnar_util import Route
from hbw.production.prepare_objects import prepare_objects

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


# extend columnflow's default hist producer
@cf_default.hist_producer(uses={"mc_weight"}, mc_only=True)
def mc_weight(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, events.mc_weight


@cf_default.hist_producer(uses={"normalization_weight"}, mc_only=True)
def norm(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, events.normalization_weight


@cf_default.hist_producer(uses={"stitched_normalization_weight_brs_from_processes"}, mc_only=True)
def norm_brs_cmsdb(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, events.stitched_normalization_weight_brs_from_processes


@cf_default.hist_producer(uses={"stitched_normalization_weight"}, mc_only=True)
def stitched_norm(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, events.stitched_normalization_weight


@cf_default.hist_producer(mc_only=True)
def no_weights(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, ak.Array(np.ones(len(events), dtype=np.float32))


@cf_default.hist_producer(
    uses={prepare_objects},
    # both used columns and dependent shifts are defined in init below
    weight_columns=None,
    # only run on mc
    mc_only=False,
    # optional categorizer to obtain baseline event mask
    categorizer_cls=None,
)
def base(self: HistProducer, events: ak.Array, task: law.Task, **kwargs) -> ak.Array:
    # apply behavior (for variable reconstruction)

    events = self[prepare_objects](events, **kwargs)

    # apply mask
    if self.categorizer_cls:
        events, mask = self[self.categorizer_cls](events, **kwargs)
        events = events[mask]

    if self.dataset_inst.is_data:
        return events, ak.Array(np.ones(len(events), dtype=np.float32))

    # build the full event weight
    weight = ak.Array(np.ones(len(events), dtype=np.float64))
    for column in self.local_weight_columns.keys():
        weight = weight * Route(column).apply(events)

    # implement dummy shift by varying weight by factor of 2
    if "dummy" in task.local_shift_inst.name:
        logger.warning("Applying dummy weight shift (should never be use for real analysis)")
        variation = task.local_shift_inst.name.split("_")[-1]
        weight = weight * {"up": 2.0, "down": 0.5}[variation]

    return events, weight


@base.setup
def base_setup(
    self: HistProducer,
    reqs: dict,
    task: law.Task,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
) -> None:
    logger.info(
        f"HistProducer '{self.cls_name}' (dataset {self.dataset_inst}) uses weight columns: \n"
        f"{', '.join(self.weight_columns.keys())}",
    )


@base.init
def base_init(self: HistProducer) -> None:
    # NOTE: this might be called multiple times, might be quite inefficient
    # if not getattr(self, "config_inst", None) or not getattr(self, "dataset_inst", None):
    #     return

    if not getattr(self, "config_inst"):
        return

    if self.categorizer_cls:
        self.uses.add(self.categorizer_cls)

    dataset_inst = getattr(self, "dataset_inst", None)
    if dataset_inst and dataset_inst.is_data:
        return

    year = self.config_inst.campaign.x.year
    cpn_tag = self.config_inst.x.cpn_tag

    if not self.weight_columns:
        raise Exception("weight_columns not set")
    self.local_weight_columns = self.weight_columns.copy()

    if dataset_inst and dataset_inst.has_tag("skip_scale"):
        # remove dependency towards mur/muf weights
        for column in [
            "normalized_mur_weight", "normalized_muf_weight", "normalized_murmuf_envelope_weight",
            "mur_weight", "muf_weight", "murmuf_envelope_weight",
        ]:
            self.local_weight_columns.pop(column, None)

    if dataset_inst and dataset_inst.has_tag("skip_pdf"):
        # remove dependency towards pdf weights
        for column in ["pdf_weight", "normalized_pdf_weight"]:
            self.local_weight_columns.pop(column, None)

    if dataset_inst and not dataset_inst.has_tag("is_ttbar"):
        # remove dependency towards top pt weights
        self.local_weight_columns.pop("top_pt_weight", None)

    if dataset_inst and not dataset_inst.has_tag("is_v_jets"):
        # remove dependency towards vjets weights
        self.local_weight_columns.pop("vjets_weight", None)

    if dataset_inst and not dataset_inst.has_tag("is_dy"):
        # remove dependency towards vjets weights
        self.local_weight_columns.pop("njet_weight", None)
        self.local_weight_columns.pop("dy_weight", None)

    if dataset_inst and not dataset_inst.has_tag("is_dy"):
        # remove dependency towards dy weights
        self.local_weight_columns.pop("dy_weight", None)

    self.shifts = set()

    # when jec sources are known btag SF source, then propagate the shift to the HistProducer
    # TODO: we should do this somewhere centrally
    btag_sf_jec_sources = (
        (set(self.config_inst.x.btag_sf_jec_sources) | {"Total"}) &
        set(self.config_inst.x.jec.Jet["uncertainty_sources"])
    )
    self.shifts |= set(get_shifts_from_sources(
        self.config_inst,
        *[f"jec_{jec_source}" for jec_source in btag_sf_jec_sources],
    ))

    for weight_column, shift_sources in self.local_weight_columns.items():
        shift_sources = law.util.make_list(shift_sources)
        shift_sources = [s.format(year=year, cpn_tag=cpn_tag) for s in shift_sources]
        shifts = get_shifts_from_sources(self.config_inst, *shift_sources)
        for shift in shifts:
            if weight_column not in shift.x("column_aliases").keys():
                # make sure that column aliases are implemented
                raise Exception(
                    f"Weight column {weight_column} implements shift {shift}, but does not use it "
                    f"in 'column_aliases' aux {shift.x('column_aliases')}",
                )

            # declare shifts that the produced event weight depends on
            self.shifts |= set(shifts)

    # remove dummy column from weight columns and uses
    self.local_weight_columns.pop("dummy_weight", "")

    # store column names referring to weights to multiply
    self.uses |= self.local_weight_columns.keys()
    # self.uses = {"*"}


# @base.post_init
# def base_post_init(self: HistProducer, task: law.Task):
#     from hbw.util import debugger; debugger()


btag_uncs = [
    "hf", "lf", "hfstats1_{year}", "hfstats2_{year}",
    "lfstats1_{year}", "lfstats2_{year}", "cferr1", "cferr2",
]


default_correction_weights = {
    # "dummy_weight": ["dummy_{cpn_tag}"],
    "normalized_pu_weight": ["minbias_xs"],
    "muon_id_weight": ["mu_id_sf"],
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
    "normalized_murmuf_envelope_weight": ["murf_envelope"],
    "normalized_mur_weight": ["mur"],
    "normalized_muf_weight": ["muf"],
    "normalized_pdf_weight": ["pdf"],
    "top_pt_weight": ["top_pt"],
}

default_weight_columns = {
    "stitched_normalization_weight": [],
    **default_correction_weights,
}
default_hist_producer = base.derive("default", cls_dict={"weight_columns": default_weight_columns})

with_vjets_weight = default_hist_producer.derive("with_vjets_weight", cls_dict={"weight_columns": {
    **default_correction_weights,
    "vjets_weight": [],  # TODO: corrections/shift missing
    "stitched_normalization_weight": [],
}})
with_trigger_weight = default_hist_producer.derive("with_trigger_weight", cls_dict={"weight_columns": {
    **default_correction_weights,
    # "vjets_weight": [],  # TODO: corrections/shift missing
    "trigger_weight": ["trigger_sf"],
    "stitched_normalization_weight": [],
}})

with_dy_weight = default_hist_producer.derive("with_dy_weight", cls_dict={"weight_columns": {
    **default_correction_weights,
    "dy_weight": [],
    "trigger_weight": ["trigger_sf"],
    "stitched_normalization_weight": [],
}})

with_dy_njet_weight = default_hist_producer.derive("with_dy_njet_weight", cls_dict={"weight_columns": {
    **default_correction_weights,
    "njet_weight": [],  # TODO: corrections/shift missing
    "dy_weight": [],
    "trigger_weight": ["trigger_sf"],
    "stitched_normalization_weight": [],
}})

base.derive("unstitched", cls_dict={"weight_columns": {
    **default_correction_weights, "normalization_weight": [],
}})

base.derive("minimal", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "muon_id_weight": ["mu_id_sf"],
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    "top_pt_weight": ["top_pt"],
}})

weight_columns_execpt_btag = default_weight_columns.copy()
weight_columns_execpt_btag.pop("normalized_ht_njet_nhf_btag_weight")

no_btag_weight = base.derive("no_btag_weight", cls_dict={"weight_columns": weight_columns_execpt_btag})
base.derive("btag_not_normalized", cls_dict={"weight_columns": {
    **weight_columns_execpt_btag,
    "btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("btag_njet_normalized", cls_dict={"weight_columns": {
    **weight_columns_execpt_btag,
    "normalized_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("btag_ht_njet_normalized", cls_dict={"weight_columns": {
    **weight_columns_execpt_btag,
    "normalized_ht_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("btag_ht_njet_nhf_normalized", cls_dict={"weight_columns": {
    **weight_columns_execpt_btag,
    "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("btag_ht_normalized", cls_dict={"weight_columns": {
    **weight_columns_execpt_btag,
    "normalized_ht_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})

# weight sets for closure tests
base.derive("norm_and_btag", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("norm_and_btag_njet", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "normalized_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("norm_and_btag_ht_njet", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "normalized_ht_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("norm_and_btag_ht", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "normalized_ht_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})


from hbw.categorization.categories import mask_fn_highpt

no_btag_weight.derive("no_btag_weight_highpt", cls_dict={"categorizer_cls": mask_fn_highpt})


# additional hist producers for scale factors
from trigger.trigger_cats import mask_fn_dl_orth_with_l1_seeds

dl_orth_with_l1_seeds = default_hist_producer.derive("dl_orth_with_l1_seeds", cls_dict={
    "categorizer_cls": mask_fn_dl_orth_with_l1_seeds,
})

