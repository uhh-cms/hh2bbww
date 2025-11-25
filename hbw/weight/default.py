# coding: utf-8

"""
Event weight producer.
"""

from hbw.util import call_once_on_config
import law

from columnflow.util import maybe_import
from columnflow.histogramming import HistProducer
from columnflow.histogramming.default import cf_default
from columnflow.config_util import get_shifts_from_sources
from columnflow.columnar_util import Route, set_ak_column
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


@cf_default.hist_producer(uses={"dataset_normalization_weight"}, mc_only=True)
def unstitched_norm(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, events.dataset_normalization_weight


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
    tt_weight=None,
    dy_weight=None,
    nondy_hist_producer=None,
    pre_label="",
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

    if self.tt_weight and self.tt_weight > 0 and self.dataset_inst.has_tag("is_ttbar"):
        weight = weight * self.tt_weight
    if self.dy_weight and self.dy_weight > 0 and self.dataset_inst.has_tag("is_dy"):
        weight = weight * self.dy_weight

    # implement dummy shift by varying weight by factor of 2
    if "dummy" in task.local_shift_inst.name:
        logger.warning("Applying dummy weight shift (should never be use for real analysis)")
        variation = task.local_shift_inst.name.split("_")[-1]
        weight = weight * {"up": 2.0, "down": 0.5}[variation]

    # special case: if only "weight_unweighted" is requested, we do not want to apply any weight at all
    if (
        hasattr(task, "variables") and
        len(task.variables) == 1 and
        task.variables[0].startswith("weight_unweighted")
    ):
        events = set_ak_column(events, "weight", weight, value_type=np.float32)
        return events, ak.Array(np.ones(len(events), dtype=np.float32))

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
    if "dy_correction_weight" in self.local_weight_columns.keys():
        # add dy_correction_weight to the reader targets
        reader_targets["dy_correction_weight"] = inputs["dy_correction_weight_producer"]["columns"]


@base.requires
def base_requires(self: HistProducer, task: law.Task, reqs: law.util.InsertableDict) -> None:
    """
    Define the requirements for the base HistProducer.
    This is called before the setup method.
    """
    from columnflow.tasks.production import ProduceColumns
    if "dy_correction_weight" in self.local_weight_columns.keys():
        reqs["dy_correction_weight_producer"] = ProduceColumns.req(
            task,
            producer="dy_correction_weight",
        )


@base.init
def base_init(self: HistProducer) -> None:

    if not getattr(self, "config_inst"):
        return

    @call_once_on_config
    def update_cat_label(config_inst, pre_label):
        for cat_inst, _, _ in config_inst.walk_categories():
            cat_inst.label = "\n".join([pre_label, cat_inst.label])

    if self.pre_label:
        update_cat_label(self.config_inst, self.pre_label)

    if self.categorizer_cls:
        self.uses.add(self.categorizer_cls)

    dataset_inst = getattr(self, "dataset_inst", None)
    if dataset_inst and dataset_inst.is_data:
        # if we are on data, we do not need any weights
        self.local_weight_columns = {}
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

    if dataset_inst and dataset_inst.has_tag("no_ps_weights"):
        self.local_weight_columns.pop("normalized_isr_weight", None)
        self.local_weight_columns.pop("normalized_fsr_weight", None)

    if dataset_inst and dataset_inst.has_tag("skip_pdf"):
        # remove dependency towards pdf weights
        for column in ["pdf_weight", "normalized_pdf_weight"]:
            self.local_weight_columns.pop(column, None)

    if dataset_inst and not dataset_inst.has_tag("is_ttbar"):
        # remove dependency towards top pt weights
        self.local_weight_columns.pop("top_pt_weight", None)
        self.local_weight_columns.pop("top_pt_theory_weight", None)

    if dataset_inst and not dataset_inst.has_tag("is_v_jets"):
        # remove dependency towards vjets weights
        self.local_weight_columns.pop("vjets_weight", None)

    if dataset_inst and not dataset_inst.has_tag("is_dy"):
        # remove dependency towards vjets weights
        self.local_weight_columns.pop("dy_correction_weight", None)
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


@base.post_init
def base_post_init(self: HistProducer, task: law.Task):
    if self.dataset_inst.is_data:
        return
    if "isr" not in task.shift:
        # no nominal ISR weight --> remove it from uses and local_weight_columns
        self.uses.discard("normalized_isr_weight")
        self.local_weight_columns.pop("normalized_isr_weight", None)
    if "fsr" not in task.shift:
        # no nominal FSR weight --> remove it from uses and local_weight_columns
        self.uses.discard("normalized_fsr_weight")
        self.local_weight_columns.pop("normalized_fsr_weight", None)


btag_uncs = [
    "hf", "lf",
    "cferr1", "cferr2",
    "hfstats1", "lfstats1",
    "hfstats2", "lfstats2",
]


default_correction_weights = {
    # "dummy_weight": ["dummy_{cpn_tag}"],
    "normalized_pu_weight": ["minbias_xs"],
    "muon_id_weight": ["mu_id_sf"],
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    "electron_reco_weight": ["e_reco_sf"],
    "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
    "normalized_murmuf_envelope_weight": ["murf_envelope"],
    "normalized_mur_weight": ["mur"],
    "normalized_muf_weight": ["muf"],
    "normalized_pdf_weight": ["pdf"],
    "normalized_isr_weight": ["isr"],
    "normalized_fsr_weight": ["fsr"],
    "top_pt_theory_weight": ["top_pt"],
}

default_weight_columns = {
    "stitched_normalization_weight": [],
    "dy_correction_weight": [],
    "trigger_weight": ["trigger_sf"],
    **default_correction_weights,
}
unstitched_weight_columns = {
    "dataset_normalization_weight": [],
    "dy_correction_weight": [],
    "trigger_weight": ["trigger_sf"],
    **default_correction_weights,
}
weight_columns_execpt_btag = default_weight_columns.copy()
weight_columns_execpt_btag.pop("normalized_ht_njet_nhf_btag_weight")

default_hist_producer = base.derive("default", cls_dict={"weight_columns": default_weight_columns})
unstitched = base.derive("unstitched", cls_dict={"weight_columns": {
    "dataset_normalization_weight": [],
    "dy_correction_weight": [],
    "trigger_weight": ["trigger_sf"],
    **default_correction_weights,
}})

with_vjets_weight = default_hist_producer.derive("with_vjets_weight", cls_dict={"weight_columns": {
    **default_correction_weights,
    "vjets_weight": [],  # TODO: corrections/shift missing
    "stitched_normalization_weight": [],
}})

with_trigger_weight = default_hist_producer.derive("with_trigger_weight", cls_dict={
    "pre_label": "Before DY correction",
    "weight_columns": {
        **default_correction_weights,
        # "vjets_weight": [],  # TODO: corrections/shift missing
        "trigger_weight": ["trigger_sf"],
        "stitched_normalization_weight": [],
    },
})

# NOTE: we added a fix that automatically uses the "with_trigger_weight" outputs for all non-DY datasets
# because the dy_correction_weight is only relevant for DY processes. This is implemented in
# hbw/analysis/create_analysis.py
with_dy_corr = default_hist_producer.derive("with_dy_corr", cls_dict={
    "pre_label": "After DY correction",
    "nondy_hist_producer": "with_trigger_weight",
    "version": 0,
    "weight_columns": {
        **default_correction_weights,
        "dy_correction_weight": ["dy_correction"],
        "trigger_weight": ["trigger_sf"],
        "stitched_normalization_weight": [],
    },
})
with_dy_corr_no_trig_sf = default_hist_producer.derive("with_dy_corr_no_trig_sf", cls_dict={
    "nondy_hist_producer": None,
    "version": 0,
    "weight_columns": {
        **default_correction_weights,
        "dy_correction_weight": ["dy_correction"],
        "stitched_normalization_weight": [],
    },
})

#
# HistProducers with masks via categorization
#

from hbw.categorization.categories import (
    mask_fn_mbb80, catid_ge2b_loose, catid_njet2, mask_fn_met70,
    mask_fn_met_geq40,
)

met70 = with_trigger_weight.derive("met70", cls_dict={
    "categorizer_cls": mask_fn_met70,
})

met_geq40 = default_hist_producer.derive("met_geq40", cls_dict={
    "nondy_hist_producer": None,
    "categorizer_cls": mask_fn_met_geq40,
})
met_geq40_with_dy_corr = with_dy_corr.derive("met_geq40_with_dy_corr", cls_dict={
    "pre_label": "\n".join(["After DY correction", r"$E_{T}^{miss} \geq 40$ GeV"]),
    "nondy_hist_producer": None,
    "categorizer_cls": mask_fn_met_geq40,
})
met_geq40_no_dycorr = default_hist_producer.derive("met_geq40_no_dycorr", cls_dict={
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
    "categorizer_cls": mask_fn_met_geq40,
})
met_geq40_with_dy_corr_unstitched = with_dy_corr.derive("met_geq40_with_dy_corr_unstitched", cls_dict={
    "weight_columns": unstitched_weight_columns,
    "nondy_hist_producer": None,
    "categorizer_cls": mask_fn_met_geq40,
})

mbb80 = with_dy_corr.derive("mbb80", cls_dict={
    "nondy_hist_producer": None,
    "categorizer_cls": mask_fn_mbb80,
})
poormans_postfit = with_dy_corr.derive("poormans_postfit", cls_dict={
    "nondy_hist_producer": None,
    "tt_weight": 0.90,
    "dy_weight": 1.04,
})
ge2jets = with_dy_corr.derive("ge2jets", cls_dict={
    "nondy_hist_producer": None,
    "categorizer_cls": catid_njet2,
})
ge2looseb = with_dy_corr.derive("ge2looseb", cls_dict={
    "nondy_hist_producer": None,
    "categorizer_cls": catid_ge2b_loose,
})

# base.derive("unstitched", cls_dict={"weight_columns": {
#     **default_correction_weights, "normalization_weight": [],
# }})

base.derive("stitched_only", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
}})
base.derive("stitched_ttdycorr", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "dy_correction_weight": [],
    "top_pt_theory_weight": ["top_pt"],
}})
base.derive("stitched_leptonsf", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "muon_id_weight": ["mu_id_sf"],
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    "electron_reco_weight": ["e_reco_sf"],
}})
base.derive("stitched_leptonsf_btag", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "muon_id_weight": ["mu_id_sf"],
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    "electron_reco_weight": ["e_reco_sf"],
    "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("stitched_leptonsf_btag_pu", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "muon_id_weight": ["mu_id_sf"],
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    "electron_reco_weight": ["e_reco_sf"],
    "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
    "normalized_pu_weight": ["minbias_xs"],
}})
base.derive("stitched_leptonsf_btag_pu_trigger", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "muon_id_weight": ["mu_id_sf"],
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    "electron_reco_weight": ["e_reco_sf"],
    "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
    "normalized_pu_weight": ["minbias_xs"],
    "trigger_weight": ["trigger_sf"],
}})
base.derive("stitched_leptonsf_btag_pu_trigger_ttdycorr", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "muon_id_weight": ["mu_id_sf"],
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    "electron_reco_weight": ["e_reco_sf"],
    "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
    "trigger_weight": ["trigger_sf"],
    "normalized_pu_weight": ["minbias_xs"],
    "dy_correction_weight": [],
    "top_pt_theory_weight": ["top_pt"],
}})

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

from hbw.categorization.categories import mask_fn_met70, mask_fn_dyvr

with_trigger_weight.derive("met70", cls_dict={"categorizer_cls": mask_fn_met70})
with_trigger_weight.derive("dyvr_derivation_region", cls_dict={"categorizer_cls": mask_fn_dyvr})

# additional hist producers for scale factors
from hbw.trigger.trigger_cats import mask_fn_dl_orth2_with_l1_seeds
no_trig_sf = default_hist_producer.derive("no_trig_sf", cls_dict={"weight_columns": {
    **default_correction_weights,
    "stitched_normalization_weight": [],
}})
dl_orth2_with_l1_seeds = no_trig_sf.derive("dl_orth2_with_l1_seeds", cls_dict={
    "categorizer_cls": mask_fn_dl_orth2_with_l1_seeds,
})
# dl_orth_with_l1_seeds = no_trig_sf.derive("dl_orth_with_l1_seeds", cls_dict={
#     "categorizer_cls": mask_fn_dl_orth_with_l1_seeds,
# })
