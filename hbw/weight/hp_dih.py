# coding: utf-8

from columnflow.util import maybe_import
from hbw.weight.default import base

np = maybe_import("numpy")
ak = maybe_import("awkward")

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
    "muon_low_pt_id_weight": ["mu_low_pt_id_sf"],
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    "electron_reco_weight": ["e_reco_sf"],
    # "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
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
# weight_columns_execpt_btag = default_weight_columns.copy()
# weight_columns_execpt_btag.pop("normalized_ht_njet_nhf_btag_weight")

default_hist_producer = base.derive(
    "default",
    cls_dict={
        "weight_columns": default_weight_columns,
        "dy_correction_weight_producer": "dy_correction_weight",
    },
)
check = base.derive("check", cls_dict={"weight_columns": default_weight_columns})
unstitched = base.derive("unstitched", cls_dict={"weight_columns": {
    "dataset_normalization_weight": [],
    # "dy_correction_weight": [],
    # "trigger_weight": ["trigger_sf"],
    **default_correction_weights,
}})

with_vjets_weight = default_hist_producer.derive("with_vjets_weight", cls_dict={"weight_columns": {
    **default_correction_weights,
    "vjets_weight": [],  # TODO: corrections/shift missing
    "stitched_normalization_weight": [],
}})

# TODO: desperatly needs clean up
from hbw.categorization.masks_dih import (
    mask_fn_mbb80, mask_fn_met70,
    mask_fn_met_geq40,
    mask_fn_lep2_pt15, mask_fn_met_geq40_lep2_pt15, mask_fn_lep2_pt10,
)
with_trigger_weight = default_hist_producer.derive("with_trigger_weight", cls_dict={
    "pre_label": "Before DY correction",
    "weight_columns": {
        **default_correction_weights,
        # "vjets_weight": [],  # TODO: corrections/shift missing
        "trigger_weight": ["trigger_sf"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "categorizer_cls": mask_fn_lep2_pt15,
})
with_trigger_weight_no_top_pt = default_hist_producer.derive("with_trigger_weight_no_top_pt", cls_dict={
    "pre_label": "No top pt correction",
    "weight_columns": {
        "normalized_pu_weight": ["minbias_xs"],
        "muon_id_weight": ["mu_id_sf"],
        "muon_iso_weight": ["mu_iso_sf"],
        "electron_weight": ["e_sf"],
        "electron_reco_weight": ["e_reco_sf"],
        # "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
        "normalized_murmuf_envelope_weight": ["murf_envelope"],
        "normalized_mur_weight": ["mur"],
        "normalized_muf_weight": ["muf"],
        "normalized_pdf_weight": ["pdf"],
        "normalized_isr_weight": ["isr"],
        "normalized_fsr_weight": ["fsr"],
        # "vjets_weight": [],  # TODO: corrections/shift missing
        "trigger_weight": ["trigger_sf"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "categorizer_cls": mask_fn_lep2_pt15,
})
with_trigger_weight_nolep2 = default_hist_producer.derive("with_trigger_weight_nolep2", cls_dict={
    # "pre_label": "Before DY correction",
    "weight_columns": {
        **default_correction_weights,
        # "vjets_weight": [],  # TODO: corrections/shift missing
        "trigger_weight": ["trigger_sf"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    # "categorizer_cls": mask_fn_lep2_pt15,
})
with_trigger_weight_lep2pt10 = default_hist_producer.derive("with_trigger_weight_lep2pt10", cls_dict={
    # "pre_label": "Before DY correction",
    "weight_columns": {
        **default_correction_weights,
        # "vjets_weight": [],  # TODO: corrections/shift missing
        "trigger_weight": ["trigger_sf"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "categorizer_cls": mask_fn_lep2_pt10,
})
with_trigger_weight_nobtag = default_hist_producer.derive("with_trigger_weight_nobtag", cls_dict={
    "pre_label": "Before DY correction",
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        "stitched_normalization_weight": [],
    },
    "categorizer_cls": mask_fn_lep2_pt15,
})
with_dy_corr_nolep2 = default_hist_producer.derive("with_dy_corr_nolep2", cls_dict={
    "pre_label": "After DY correction",
    "nondy_hist_producer": "with_trigger_weight_nolep2",
    "weight_columns": {
        **default_correction_weights,
        "dy_correction_weight": ["dy_correction"],
        "trigger_weight": ["trigger_sf"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "dy_correction_weight_producer": "dy_correction_weight",
    # "categorizer_cls": mask_fn_lep2_pt15,
})
with_dy_corr_lep2pt10 = default_hist_producer.derive("with_dy_corr_lep2pt10", cls_dict={
    "pre_label": "After DY correction",
    "nondy_hist_producer": "with_trigger_weight_lep2pt10",
    "weight_columns": {
        **default_correction_weights,
        "dy_correction_weight": ["dy_correction"],
        "trigger_weight": ["trigger_sf"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "dy_correction_weight_producer": "dy_correction_weight",
    "categorizer_cls": mask_fn_lep2_pt10,
})

# NOTE: we added a fix that automatically uses the "with_trigger_weight" outputs for all non-DY datasets
# because the dy_correction_weight is only relevant for DY processes. This is implemented in
# hbw/analysis/create_analysis.py
with_dy_corr = default_hist_producer.derive("with_dy_corr", cls_dict={
    "pre_label": "After DY correction",
    "nondy_hist_producer": "with_trigger_weight",
    "weight_columns": {
        **default_correction_weights,
        "dy_correction_weight": ["dy_correction"],
        "trigger_weight": ["trigger_sf"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "dy_correction_weight_producer": "dy_correction_weight",
    "categorizer_cls": mask_fn_lep2_pt15,
})
incl_dy_corr = default_hist_producer.derive("incl_dy_corr", cls_dict={
    "pre_label": "After DY correction",
    "nondy_hist_producer": "with_trigger_weight",
    "weight_columns": {
        **default_correction_weights,
        "dy_correction_weight": ["dy_correction"],
        "trigger_weight": ["trigger_sf"],
        "stitched_normalization_weight": [],
    },
    "dy_correction_weight_producer": "dy_incl_corr_weight",
})
with_dy_corr_nobtag = default_hist_producer.derive("with_dy_corr_nobtag", cls_dict={
    "pre_label": "After DY correction",
    "nondy_hist_producer": "with_trigger_weight_nobtag",
    "weight_columns": {
        **default_correction_weights,
        "dy_correction_weight": ["dy_correction"],
        "trigger_weight": ["trigger_sf"],
        "stitched_normalization_weight": [],
    },
    "dy_correction_weight_producer": "dy_correction_weight",
    "categorizer_cls": mask_fn_lep2_pt15,
})
with_trigger_weight_metlep = default_hist_producer.derive("with_trigger_weight_metlep", cls_dict={
    "pre_label": "Before DY correction",
    "weight_columns": {
        **default_correction_weights,
        # "vjets_weight": [],  # TODO: corrections/shift missing
        "trigger_weight": ["trigger_sf"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "categorizer_cls": mask_fn_met_geq40_lep2_pt15,
})
with_dy_corr_metlep = default_hist_producer.derive("with_dy_corr_metlep", cls_dict={
    "pre_label": "After DY correction",
    "nondy_hist_producer": "with_trigger_weight_metlep",
    "weight_columns": {
        **default_correction_weights,
        "dy_correction_weight": ["dy_correction"],
        "trigger_weight": ["trigger_sf"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "dy_correction_weight_producer": "dy_correction_weight",
    "categorizer_cls": mask_fn_met_geq40_lep2_pt15,
})
#
# HistProducers with masks via categorization
#

from hbw.categorization.categories import (
    catid_ge2b_loose, catid_njet2,
)


with_trigger_weight_lep2cut = default_hist_producer.derive("with_trigger_weight_lep2cut", cls_dict={
    "pre_label": "Before DY correction",
    "weight_columns": {
        **default_correction_weights,
        # "vjets_weight": [],  # TODO: corrections/shift missing
        "trigger_weight": ["trigger_sf"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "categorizer_cls": mask_fn_lep2_pt15,
})

with_dy_corr_lep2cut = default_hist_producer.derive("with_dy_corr_lep2cut", cls_dict={
    "pre_label": "After DY correction",
    "nondy_hist_producer": "with_trigger_weight_lep2cut",
    "weight_columns": {
        **default_correction_weights,
        "dy_correction_weight": ["dy_correction"],
        "trigger_weight": ["trigger_sf"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "dy_correction_weight_producer": "dy_correction_weight",
    "categorizer_cls": mask_fn_lep2_pt15,
})

met70 = with_trigger_weight.derive("met70", cls_dict={
    "categorizer_cls": mask_fn_met70,
})

no_dycorr = default_hist_producer.derive("no_dycorr", cls_dict={
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
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
met_geq40_with_dy_corr = with_dy_corr.derive("met_geq40_with_dy_corr", cls_dict={
    "pre_label": "\n".join([r"$p_{T}^{miss} \geq 40$ GeV"]),
    "nondy_hist_producer": "met_geq40_no_dycorr",
    "categorizer_cls": mask_fn_met_geq40,
    "dy_correction_weight_producer": "dy_correction_weight",
})
met_geq40_incl_dy_corr = with_dy_corr.derive("met_geq40_incl_dy_corr", cls_dict={
    "pre_label": "\n".join([r"$p_{T}^{miss} \geq 40$ GeV"]),
    "nondy_hist_producer": "met_geq40_no_dycorr",
    "categorizer_cls": mask_fn_met_geq40,
    "dy_correction_weight_producer": "dy_incl_corr_weight",
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

# no_btag_weight = base.derive("no_btag_weight", cls_dict={"weight_columns": weight_columns_execpt_btag})
# base.derive("btag_not_normalized", cls_dict={"weight_columns": {
#     **weight_columns_execpt_btag,
#     "btag_weight": [f"btag_{unc}" for unc in btag_uncs],
# }})
# base.derive("btag_njet_normalized", cls_dict={"weight_columns": {
#     **weight_columns_execpt_btag,
#     "normalized_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
# }})
# base.derive("btag_ht_njet_normalized", cls_dict={"weight_columns": {
#     **weight_columns_execpt_btag,
#     "normalized_ht_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
# }})
# base.derive("btag_ht_njet_nhf_normalized", cls_dict={"weight_columns": {
#     **weight_columns_execpt_btag,
#     "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
# }})
# base.derive("btag_ht_normalized", cls_dict={"weight_columns": {
#     **weight_columns_execpt_btag,
#     "normalized_ht_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
# }})

# weight sets for closure tests
# base.derive("norm_and_btag", cls_dict={"weight_columns": {
#     "stitched_normalization_weight": [],
#     "btag_weight": [f"btag_{unc}" for unc in btag_uncs],
# }})
# base.derive("norm_and_btag_njet", cls_dict={"weight_columns": {
#     "stitched_normalization_weight": [],
#     "normalized_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
# }})
# base.derive("norm_and_btag_ht_njet", cls_dict={"weight_columns": {
#     "stitched_normalization_weight": [],
#     "normalized_ht_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
# }})
# base.derive("norm_and_btag_ht", cls_dict={"weight_columns": {
#     "stitched_normalization_weight": [],
#     "normalized_ht_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
# }})


# from hbw.categorization.categories import mask_fn_highpt

# no_btag_weight.derive("no_btag_weight_highpt", cls_dict={"categorizer_cls": mask_fn_highpt})

from hbw.categorization.masks_dih import mask_fn_met70, mask_fn_dyvr, mask_fn_single_lep_triggers

with_trigger_weight.derive("met70", cls_dict={"categorizer_cls": mask_fn_met70})
with_trigger_weight.derive("dyvr_derivation_region", cls_dict={"categorizer_cls": mask_fn_dyvr})

# additional hist producers for scale factors
from hbw.trigger.trigger_cats import mask_fn_dl_orth2_with_l1_seeds
no_trig_sf = default_hist_producer.derive("no_trig_sf", cls_dict={
    "pre_label": "Before trigger SF",
    "weight_columns": {
        **default_correction_weights,
        "stitched_normalization_weight": [],
        "btag_weight": ["btag_bc", "btag_light"],
    },
})
dl_orth2_with_l1_seeds = no_trig_sf.derive("dl_orth2_with_l1_seeds", cls_dict={
    "categorizer_cls": mask_fn_dl_orth2_with_l1_seeds,
})
no_trig_sf_lep2cut = no_trig_sf.derive("no_trig_sf_lep2cut", cls_dict={
    "categorizer_cls": mask_fn_lep2_pt15,
})
with_dy_corr_notrig = default_hist_producer.derive("with_dy_corr_notrig", cls_dict={
    "pre_label": "After DY correction",
    "nondy_hist_producer": "no_trig_sf_lep2cut",
    "weight_columns": {
        **default_correction_weights,
        "dy_correction_weight": ["dy_correction"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "dy_correction_weight_producer": "dy_correction_weight",
    "categorizer_cls": mask_fn_lep2_pt15,
})
test_wdy = with_dy_corr_notrig.derive("test_wdy", cls_dict={})
with_dy_corr_notrig_nolep2 = default_hist_producer.derive("with_dy_corr_notrig_nolep2", cls_dict={
    "pre_label": "After DY correction",
    "nondy_hist_producer": "no_trig_sf",
    "weight_columns": {
        **default_correction_weights,
        "dy_correction_weight": ["dy_correction"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "dy_correction_weight_producer": "dy_correction_weight",
})
no_trig_sf_sltrig = no_trig_sf.derive("no_trig_sf_sltrig", cls_dict={
    "categorizer_cls": mask_fn_single_lep_triggers,
})
with_dy_corr_sltrig = default_hist_producer.derive("with_dy_corr_sltrig", cls_dict={
    "pre_label": "After DY correction",
    "nondy_hist_producer": "no_trig_sf_sltrig",
    "weight_columns": {
        **default_correction_weights,
        "dy_correction_weight": ["dy_correction"],
        "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "dy_correction_weight_producer": "dy_correction_weight",
    "categorizer_cls": mask_fn_single_lep_triggers,
})
# dl_orth_with_l1_seeds = no_trig_sf.derive("dl_orth_with_l1_seeds", cls_dict={
#     "categorizer_cls": mask_fn_dl_orth_with_l1_seeds,
# })
