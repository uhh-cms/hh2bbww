

from columnflow.util import maybe_import
from hbw.weight.default import base
from hbw.categorization.masks_trih import (
    mask_fn_hhh_sr, mask_fn_hhh_sr_bcut,
)

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
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    # "electron_reco_weight": ["e_reco_sf"],
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
    "trigger_weight": ["trigger_sf"],
    **default_correction_weights,
}
no_sel_weight_columns = {
    "stitched_normalization_weight": [],
    "normalized_murmuf_envelope_weight": ["murf_envelope"],
    "normalized_mur_weight": ["mur"],
    "normalized_muf_weight": ["muf"],
    "normalized_pdf_weight": ["pdf"],
    "normalized_isr_weight": ["isr"],
    "normalized_fsr_weight": ["fsr"],
    "top_pt_theory_weight": ["top_pt"],
    "electron_weight": ["e_sf"],
    "normalized_pu_weight": ["minbias_xs"],
    # "dy_correction_weight": [],
    # "trigger_weight": ["trigger_sf"],
    # **default_correction_weights,
}
unstitched_weight_columns = {
    "dataset_normalization_weight": [],
    "trigger_weight": ["trigger_sf"],
    **default_correction_weights,
}
# weight_columns_execpt_btag = default_weight_columns.copy()
# weight_columns_execpt_btag.pop("normalized_ht_njet_nhf_btag_weight")

default_hist_producer = base.derive("default", cls_dict={"weight_columns": default_weight_columns})
no_sel_hist_producer = base.derive("no_sel", cls_dict={"weight_columns": no_sel_weight_columns})
check = base.derive("check", cls_dict={"weight_columns": default_weight_columns})
unstitched = base.derive("unstitched", cls_dict={"weight_columns": {
    "dataset_normalization_weight": [],
    # "dy_correction_weight": [],
    # "trigger_weight": ["trigger_sf"],
    **default_correction_weights,
}})

default_hist_producer_trih = default_hist_producer.derive("default_hist_producer_trih", cls_dict={
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
})

test_hhh_sr = default_hist_producer_trih.derive("test_hhh_sr", cls_dict={
    "categorizer_cls": mask_fn_hhh_sr,
})

test_hhh_sr_bcut = default_hist_producer_trih.derive("test_hhh_sr_bcut", cls_dict={
    "categorizer_cls": mask_fn_hhh_sr_bcut,
})
