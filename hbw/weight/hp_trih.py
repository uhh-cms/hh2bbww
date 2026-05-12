

from columnflow.util import maybe_import
from hbw.weight.default import base
from hbw.categorization.masks_trih import (
    mask_fn_ar,  # mask_fn_sr,
)

np = maybe_import("numpy")
ak = maybe_import("awkward")

btag_uncs = [
    "hf", "lf",
    "cferr1", "cferr2",
    "hfstats1", "lfstats1",
    "hfstats2", "lfstats2",
]

btag_uncs_2024 = [
    "fsrdef_bc", "isrdef_bc",
    "hdamp_bc", "jer_bc", "jes_bc",
    "mass_bc", "statistic_bc",
    "tune_bc",
    "correlated_light",
    "uncorrelated_light",
]


default_correction_weights = {
    # "dummy_weight": ["dummy_{cpn_tag}"],
    "normalized_pu_weight": ["minbias_xs"],
    "muon_id_weight": ["mu_id_sf"],
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    # "btag_weight": [],
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
    # "btag_weight": [],
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

unstitched = base.derive("unstitched", cls_dict={"weight_columns": {
    "dataset_normalization_weight": [],
    **default_correction_weights,
}})

hhh_default = base.derive("hhh_default", cls_dict={
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        "btag_weight": [],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
    "categorizer_cls": mask_fn_ar,
})

hhh_shape = base.derive("hhh_shape", cls_dict={
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        "btag_weight": [f"btag_{unc}" for unc in btag_uncs_2024],
        # "btag_weight": [ "btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
    "categorizer_cls": mask_fn_ar,
})
