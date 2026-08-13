

from columnflow.util import maybe_import
from hbw.weight.default import base
from hbw.categorization.masks_trih import (
    mask_fn_ar, _yield,
)

np = maybe_import("numpy")
ak = maybe_import("awkward")

# Btag uncertainties for 2022/23
btag_uncs = [
    "hf", "lf",
    "cferr1", "cferr2",
    "hfstats1", "lfstats1",
    "hfstats2", "lfstats2",
]

# Verbose set of btag uncertainties for 2024
# 12. August: Strong tensions experienced when using bc/light unc only
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
    "electron_reco_weight": ["e_reco_sf"],
    "normalized_murmuf_envelope_weight": ["murf_envelope"],
    "normalized_mur_weight": ["mur"],
    "normalized_muf_weight": ["muf"],
    "normalized_pdf_weight": ["pdf"],
    "normalized_isr_weight": ["isr"],
    "normalized_fsr_weight": ["fsr"],
    "top_pt_theory_weight": ["top_pt"],
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
        "btag_weight": [f"btag_{unc}" for unc in btag_uncs_2024],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
    "categorizer_cls": mask_fn_ar,
})

_yield = base.derive("_yield", cls_dict={
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        "btag_weight": [],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
    "categorizer_cls": _yield,
})
