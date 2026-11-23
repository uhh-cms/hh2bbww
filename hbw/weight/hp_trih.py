

from columnflow.util import maybe_import
from hbw.weight.default import base
from hbw.categorization.masks_trih import (
    mask_fn_ar, _yield,
    mask_fn_check_JME, mask_fn_j1, mask_fn_j2, mask_fn_j3, mask_fn_geqj3,
    mask_fn_check_BTV, mask_fn_bj1, mask_fn_bj2, mask_fn_bj3, mask_fn_geqbj3, mask_fn_geqbj2,
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
    "muon_id_weight": ["mu_id_sf"],
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    # "electron_weight": ["e_scale"],
    # "electron_weight": ["e_res"],
    "electron_reco_weight": ["e_reco_sf"],
    "normalized_murmuf_envelope_weight": ["murf_envelope"],
    "normalized_mur_weight": ["mur"],
    "normalized_muf_weight": ["muf"],
    "normalized_pdf_weight": ["pdf"],
    "normalized_isr_weight": ["isr"],
    "normalized_fsr_weight": ["fsr"],
    "top_pt_theory_weight": ["top_pt"],
    "normalized_pu_weight": ["minbias_xs"],
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
        # "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
    "categorizer_cls": mask_fn_ar,
})

default_nobjet = base.derive("default_nobjet", cls_dict={
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        "btag_weight": [f"btag_{unc}" for unc in btag_uncs_2024],
        # "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
})

# ----------------- Checks on JME -------------------------------------------------------------------------------------

check_JME = hhh_default.derive("check_JME", cls_dict={
    "categorizer_cls": mask_fn_check_JME,
})
check_j1 = hhh_default.derive("check_j1", cls_dict={
    "categorizer_cls": mask_fn_j1,
})
check_j2 = hhh_default.derive("check_j2 ", cls_dict={
    "categorizer_cls": mask_fn_j2,
})
check_j3 = hhh_default.derive("check_j3", cls_dict={
    "categorizer_cls": mask_fn_j3,
})

check_geqj3 = hhh_default.derive("check_geqj3", cls_dict={
    "categorizer_cls": mask_fn_geqj3,
})
# ----------------- Checks on BTV -------------------------------------------------------------------------------------
check_BTV = base.derive("check_BTV", cls_dict={
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        "btag_weight": [f"btag_{unc}" for unc in btag_uncs_2024],
        # "btag_weight": ["btag_bc", "btag_light"],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
    "categorizer_cls": mask_fn_check_BTV,
})
check_bj1 = check_BTV.derive("check_bj1", cls_dict={
    "categorizer_cls": mask_fn_bj1,
})
check_bj2 = check_BTV.derive("check_bj2", cls_dict={
    "categorizer_cls": mask_fn_bj2,
})
check_bj3 = check_BTV.derive("check_bj3", cls_dict={
    "categorizer_cls": mask_fn_bj3,
})
check_geqbj3 = check_BTV.derive("check_geqbj3", cls_dict={
    "categorizer_cls": mask_fn_geqbj3,
})
check_geqbj2 = check_BTV.derive("check_geqbj2", cls_dict={
    "categorizer_cls": mask_fn_geqbj2,
})

hhh_shape = base.derive("hhh_shape", cls_dict={
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        # "btag_weight": [f"btag_{unc}" for unc in btag_uncs_2024],
        "btag_weight": ["btag_statistic_bc", "btag_correlated_bc", "btag_correlated_light", "btag_uncorrelated_light"],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
    "categorizer_cls": mask_fn_ar,
})

test_year_split = base.derive("test_year_split", cls_dict={
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
    "categorizer_cls": mask_fn_ar,
})

_yield = base.derive("_yield", cls_dict={  # noqa
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        "btag_weight": [],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
    # "categorizer_cls": _yield,
})

cr = base.derive("cr", cls_dict={
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        "btag_weight": [],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": None,
})

test_dctr = base.derive("test_dctr", cls_dict={
    "weight_columns": {
        **default_correction_weights,
        "trigger_weight": ["trigger_sf"],
        "btag_weight": [],
        "ttbb_correction_weight": [],
        "stitched_normalization_weight": [],
    },
    "nondy_hist_producer": cr,
    "dy_correction_weight_producer": "dctr_ttbb_reweighter",
})
