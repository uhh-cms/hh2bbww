# coding: utf-8

"""
Collection of helpers for styling, e.g.
- dicitonaries of defaults for variable definition, colors, labels, etc.
- functions to quickly create variable insts in a predefined way
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT

#
# Processes
#

default_process_colors = {
    "data": "#000000",  # black
    "tt": "#e41a1c",  # red
    "qcd": "#377eb8",  # blue
    "qcd_mu": "#377eb8",  # blue
    "qcd_ele": "#377eb8",  # blue
    "w_lnu": "#4daf4a",  # green
    "v_lep": "#4daf4a",  # green
    "higgs": "#984ea3",  # purple
    "st": "#ff7f00",  # orange
    "t_bkg": "#e41a1c",  # orange
    "dy": "#ffff33",  # yellow
    "ttV": "#a65628",  # brown
    "VV": "#f781bf",  # pink
    "other": "#999999",  # grey
    "hh_ggf_hbb_hvvqqlnu_kl1_kt1": "#000000",  # black
    "hh_ggf_hbb_hvvqqlnu_kl0_kt1": "#1b9e77",  # green2
    "hh_ggf_hbb_hvvqqlnu_kl2p45_kt1": "#d95f02",  # orange2
    # "hh_ggf_hbb_hvvqqlnu_kl5_kt1": "#e7298a",  # pink2
    "hh_ggf_hbb_hvvqqlnu_kl5_kt1": "#000080",  # navy
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1": "#999999",  # grey
    # "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1": "#e41a1c",  # red
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl0": "#377eb8",  # blue
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl2": "#4daf4a",  # green
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v0_kl1": "#984ea3",  # purple
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v2_kl1": "#ff7f00",  # orange
    "hh_vbf_hbb_hvvqqlnu_kv0p5_k2v1_kl1": "#a65628",  # brown
    "hh_vbf_hbb_hvvqqlnu_kv1p5_k2v1_kl1": "#f781bf",  # pink
    "hh_ggf_hbb_hvv2l2nu_kl1_kt1": "#000000",  # black
    "hh_ggf_hbb_hvv2l2nu_kl0_kt1": "#1b9e77",  # green2
    "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1": "#d95f02",  # orange2
    "hh_ggf_hbb_hvv2l2nu_kl5_kt1": "#e7298a",  # pink2
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": "#999999",  # grey
    # "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": "#e41a1c",  # red
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl0": "#377eb8",  # blue
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl2": "#4daf4a",  # green
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1": "#984ea3",  # purple
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v2_kl1": "#ff7f00",  # orange
    "hh_vbf_hbb_hvv2l2nu_kv0p5_k2v1_kl1": "#a65628",  # brown
    "hh_vbf_hbb_hvv2l2nu_kv1p5_k2v1_kl1": "#f781bf",  # pink
    "hh_ggf_bbtautau": "#984ea3",  # purple
}

ml_labels = {
    "tt": "$t\\bar{t}$",
    "hh_ggf_hbb_hvvqqlnu_kl1_kt1": "hh_ggf (sl)",
    "hh_ggf_hbb_hvv2l2nu_kl1_kt1": "hh_ggf (dl)",
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1": "hh_vbf (sl)",
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": "hh_vbf (dl)",
    "graviton_hh_ggf_bbww_m250": "grav250",
    "graviton_hh_ggf_bbww_m350": "grav350",
    "graviton_hh_ggf_bbww_m450": "grav450",
    "graviton_hh_ggf_bbww_m600": "grav600",
    "graviton_hh_ggf_bbww_m750": "grav750",
    "graviton_hh_ggf_bbww_m1000": "grav1000",
    "st": "st",
    "w_lnu": "W",
    "dy": "DY",
    "v_lep": "W+DY",
    "t_bkg": "tt+st",
}

short_labels = {
    "hh_ggf_hbb_hvvqqlnu_kl0_kt1": r"$HH_{ggf}^{\kappa\lambda=0}$ (SL)",
    "hh_ggf_hbb_hvvqqlnu_kl1_kt1": r"$HH_{ggf}^{\kappa\lambda=1}$ (SL)",
    "hh_ggf_hbb_hvvqqlnu_kl2p45_kt1": r"$HH_{ggf}^{\kappa\lambda=2.45}$ (SL)",
    "hh_ggf_hbb_hvvqqlnu_kl5_kt1": r"$HH_{ggf}^{\kappa\lambda=5}$ (SL)",
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1": r"$HH_{vbf}^{1,1,1} (SL)$",
    "hh_ggf_hbb_hvv2l2nu_kl0_kt1": r"$HH_{ggf}^{\kappa\lambda=0}$ (DL)",
    "hh_ggf_hbb_hvv2l2nu_kl1_kt1": r"$HH_{ggf}^{\kappa\lambda=1}$ (DL)",
    "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1": r"$HH_{ggf}^{\kappa\lambda=2.45}$ (DL)",
    "hh_ggf_hbb_hvv2l2nu_kl5_kt1": r"$HH_{ggf}^{\kappa\lambda=5}$ (DL)",
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1": r"$HH_{vbf}^{1,1,1} (SL)$",
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl0": r"$HH_{vbf}^{1,1,0} (SL)$",
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl2": r"$HH_{vbf}^{1,1,2} (SL)$",
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v0_kl1": r"$HH_{vbf}^{1,0,1} (SL)$",
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v2_kl1": r"$HH_{vbf}^{1,2,1} (SL)$",
    "hh_vbf_hbb_hvvqqlnu_kv0p5_k2v1_kl1": r"$HH_{vbf}^{0.5,1,1} (SL)$",
    "hh_vbf_hbb_hvvqqlnu_kv1p5_k2v1_kl1": r"$HH_{vbf}^{1.5,1,1} (SL)$",
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": r"$HH_{vbf}^{1,1,1} (DL)$",
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl0": r"$HH_{vbf}^{1,1,0} (DL)$",
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl2": r"$HH_{vbf}^{1,1,2} (DL)$",
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1": r"$HH_{vbf}^{1,0,1} (DL)$",
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v2_kl1": r"$HH_{vbf}^{1,2,1} (DL)$",
    "hh_vbf_hbb_hvv2l2nu_kv0p5_k2v1_kl1": r"$HH_{vbf}^{0.5,1,1} (DL)$",
    "hh_vbf_hbb_hvv2l2nu_kv1p5_k2v1_kl1": r"$HH_{vbf}^{1.5,1,1} (DL)$",
    "w_lnu": r"$W \rightarrow l\nu$",
    "dy": r"$Z \rightarrow ll$",
    "qcd_mu": r"$QCD \mu$",
    "qcd_ele": r"$QCD e$",
}


def stylize_processes(config: od.Config) -> None:
    """
    Small helper that sets the process insts to analysis-appropriate defaults
    For now: only colors and unstacking
    Could also include some more defaults (labels, unstack, ...)
    """

    for proc, _, _ in config.walk_processes():
        # set default colors
        if color := default_process_colors.get(proc.name, None):
            proc.color1 = color

        if short_label := short_labels.get(proc.name, None):
            proc.short_label = short_label

        # unstack signal in plotting
        if "hh_" in proc.name.lower():
            proc.unstack = True

        # labels used for ML categories
        proc.x.ml_label = ml_labels.get(proc.name, proc.name)


#
# Variables
#

default_var_binning = {
    # General object fields
    "pt": (40, 0, 400),
    "eta": (40, -2.5, 2.5),
    "eta_full": (40, -5.0, 5.0),
    "phi": (32, -3.2, 3.2),
    "mass": (40, 0, 400),
    # Jet
    "btagDeepB": (40, 0, 1),
    "btagDeepFlavB": (40, 0, 1),
    "btagPNetB": (40, 0, 1),
    "b_score": (40, 0, 1),
    "qgl": (40, 0, 1),
    "puId": (8, -0.5, 7.5),
    "puIdDisc": (40, -2, 1),
    "chHEF": (40, 0, 1),
    "bRegRes": (80, -1, 1),
    "bRegCorr": (80, 0, 2),
    # FatJet
    "msoftdrop": (40, 0, 400),
    "deepTagMD_HbbvsQCD": (40, 0, 1),
    # Leptons
    "dxy": (40, 0, 0.1),
    "dz": (40, 0, 0.1),
    "mvaTTH": (40, 0, 1),
    "miniPFRelIso_all": (40, 0, 1),
    "pfRelIso03_all": (40, 0, 1),
    "pfRelIso04_all": (40, 0, 1),
    "mvaFall17V2Iso": (40, 0, 1),
    "mvaFall17V2noIso": (40, 0, 1),
    "mvaLowPt": (40, 0, 1),
    # Event properties
    "ht": (40, 0, 800),
    "n_jet": (11, -.5, 10.5),
    "n_lep": (4, -.5, 3.5),
    # Propierties of two combined objects
    "m": (40, 0, 400),
    "dr": (40, 0, 8),
    "deta": (40, 0, 5),
    "dphi": (32, 0, 3.2),
}

# TODO: use
default_var_to_expr = {
    "eta_full": "eta",
}

default_var_unit = {
    "pt": "GeV",
    "mass": "GeV",
    "msoftdrop": "GeV",
    "dxy": "cm",
    "dz": "cm",
    "ht": "GeV",
}

default_var_title_format = {
    "pt": r"$p_{T}$",
    "eta": r"$\eta$",
    "phi": r"$\phi$",
    "dxy": r"$d_{xy}$",
    "dz": r"d_{z}",
}

name = "cf_{obj}{i}_{var}"
expr = "cutflow.{obj}.{var}[:, {i}]"
x_title_base = r"{obj} {i} "


def quick_addvar(config: od.Config, obj: str, i: int, var: str):
    """
    Helper to quickly generate generic variable instances for variable *var* from
    the *i*th entry of some object *obj*

    Variables are lowercase and have names in the format:
    cf_{obj}{i}_{var}, where `obj` is the name of the given object, `i` is the index of the
    object (starting at 1) and `var` is the variable of interest; example: cf_loosejet1_pt
    """
    config.add_variable(
        name=name.format(obj=obj, i=i + 1, var=var).lower(),
        expression=expr.format(obj=obj, i=i, var=var),
        null_value=EMPTY_FLOAT,
        binning=default_var_binning[var],
        unit=default_var_unit.get(var, "1"),
        x_title=x_title_base.format(obj=obj, i=i + 1) + default_var_title_format.get(var, var),
    )
