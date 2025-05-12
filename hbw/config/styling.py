# coding: utf-8

"""
Collection of helpers for styling, e.g.
- dicitonaries of defaults for variable definition, colors, labels, etc.
- functions to quickly create variable insts in a predefined way
"""

import re

import order as od

from columnflow.columnar_util import EMPTY_FLOAT

#
# Processes
#

cms_color_palette_1 = {
    "blue": "#5790fc",  # Blueberry
    "yellow": "#f89c20",  # Marigold
    "red": "#e42536",  # Alizarin Crimson
    "purple": "#964a8b",  # Plum
    "grey": "#9c9ca1",  # Manatee
    "violet": "#7a21dd",  # Blue-Violet
    "black": "#000000",
}

cms_color_palette_2 = {
    "blue": "#3f90da",  # Tufts Blue
    "yellow": "#ffa90e",  # Dark Tangerine
    "red": "#bd1f01",  # International Orange (Engineering)
    "grey": "#94a4a2",  # Morning Blue
    "purple": "#832db6",  # Grape
    "brown": "#a96b59",  # Blast-Off Bronze
    "orange": "#e76300",  # Spanish Orange
    "green": "#b9ac70",  # Misty Moss
    "darkgrey": "#717581",  # AuroMetalSaurus
    "turqoise": "#92dadd",  # Pale Robin Egg Blue
    "black": "#000000",
}

# tt: red
# dy: yellow
# st: orange
# vv
# ttV ?
# w_lnu
# h
# qcd: blue

color_palette_1 = {
    "black": "#000000",
    "red": "#e41a1c",
    "blue": "#377eb8",
    "green": "#4daf4a",
    "purple": "#984ea3",
    "orange": "#ff7f00",
    "yellow": "#ffff33",
    "brown": "#a65628",
    "pink": "#f781bf",
    "grey": "#999999",
}

color_palette = cms_color_palette_2

default_process_colors = {
    "data": color_palette["black"],
    "tt": color_palette["red"],
    "qcd": color_palette["blue"],
    "qcd_mu": color_palette["blue"],
    "qcd_ele": color_palette["blue"],
    "w_lnu": color_palette["green"],
    "v_lep": color_palette["green"],
    "h": color_palette["purple"],
    "st": color_palette["orange"],
    "t_bkg": color_palette["orange"],
    # "dy": color_palette["yellow"],
    "dy": color_palette["yellow"],
    "dy_hf": color_palette["yellow"],
    "dy_lf": color_palette["brown"],
    "dy_m50toinf": color_palette["yellow"],
    "dy_m10to50": color_palette["brown"],
    "dy_m4to10": color_palette["darkgrey"],
    "ttv": color_palette["turqoise"],
    "vv": color_palette["blue"],
    "other": color_palette["grey"],
    "hh_ggf_hbb_htt": color_palette["grey"],
    "signal_ggf2": color_palette["black"],
    "signal_vbf2": color_palette["grey"],
    "hh_ggf_hbb_hww2l2nu_kl1_kt1": color_palette["black"],
    "hh_vbf_hbb_hww2l2nu_kv1_k2v1_kl1": color_palette["grey"],
}

for decay in ("", "qqlnu", "2l2nu"):
    default_process_colors[f"hh_ggf_hbb_hvv{decay}"] = "#000000"  # black
    default_process_colors[f"signal_ggf2{decay}"] = "#000000"  # black
    default_process_colors[f"signal_vbf2{decay}"] = "#999999"  # black
    default_process_colors[f"hh_ggf_hbb_hvv{decay}_kl1_kt1"] = "#000000"  # black
    default_process_colors[f"hh_ggf_hbb_hvv{decay}_kl0_kt1"] = "#1b9e77"  # green2
    default_process_colors[f"hh_ggf_hbb_hvv{decay}_kl2p45_kt1"] = "#d95f02"  # orange2
    default_process_colors[f"hh_ggf_hbb_hvv{decay}_kl5_kt1"] = "#e7298a"  # pink2

    default_process_colors[f"hh_vbf_hbb_hvv{decay}"] = "#999999"  # grey
    default_process_colors[f"hh_vbf_hbb_hvv{decay}_kv1_k2v1_kl1"] = color_palette["darkgrey"]
    # default_process_colors[f"hh_vbf_hbb_hvv{decay}_kv1_k2v1_kl1"] = color_palette["brown"]
    # default_process_colors[f"hh_vbf_hbb_hvv{decay}_kv1_k2v1_kl1"] = "#999999"  # grey
    default_process_colors[f"hh_vbf_hbb_hvv{decay}_kv1_k2v1_kl0"] = "#377eb8"  # blue
    default_process_colors[f"hh_vbf_hbb_hvv{decay}_kv1_k2v1_kl2"] = "#4daf4a"  # green
    default_process_colors[f"hh_vbf_hbb_hvv{decay}_kv1_k2v0_kl1"] = "#984ea3"  # purple
    default_process_colors[f"hh_vbf_hbb_hvv{decay}_kv1_k2v2_kl1"] = "#ff7f00"  # orange
    default_process_colors[f"hh_vbf_hbb_hvv{decay}_kv0p5_k2v1_kl1"] = "#a65628"  # brown
    default_process_colors[f"hh_vbf_hbb_hvv{decay}_kv1p5_k2v1_kl1"] = "#f781bf"  # pink


default_labels = {
    "dy_m50toinf": "DY ($M > 50$)",
    "dy_m50toinf_0j": "DY ($M > 50$, 0 jets)",
    "dy_m50toinf_1j": "DY ($M > 50$, 1 jets)",
    "dy_m50toinf_2j": "DY ($M > 50$, 2 jets)",
    "dy_m10to50": "DY ($10 < M < 50$)",
    "dy_m4to10": "DY ($4 < M < 10$)",
    "st_tchannel_t": "st (t-channel, t)",
    "st_tchannel_tbar": r"st (t-channel, $\bar{t}$)",
    "st_twchannel_t_sl": "tW (t, sl)",
    "st_twchannel_tbar_sl": r"tW ($\bar{t}$, sl)",
    "st_twchannel_t_dl": "tW (t, dl)",
    "st_twchannel_tbar_dl": r"tW ($\bar{t}$, dl)",
    # "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": r"hh_{vbf} (dl)",
}

ml_labels = {
    "tt": "$t\\bar{t}$",
    "hh_ggf_hbb_hvv": r"hh_{ggf}",
    "hh_vbf_hbb_hvv": r"hh_{vbf}",
    "hh_ggf_hbb_hvvqqlnu_kl1_kt1": r"hh_{ggf} (sl)",
    "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1": r"hh_{vbf} (sl)",
    "hh_ggf_hbb_hvv2l2nu_kl1_kt1": r"ggHH",
    "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": r"qqHH",
    # "hh_ggf_hbb_hvv2l2nu_kl1_kt1": r"hh_{ggf} (dl)",
    # "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1": r"hh_{vbf} (dl)",
    "graviton_hh_ggf_bbww_m250": "grav250",
    "graviton_hh_ggf_bbww_m350": "grav350",
    "graviton_hh_ggf_bbww_m450": "grav450",
    "graviton_hh_ggf_bbww_m600": "grav600",
    "graviton_hh_ggf_bbww_m750": "grav750",
    "graviton_hh_ggf_bbww_m1000": "grav1000",
    "st": "st",
    "w_lnu": "W",
    "dy": "DY",
    "h": "H",
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

        if label := default_labels.get(proc.name, None):
            proc.label = label
        elif "hh_vbf" in proc.name:
            label = proc.label
            pattern = r"hh_vbf_kv([mp\d]+)_k2v([mp\d]+)_kl([mp\d]+)\s?(.*)"
            replacement = r"$HH_{vbf}^{\1,\2,\3}$\4"
            proc.label = re.sub(pattern, replacement, label)

        if short_label := short_labels.get(proc.name, None):
            proc.short_label = short_label

        # unstack signal in plotting
        if "hh_" in proc.name.lower():
            proc.add_tag("is_signal")
            proc.unstack = True
            # proc.scale = "stack"
            # proc.scale = "stack"

        # labels used for ML categories
        proc.x.ml_label = ml_labels.get(proc.name, proc.name)


#
# Variables
#

default_var_binning = {
    # General object fields
    "pt": (400, 0, 400),
    "eta": (40, -5., 5.),
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
    "particleNet_XbbVsQCD": (40, 0, 1),
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
        name=name.format(obj=obj, i=i, var=var).lower(),
        expression=expr.format(obj=obj, i=i, var=var),
        null_value=EMPTY_FLOAT,
        binning=default_var_binning[var],
        unit=default_var_unit.get(var, "1"),
        x_title=x_title_base.format(obj=obj, i=i) + default_var_title_format.get(var, var),
    )
