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
    "dy_lep": "#ffff33",  # yellow
    "ttV": "#a65628",  # brown
    "VV": "#f781bf",  # pink
    "other": "#999999",  # grey
    "ggHH_kl_1_kt_1_sl_hbbhww": "#000000",  # black
    "ggHH_kl_0_kt_1_sl_hbbhww": "#1b9e77",  # green2
    "ggHH_kl_2p45_kt_1_sl_hbbhww": "#d95f02",  # orange2
    # "ggHH_kl_5_kt_1_sl_hbbhww": "#e7298a",  # pink2
    "ggHH_kl_5_kt_1_sl_hbbhww": "#000080",  # navy
    "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww": "#e41a1c",  # red
    "qqHH_CV_1_C2V_1_kl_0_sl_hbbhww": "#377eb8",  # blue
    "qqHH_CV_1_C2V_1_kl_2_sl_hbbhww": "#4daf4a",  # green
    "qqHH_CV_1_C2V_0_kl_1_sl_hbbhww": "#984ea3",  # purple
    "qqHH_CV_1_C2V_2_kl_1_sl_hbbhww": "#ff7f00",  # orange
    "qqHH_CV_0p5_C2V_1_kl_1_sl_hbbhww": "#a65628",  # brown
    "qqHH_CV_1p5_C2V_1_kl_1_sl_hbbhww": "#f781bf",  # pink
    "ggHH_kl_1_kt_1_dl_hbbhww": "#000000",  # black
    "ggHH_kl_0_kt_1_dl_hbbhww": "#1b9e77",  # green2
    "ggHH_kl_2p45_kt_1_dl_hbbhww": "#d95f02",  # orange2
    "ggHH_kl_5_kt_1_dl_hbbhww": "#e7298a",  # pink2
    "qqHH_CV_1_C2V_1_kl_1_dl_hbbhww": "#e41a1c",  # red
    "qqHH_CV_1_C2V_1_kl_0_dl_hbbhww": "#377eb8",  # blue
    "qqHH_CV_1_C2V_1_kl_2_dl_hbbhww": "#4daf4a",  # green
    "qqHH_CV_1_C2V_0_kl_1_dl_hbbhww": "#984ea3",  # purple
    "qqHH_CV_1_C2V_2_kl_1_dl_hbbhww": "#ff7f00",  # orange
    "qqHH_CV_0p5_C2V_1_kl_1_dl_hbbhww": "#a65628",  # brown
    "qqHH_CV_1p5_C2V_1_kl_1_dl_hbbhww": "#f781bf",  # pink
    "hh_ggf_bbtautau": "#984ea3",  # purple
}

ml_labels = {
    "tt": "$t\\bar{t}$",
    "ggHH_kl_1_kt_1_sl_hbbhww": "ggHH (sl)",
    "ggHH_kl_1_kt_1_dl_hbbhww": "ggHH (dl)",
    "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww": "qqHH (sl)",
    "qqHH_CV_1_C2V_1_kl_1_dl_hbbhww": "qqHH (dl)",
    "st": "st",
    "w_lnu": "W",
    "dy_lep": "DY",
    "v_lep": "W+DY",
    "t_bkg": "tt+st",
}


def stylize_processes(config: od.Config) -> None:
    """
    Small helper that sets the process insts to analysis-appropriate defaults
    For now: only colors and unstacking
    Could also include some more defaults (labels, unstack, ...)
    """

    for proc in config.processes:
        # set default colors
        if color := default_process_colors.get(proc.name, None):
            proc.color1 = color

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
    """
    config.add_variable(
        name=name.format(obj=obj, i=i + 1, var=var).lower(),
        expression=expr.format(obj=obj, i=i, var=var),
        null_value=EMPTY_FLOAT,
        binning=default_var_binning[var],
        unit=default_var_unit.get(var, "1"),
        x_title=x_title_base.format(obj=obj, i=i + 1) + default_var_title_format.get(var, var),
    )
