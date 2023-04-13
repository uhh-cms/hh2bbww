# coding: utf-8

"""
Definition of variables that can be plotted via `PlotCutflowVariables` tasks.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT


# cutflow variables
def add_cutflow_variables(config: od.Config) -> None:
    """
    defines reco-level cutflow variables in a convenient way; variables are lowercase and have
    names in the format:
    cf_{obj}{i}_{var}, where `obj` is the name of the given object, `i` is the index of the
    object (starting at 1) and `var` is the variable of interest; example: cf_loosejet1_pt
    """
    # default xtitle formatting per variable
    var_title_format = {
        "pt": r"$p_{T}$",
        "eta": r"$\eta$",
        "phi": r"$\phi$",
        "dxy": r"$d_{xy}$",
        "dz": r"d_{z}",
    }
    # default binning per variable
    var_binning = {
        "pt": (40, 0, 100),
        "eta": (40, -5, 5),
        "phi": (40, 0, 3.2),
        "mass": (40, 0, 200),
        "dxy": (40, 0, 0.1),
        "dz": (40, 0, 0.1),
        "mvaTTH": (40, 0, 1),
        "miniPFRelIso_all": (40, 0, 1),
        "pfRelIso03_all": (40, 0, 1),
        "pfRelIso04_all": (40, 0, 1),
        "mvaFall17V2Iso": (40, 0, 1),
        "mvaFall17V2noIso": (40, 0, 1),
        "mvaLowPt": (40, 0, 1),
    }
    var_unit = {
        "pt": "GeV",
        "dxy": "cm",
        "dz": "cm",
    }

    name = "cf_{obj}{i}_{var}"
    expr = "cutflow.{obj}.{var}[:, {i}]"
    x_title_base = r"{obj} {i} "
    def quick_addvar(obj: str, i: int, var: str):
        """
        Helper to quickly generate generic variable instances
        """
        config.add_variable(
            name=name.format(obj=obj, i=i + 1, var=var).lower(),
            expression=expr.format(obj=obj, i=i, var=var),
            null_value=EMPTY_FLOAT,
            binning=var_binning[var],
            unit=var_unit.get(var, "1"),
            x_title=x_title_base.format(obj=obj, i=i + 1) + var_title_format.get(var, var),
        )

    # Jets
    for i in range(4):
        # loose jets
        for var in ("pt",):
            quick_addvar("LooseJet", i, var)

    # Leptons
    for i in range(3):
        # veto leptons
        for var in ("pt", "eta", "dxy", "dz", "mvaTTH", "miniPFRelIso_all"):
            quick_addvar("VetoLepton", i, var)
        # veto electrons
        for var in ("mvaFall17V2Iso", "mvaFall17V2noIso", "pfRelIso03_all"):
            quick_addvar("VetoElectron", i, var)
        # veto muons
        for var in ("pfRelIso04_all", "mvaLowPt"):
            quick_addvar("VetoMuon", i, var)

    # number of objects
    for obj in (
            "jet", "deepjet_med", "electron", "muon", "lepton",
            "veto_electron", "veto_muon", "veto_lepton", "veto_tau",
    ):
        config.add_variable(
            name=f"cf_n_{obj}",
            expression=f"cutflow.n_{obj}",
            binning=(11, -0.5, 10.5),
            x_title=f"Number of {obj}s",
        )


def add_gen_variables(config: od.Config) -> None:
    """
    defines gen-level cutflow variables
    """
    for gp in ["h1", "h2", "b1", "b2", "wlep", "whad", "l", "nu", "q1", "q2", "sec1", "sec2"]:
        config.add_variable(
            name=f"gen_{gp}_pt",
            expression=f"cutflow.{gp}_pt",
            binning=(40, 0., 400.),
            unit="GeV",
            x_title=r"$p_{T, %s}^{gen}$" % (gp),
        )
        config.add_variable(
            name=f"gen_{gp}_mass",
            expression=f"cutflow.{gp}_mass",
            binning=(40, 0., 400.),
            unit="GeV",
            x_title=r"$m_{%s}^{gen}$" % (gp),
        )
        config.add_variable(
            name=f"gen_{gp}_eta",
            expression=f"cutflow.{gp}_eta",
            binning=(12, -6., 6.),
            unit="GeV",
            x_title=r"$\eta_{%s}^{gen}$" % (gp),
        )
        config.add_variable(
            name=f"gen_{gp}_phi",
            expression=f"cutflow.{gp}_phi",
            binning=(8, -4, 4),
            unit="GeV",
            x_title=r"$\phi_{%s}^{gen}$" % (gp),
        )
