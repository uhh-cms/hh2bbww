# coding: utf-8

"""
Definition of variables that can be plotted via `PlotCutflowVariables` tasks.
"""

import order as od

# from columnflow.columnar_util import EMPTY_FLOAT


# cutflow variables
def add_cutflow_variables(config: od.Config) -> None:
    # looose jets
    for i in range(4):
        config.add_variable(
            name=f"cf_loose_jet{i+1}_pt",
            expression=f"cutflow.loose_jet{i+1}_pt",
            binning=(40, 0, 100),
            unit="GeV",
            x_title=r"Loose jet %i $p_{T}$" % (i + 1),
        )
    # veto leptons
    for i in range(3):
        config.add_variable(
            name=f"cf_veto_lepton{i+1}_pt",
            expression=f"cutflow.veto_lepton{i+1}_pt",
            binning=(40, 0, 50),
            unit="GeV",
            x_title=r"Veto lepton %i $p_{T}$" % (i + 1),
        )

    # number of objects
    for obj in ("jet", "deepjet_med", "electron", "muon", "lepton", "veto_electron", "veto_muon", "veto_lepton"):
        config.add_variable(
            name=f"cf_n_{obj}",
            expression=f"cutflow.n_{obj}",
            binning=(11, -0.5, 10.5),
            x_title=f"Number of {obj}s",
        )


# Gen particles
def add_gen_variables(config: od.Config) -> None:
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
