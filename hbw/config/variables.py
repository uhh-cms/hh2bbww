# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT


def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    config.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )
    config.add_variable(
        name="ht",
        binning=[0, 80, 120, 160, 200, 240, 280, 320, 400, 500, 600, 800],
        unit="GeV",
        x_title="HT",
    )
    config.add_variable(
        name="n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
    )
    config.add_variable(
        name="n_electron",
        binning=(11, -0.5, 10.5),
        x_title="Number of electrons",
    )
    config.add_variable(
        name="n_muon",
        binning=(11, -0.5, 10.5),
        x_title="Number of muons",
    )

    config.add_variable(
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )

    # cutflow variables
    config.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="cf_n_jets",
        expression="cutflow.n_jet",
        binning=(11, -0.5, 10.5),
        x_title=r"Number of jets",
    )
    config.add_variable(
        name="cf_n_electron",
        expression="cutflow.n_electron",
        binning=(11, -0.5, 10.5),
        x_title=r"Number of electrons",
    )
    config.add_variable(
        name="cf_n_muon",
        expression="cutflow.n_muon",
        binning=(11, -0.5, 10.5),
        x_title=r"Number of muons",
    )
