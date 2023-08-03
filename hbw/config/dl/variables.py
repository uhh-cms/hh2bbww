# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")

from columnflow.columnar_util import EMPTY_FLOAT  # noqa


def add_dl_variables(config: od.Config) -> None:
    # bjet features
    config.add_variable(
        name="wp_score",
        expression="Bjet.btagDeepFlavB",  # NOTE: this gives the b-score sum over all bjets
        binning=(40, -0.5, 1.5),
        x_title="wp score",
    )
    # dl features
    config.add_variable(
        name="m_ll",
        binning=(40, 0., 80.),
        x_title=r"$m_{ll}$",
        unit="GeV",
    )
    config.add_variable(
        name="m_ll_check",
        binning=(40, 0., 80.),
        x_title=r"$m_{ll,test}$",
        unit="GeV",
    )
    config.add_variable(
        name="m_lljjMET",
        binning=(40, 0, 600),
        x_title=r"$m_{lljj \not{E_T}}}$",
        unit="GeV",
    )
    config.add_variable(
        name="channel_id",
        binning=(6, -0.5, 5.5),
        x_title="Channel Id",
        discrete_x=True,
    )
    config.add_variable(
        name="ll_pt",
        binning=(40, 0., 300),
        x_title=r"$dilepton \,\, system \,\, p_T$",
        unit="GeV",
    )
    config.add_variable(  # NOTE: this is no longer produced
        name="lep1_pt",
        binning=(40, 0., 200),
        x_title=r"$Leading\,\, lepton \,\, p_T$",
        unit="GeV",
    )
    config.add_variable(  # NOTE: this is no longer produced
        name="lep2_pt",
        binning=(40, 0., 200),
        x_title=r"$Subleading \,\, lepton \,\, p_T$",
        unit="GeV",
    )
    config.add_variable(
        name="charge",
        binning=(3, -1.5, 1.5),
        x_title=r"$Charge$",
        discrete_x=True,
    )
    config.add_variable(
        name="deltaR_ll",
        binning=(40, 0., 4),
        x_title=r"$\Delta R (l,l)$",
    )
    config.add_variable(
        name="E_miss",
        expression="MET.pt",
        binning=(40, 0., 250),
        x_title=r"$E_T \not$",
        unit="GeV",
    )
    config.add_variable(
        name="MT",
        binning=(40, 0., 300),
        x_title=r"$MT$",
        unit="GeV",
    )
    config.add_variable(
        name="min_dr_lljj",
        binning=(40, 0, 4),
        x_title=r"$min \Delta R(l,b)$",
    )
    config.add_variable(
        name="delta_Phi",
        binning=(40, 0, 3),
        x_title=r"$ \Delta \phi(ll,jj)$",
    )
