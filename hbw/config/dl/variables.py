# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")

from hbw.config.styling import default_var_binning, default_var_unit
from hbw.util import call_once_on_config


@call_once_on_config()
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
        x_title=r"$m_{lljj Emiss}$",
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


@call_once_on_config()
def add_dl_ml_variables(config: od.Config) -> None:
    """
    Adds ML input variables to a *config*.
    """

    # reconstructed variables
    config.add_variable(
        name="mli_ht",
        expression="mli_ht",
        binning=(40, 0, 1200),
        unit="GeV",
        x_title="HT",
    )
    config.add_variable(
        name="mli_n_jet",
        expression="mli_n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
    )
    config.add_variable(
        name="mli_n_deepjet",
        expression="mli_n_deepjet",
        binning=(11, -0.5, 10.5),
        x_title="Number of b-tagged jets (deepjet medium WP)",
    )
    config.add_variable(
        name="mli_deepjetsum",
        expression="mli_deepjetsum",
        binning=(40, 0, 4),
        x_title="sum of deepjet scores",
    )
    config.add_variable(
        name="mli_b_deepjetsum",
        expression="mli_b_deepjetsum",
        binning=(40, 0, 4),
        x_title="sum of bjet deepjet scores",
    )
    config.add_variable(
        name="mli_dr_bb",
        expression="mli_dr_bb",
        binning=(40, 0, 8),
        x_title=r"$\Delta R(b,b)$",
    )
    config.add_variable(
        name="mli_dphi_bb",
        expression="mli_dphi_bb",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(b,b)$",
    )
    config.add_variable(
        name="mli_mbb",
        expression="mli_mbb",
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"m(b,b)",
    )
    config.add_variable(
        name="mli_mindr_lb",
        expression="mli_mindr_lb",
        binning=(40, 0, 8),
        x_title=r"min $\Delta R(l,b)$",
    )
    config.add_variable(
        name="mli_dphi_bb_nu",
        expression="mli_dphi_bb_nu",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(bb,\nu)$",
    )
    config.add_variable(
        name="mli_dr_bb_l",
        expression="mli_dr_bb_l",
        binning=(40, 0, 8),
        x_title=r"$\Delta R(bb,l)$",
    )
    config.add_variable(
        name="mli_mll",
        expression="mli_mll",
        binning=(40, 0, 80),
        x_title=r"$m_{ll}$",
    )
    config.add_variable(
        name="mli_dr_ll",
        expression="mli_dr_ll",
        binning=(40, 0, 8),
        x_title=r"$\Delta R(ll)$",
    )
    config.add_variable(
        name="mli__min_dr_llbb",
        expression="mli_min_dr_llbb",
        binning=(40, 0, 8),
        x_title=r"$\Delta R(bb,ll)$",
    )
    config.add_variable(
        name="mli_bb_pt",
        expression="mli_bb_pt",
        binning=(40, 0, 500),
        unit="GeV",
        x_title=r"$bb_p_T$",
    )
    config.add_variable(
        name="mli_mllMET",
        expression="mli_mllMET",
        binning=(40, 0, 200),
        x_title=r"$m_{llMET}$",
    )
    config.add_variable(
        name="mli_dr_bb_llMET",
        expression="mli_dr_bb_llMET",
        binning=(40, 0, 8),
        x_title=r"$\Delta R(bb,llMET)$",
    )
    config.add_variable(
        name="mli_dphi_bb_llMET",
        expression="mli_dphi_bb_llMET",
        binning=(40, 0, 8),
        x_title=r"$\Delta \phi(bb,llMET)$",
    )
    config.add_variable(
        name="mli_mbbllMET",
        expression="mli_mbbllMET",
        binning=(40, 0, 500),
        unit="GeV",
        x_title=r"$m_{bbllMET}$",
    )
    config.add_variable(
        name="mli_dphi_ll",
        expression="mli_dphi_ll",
        binning=(40, 0, 8),
        unit="GeV",
        x_title=r"$\Delta \phi_{ll}$",
    )
    config.add_variable(
        name="mli_ll_pt",
        expression="mli_ll_pt",
        binning=(40, 0, 200),
        unit="GeV",
        x_title=r"$ll p_T$",
    )

    for obj in ["b1", "b2", "lep", "lep2", "met"]:
        for var in ["pt", "eta"]:
            if var == "eta" and obj == "met":
                continue
            config.add_variable(
                name=f"mli_{obj}_{var}",
                expression=f"mli_{obj}_{var}",
                binning=default_var_binning[var],
                unit=default_var_unit.get(var, var),
                x_title="{obj} {var}".format(obj=obj, var=var),
            )
