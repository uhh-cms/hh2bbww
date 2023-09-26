# coding: utf-8

"""
Definition of ML input variables.
"""

import order as od

# from columnflow.columnar_util import EMPTY_FLOAT
from hbw.config.styling import default_var_binning, default_var_unit
from hbw.util import call_once_on_config


@call_once_on_config()
def add_ml_variables(config: od.Config) -> None:
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
        name="mli_l_deepjetsum",
        expression="mli_l_deepjetsum",
        binning=(40, 0, 4),
        x_title="sum of lightjet deepjet scores",
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
        name="mli_dr_jj",
        expression="mli_dr_jj",
        binning=(40, 0, 8),
        x_title=r"$\Delta R(j,j)$",
    )
    config.add_variable(
        name="mli_dphi_jj",
        expression="mli_dphi_jj",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(j,j)$",
    )
    config.add_variable(
        name="mli_mjj",
        expression="mli_mjj",
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"m(j,j)",
    )
    config.add_variable(
        name="mli_mindr_lj",
        expression="mli_mindr_lj",
        binning=(40, 0, 8),
        x_title=r"min $\Delta R(l,j)$",
    )
    config.add_variable(
        name="mli_dphi_lnu",
        expression="mli_dphi_lnu",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(l,\nu)$",
    )
    config.add_variable(
        name="mli_mlnu",
        expression="mli_mlnu",
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"$m(l,\nu)$",
    )
    config.add_variable(
        name="mli_mjjlnu",
        expression="mli_mjjlnu",
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"$m(jj,l\nu)$",
    )
    config.add_variable(
        name="mli_mjjl",
        expression="mli_mjjl",
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"$m(jj,l)$",
    )
    config.add_variable(
        name="mli_dphi_bb_jjlnu",
        expression="mli_dphi_bb_jjlnu",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(bb,jjl\nu)$",
    )
    config.add_variable(
        name="mli_dr_bb_jjlnu",
        expression="mli_dr_bb_jjlnu",
        binning=(40, 0, 8),
        x_title=r"$\Delta R(nn,jjlnu)$",
    )
    config.add_variable(
        name="mli_dphi_bb_jjl",
        expression="mli_dphi_bb_jjl",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(bb,jjl)$",
    )
    config.add_variable(
        name="mli_dr_bb_jjl",
        expression="mli_dr_bb_jjl",
        binning=(40, 0, 8),
        x_title=r"$\Delta R(nn,jjl)$",
    )
    config.add_variable(
        name="mli_dphi_bb_nu",
        expression="mli_dphi_bb_nu",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(bb,\nu)$",
    )
    config.add_variable(
        name="mli_dphi_jj_nu",
        expression="mli_dphi_jj_nu",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(jj,\nu)$",
    )
    config.add_variable(
        name="mli_dr_bb_l",
        expression="mli_dr_bb_l",
        binning=(40, 0, 8),
        x_title=r"$\Delta R(bb,l)$",
    )
    config.add_variable(
        name="mli_dr_jj_l",
        expression="mli_dr_jj_l",
        binning=(40, 0, 8),
        x_title=r"$\Delta R(jj,l)$",
    )
    config.add_variable(
        name="mli_mbbjjlnu",
        expression="mli_mbbjjlnu",
        binning=(40, 0, 800),
        unit="GeV",
        x_title=r"$m(bbjjlnu)$",
    )
    config.add_variable(
        name="mli_mbbjjl",
        expression="mli_mbbjjl",
        binning=(40, 0, 800),
        unit="GeV",
        x_title=r"$m(bbjjl)$",
    )
    config.add_variable(
        name="mli_s_min",
        expression="mli_s_min",
        binning=(50, 1, 10000),
        log_x=True,
        x_title=r"$S_{min}$",
    )
    config.add_variable(
        name="mli_vbf_deta",
        expression="mli_vbf_deta",
        binning=(50, 2, 9.5),
        x_title=r"$\Delta\eta(vbfjet1,vbfjet2)$",
    )
    config.add_variable(
        name="mli_vbf_invmass",
        expression="mli_vbf_invmass",
        binning=(50, 400, 4000),
        unit="GeV",
        x_title="invarint mass of two vbf jets",
    )
    config.add_variable(
        name="mli_vbf_tag",
        expression="mli_vbf_tag",
        binning=(2, -0.5, 1.5),
        x_title="existence of at least two vbf jets = 1, else 0",
    )

    for obj in ["b1", "b2", "j1", "j2", "lep", "met"]:
        for var in ["pt", "eta"]:
            config.add_variable(
                name=f"mli_{obj}_{var}",
                expression=f"mli_{obj}_{var}",
                binning=default_var_binning[var],
                unit=default_var_unit.get(var, var),
                x_title="{obj} {var}".format(obj=obj, var=var),
            )

    for obj in ["fj"]:
        for var in ["pt", "eta", "phi", "mass", "msoftdrop", "deepTagMD_HbbvsQCD"]:
            config.add_variable(
                name=f"mli_{obj}_{var}",
                expression=f"mli_{obj}_{var}",
                binning=default_var_binning[var],
                unit=default_var_unit.get(var, var),
                x_title="{obj} {var}".format(obj=obj, var=var),
            )
