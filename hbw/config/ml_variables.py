# coding: utf-8

"""
Definition of ML input variables.
"""

import order as od

# from columnflow.columnar_util import EMPTY_FLOAT
from hbw.config.styling import default_var_binning, default_var_unit
from hbw.util import call_once_on_config


@call_once_on_config()
def add_common_ml_variables(config: od.Config) -> None:
    """
    Adds common ML input variables to a *config*.
    """

    config.add_variable(
        name="mli_ht",
        expression="mli_ht",
        binning=(40, 0, 1200),
        unit="GeV",
        x_title="HT",
    )
    config.add_variable(
        name="mli_lt",
        expression="mli_lt",
        binning=(40, 0, 800),
        unit="GeV",
        x_title="LT",
    )
    config.add_variable(
        name="mli_n_jet",
        expression="mli_n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
    )
    config.add_variable(
        name="mli_n_btag",
        expression="mli_n_btag",
        binning=(11, -0.5, 10.5),
        x_title="Number of b-tagged jets",
    )
    config.add_variable(
        name="mli_b_score_sum",
        expression="mli_b_score_sum",
        binning=(40, 0, 4),
        x_title="sum of btag scores",
    )
    config.add_variable(
        name="mli_b_b_score_sum",
        expression="mli_b_b_score_sum",
        binning=(40, 0, 4),
        x_title="sum of bjet btag scores",
    )
    config.add_variable(
        name="mli_l_b_score_sum",
        expression="mli_l_b_score_sum",
        binning=(40, 0, 4),
        x_title="sum of lightjet btag scores",
    )
    config.add_variable(
        name="mli_bb_pt",
        expression="mli_bb_pt",
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"$p_{T}^{bb}$",
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
        name="mli_deta_bb",
        expression="mli_deta_bb",
        binning=(40, 0, 6),
        x_title=r"$\Delta\eta(b,b)$",
    )
    config.add_variable(
        name="mli_mbb",
        expression="mli_mbb",
        binning=(40, 0, 1200),
        unit="GeV",
        aux={"overflow": True},
        x_title=r"$m_{bb}$",
    )
    config.add_variable(
        name="mli_mindr_lb",
        expression="mli_mindr_lb",
        binning=(40, 0, 8),
        x_title=r"min $\Delta R(l,b)$",
    )
    config.add_variable(
        name="mli_mindr_lj",
        expression="mli_mindr_lj",
        binning=(40, 0, 8),
        x_title=r"min $\Delta R(l,j)$",
    )
    config.add_variable(
        name="mli_mindr_jj",
        expression="mli_mindr_jj",
        binning=(40, 0, 8),
        x_title=r"min $\Delta R(j,j)$",
    )
    config.add_variable(
        name="mli_maxdr_jj",
        expression="mli_maxdr_jj",
        binning=(40, 0, 12),
        x_title=r"max $\Delta R(j,j)$",
    )

    # vbf features for central jets and incljets
    for eta_range, prefix in (
        ("2.4", ""),
        ("4.7", "full_"),
    ):
        config.add_variable(
            name=f"mli_{prefix}vbf_pt",
            expression=f"mli_{prefix}vbf_pt",
            binning=(40, 0, 1000),
            x_title=rf"VBF pair $p_{{T}}$ ($|\eta| < {eta_range}|$)",
        )
        config.add_variable(
            name=f"mli_{prefix}vbf_phi",
            expression=f"mli_{prefix}vbf_phi",
            binning=(50, -3.2, 3.2),
            x_title=rf"VBF pair $\phi$ ($|\eta| < {eta_range}|$)",
        )
        config.add_variable(
            name=f"mli_{prefix}vbf_eta",
            expression=f"mli_{prefix}vbf_eta",
            binning=(48, -4.7, 4.7),
            x_title=rf"VBF pair $\eta$ ($|\eta| < {eta_range}|$)",
        )
        config.add_variable(
            name=f"mli_{prefix}vbf_deta",
            expression=f"mli_{prefix}vbf_deta",
            binning=(50, 2, 9.5),
            x_title=rf"VBF pair $\Delta\eta$ ($|\eta| < {eta_range}|$)",
        )
        config.add_variable(
            name=f"mli_{prefix}vbf_mass",
            expression=f"mli_{prefix}vbf_mass",
            binning=(50, 400, 4000),
            unit="GeV",
            x_title=rf"VBF pair mass ($|\eta| < {eta_range}|$)",
        )
        config.add_variable(
            name=f"mli_{prefix}vbf_tag",
            expression=f"mli_{prefix}vbf_tag",
            binning=(2, -0.5, 1.5),
            x_title=rf"VBF pair tag ($|\eta| < {eta_range}|$)",
        )

    #
    # low-level variables
    #

    for obj in ["b1", "b2", "j1", "j2"]:
        for var in ["b_score"]:
            config.add_variable(
                name=f"mli_{obj}_{var}",
                expression=f"mli_{obj}_{var}",
                binning=default_var_binning[var],
                unit=default_var_unit.get(var, "1"),
                x_title="{obj} {var}".format(obj=obj, var=var),
            )

    for obj in ["b1", "b2", "j1", "j2", "lep", "met"]:
        for var in ["pt", "eta", "phi"]:
            if var == "eta" and obj == "met":
                continue
            if var == "phi" and obj != "met":
                continue
            config.add_variable(
                name=f"mli_{obj}_{var}",
                expression=f"mli_{obj}_{var}",
                binning=default_var_binning[var],
                unit=default_var_unit.get(var, "1"),
                x_title="{obj} {var}".format(obj=obj, var=var),
            )

    for obj in ["fj"]:
        for var in ["pt", "eta", "phi", "mass", "msoftdrop", "particleNet_XbbVsQCD", "particleNetWithMass_HbbvsQCD"]:
            config.add_variable(
                name=f"mli_{obj}_{var}",
                expression=f"mli_{obj}_{var}",
                binning=default_var_binning[var],
                unit=default_var_unit.get(var, "1"),
                x_title="{obj} {var}".format(obj=obj, var=var),
            )

        b1_pt = config.get_variable("mli_b1_pt")
        b1_pt.x_title = r"$p_{T}^{b1}$"


@call_once_on_config()
def add_sl_ml_variables(config: od.Config) -> None:
    """
    Adds SL ML input variables to a *config*.
    """
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
        name="mli_dphi_lnu",
        expression="mli_dphi_lnu",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(l,\nu)$",
    )
    config.add_variable(
        name="mli_dphi_wl",
        expression="mli_dphi_wl",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(W,l)$",
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
        binning=(40, 1, 10000),
        log_x=True,
        x_title=r"$S_{min}$",
    )
