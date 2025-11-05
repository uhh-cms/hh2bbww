# coding: utf-8

"""
Definition of ML input variables.
"""

import order as od

# from columnflow.columnar_util import EMPTY_FLOAT
from hbw.config.styling import default_var_binning, default_var_unit, default_var_title_format
from hbw.util import call_once_on_config


@call_once_on_config()
def add_common_ml_variables(config: od.Config) -> None:
    """
    Adds common ML input variables to a *config*.
    """

    for postfix, object_label in (
        ("", "central jets"),
        ("_alljets", "central + forward jets"),
        ("_fwjets", "forward jets"),
    ):
        config.add_variable(
            name=f"mli_ht{postfix}",
            expression=f"mli_ht{postfix}",
            binning=(40, 0, 1200),
            unit="GeV",
            x_title=f"HT ({object_label})",
            aux={"overflow": True},
        )
        config.add_variable(
            name=f"mli_n_jet{postfix}",
            expression=f"mli_n_jet{postfix}",
            binning=(11, -0.5, 10.5),
            x_title=f"Number of {object_label}",
            aux={"overflow": True},
        )

    config.add_variable(
        name="mli_lt",
        expression="mli_lt",
        binning=(40, 0, 800),
        unit="GeV",
        x_title="LT",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_n_btag",
        expression="mli_n_btag",
        binning=(11, -0.5, 10.5),
        x_title="Number of b-tagged jets",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_b_score_sum",
        expression="mli_b_score_sum",
        binning=(40, 0, 4),
        x_title="sum of btag scores",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_b_b_score_sum",
        expression="mli_b_b_score_sum",
        binning=(40, 0, 4),
        x_title="sum of bjet btag scores",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_l_b_score_sum",
        expression="mli_l_b_score_sum",
        binning=(40, 0, 4),
        x_title="sum of lightjet btag scores",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_bb_pt",
        expression="mli_bb_pt",
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"$p_{T}^{bb}$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_dr_bb",
        expression="mli_dr_bb",
        binning=(40, 0, 6),
        x_title=r"$\Delta R(b,b)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_dphi_bb",
        expression="mli_dphi_bb",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(b,b)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_deta_bb",
        expression="mli_deta_bb",
        binning=(40, 0, 6),
        x_title=r"$\Delta\eta(b,b)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_mbb",
        expression="mli_mbb",
        binning=(40, 0, 800),
        unit="GeV",
        x_title=r"$m_{bb}$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_mindr_lb",
        expression="mli_mindr_lb",
        binning=(40, 0, 6),
        x_title=r"min $\Delta R(\ell0,b)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_mindr_lj",
        expression="mli_mindr_lj",
        binning=(40, 0, 8),
        x_title=r"min $\Delta R(\ell,j)$",
        aux={"overflow": True},
    )
    for postfix, object_label in (
        ("", "central jets"),
        ("_alljets", "central + forward jets"),
    ):
        config.add_variable(
            name=f"mli_mindr_jj{postfix}",
            expression=f"mli_mindr_jj{postfix}",
            binning=(40, 0, 6),
            x_title=rf"min $\Delta R(j,j)$ ({object_label})",
            aux={"overflow": True},
        )
        config.add_variable(
            name=f"mli_maxdr_jj{postfix}",
            expression=f"mli_maxdr_jj{postfix}",
            binning=(40, 0, 6),
            x_title=rf"max $\Delta R(j,j)$ ({object_label})",
            aux={"overflow": True},
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
            aux={"overflow": True},
        )
        config.add_variable(
            name=f"mli_{prefix}vbf_phi",
            expression=f"mli_{prefix}vbf_phi",
            binning=(50, -3.2, 3.2),
            x_title=rf"VBF pair $\phi$ ($|\eta| < {eta_range}|$)",
            aux={"overflow": True},
        )
        config.add_variable(
            name=f"mli_{prefix}vbf_eta",
            expression=f"mli_{prefix}vbf_eta",
            binning=(48, -4.7, 4.7),
            x_title=rf"VBF pair $\eta$ ($|\eta| < {eta_range}|$)",
            aux={"overflow": True},
        )
        config.add_variable(
            name=f"mli_{prefix}vbf_deta",
            expression=f"mli_{prefix}vbf_deta",
            binning=(50, 2, 9.5),
            x_title=rf"VBF pair $\Delta\eta$ ($|\eta| < {eta_range}|$)",
            aux={"overflow": True},
        )
        config.add_variable(
            name=f"mli_{prefix}vbf_mass",
            expression=f"mli_{prefix}vbf_mass",
            binning=(50, 400, 4000),
            unit="GeV",
            aux={"overflow": True},
            x_title=rf"VBF pair mass ($|\eta| < {eta_range}|$)",
        )
        config.add_variable(
            name=f"mli_{prefix}vbf_tag",
            expression=f"mli_{prefix}vbf_tag",
            binning=(2, -0.5, 1.5),
            x_title=rf"VBF pair tag ($|\eta| < {eta_range}|$)",
            aux={"overflow": True},
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
                aux={"overflow": True},
            )

    for obj in ["b1", "b2", "j1", "j2", "vbfcand1", "vbfcand2", "lep", "met"]:
        for var in ["pt", "eta", "phi"]:
            if var == "eta" and obj == "met":
                continue
            if var == "phi" and obj != "met":
                continue
            binning = default_var_binning[var]
            if "vbfcand" in obj and var == "eta":
                binning = (48, -4.7, 4.7)
            elif obj == "lep" and var == "pt":
                binning = (40, 0, 240)
            config.add_variable(
                name=f"mli_{obj}_{var}",
                expression=f"mli_{obj}_{var}",
                binning=binning,
                unit=default_var_unit.get(var, "1"),
                x_title="{obj} {var}".format(obj=obj, var=var),
                aux={"overflow": True},
            )

    for obj in ["fj"]:
        obj_label = {"fj": "FatJet"}[obj]
        for var in ["pt", "eta", "phi", "mass", "msoftdrop", "particleNet_XbbVsQCD", "particleNetWithMass_HbbvsQCD"]:
            var_label = default_var_title_format.get(var, var)
            config.add_variable(
                name=f"mli_{obj}_{var}",
                expression=f"mli_{obj}_{var}",
                binning=default_var_binning[var],
                unit=default_var_unit.get(var, "1"),
                x_title="{obj} {var} (Hbb-score leading)".format(obj=obj_label, var=var_label),
                aux={"overflow": True},
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
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_dphi_jj",
        expression="mli_dphi_jj",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(j,j)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_mjj",
        expression="mli_mjj",
        binning=(40, 0, 400),
        unit="GeV",
        aux={"overflow": True},
        x_title=r"m(j,j)",
    )
    config.add_variable(
        name="mli_dphi_lnu",
        expression="mli_dphi_lnu",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(\elll,\nu)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_dphi_wl",
        expression="mli_dphi_wl",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(W,\ell)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_mlnu",
        expression="mli_mlnu",
        binning=(40, 0, 400),
        unit="GeV",
        aux={"overflow": True},
        x_title=r"$m(\ell,\nu)$",
    )
    config.add_variable(
        name="mli_mjjlnu",
        expression="mli_mjjlnu",
        binning=(40, 0, 400),
        unit="GeV",
        aux={"overflow": True},
        x_title=r"$m(jj,\ell\nu)$",
    )
    config.add_variable(
        name="mli_mjjl",
        expression="mli_mjjl",
        binning=(40, 0, 400),
        unit="GeV",
        aux={"overflow": True},
        x_title=r"$m(jj,\ell)$",
    )
    config.add_variable(
        name="mli_dphi_bb_jjlnu",
        expression="mli_dphi_bb_jjlnu",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(bb,jj\ell\nu)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_dr_bb_jjlnu",
        expression="mli_dr_bb_jjlnu",
        binning=(40, 0, 6),
        x_title=r"$\Delta R(bb,jj\ell\nu)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_dphi_bb_jjl",
        expression="mli_dphi_bb_jjl",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(bb,jj\ell)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_dr_bb_jjl",
        expression="mli_dr_bb_jjl",
        binning=(40, 0, 6),
        x_title=r"$\Delta R(bb,jj\ell)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_dphi_bb_nu",
        expression="mli_dphi_bb_nu",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(bb,\nu)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_dphi_jj_nu",
        expression="mli_dphi_jj_nu",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta\Phi(jj,\nu)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_dr_bb_l",
        expression="mli_dr_bb_l",
        binning=(40, 0, 6),
        x_title=r"$\Delta R(bb,\ell)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_dr_jj_l",
        expression="mli_dr_jj_l",
        binning=(40, 0, 6),
        x_title=r"$\Delta R(jj,\ell)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_mbbjjlnu",
        expression="mli_mbbjjlnu",
        binning=(40, 0, 800),
        unit="GeV",
        x_title=r"$m(bbjj\ell\nu)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_mbbjjl",
        expression="mli_mbbjjl",
        binning=(40, 0, 800),
        unit="GeV",
        x_title=r"$m(bbjj\ell)$",
        aux={"overflow": True},
    )
    config.add_variable(
        name="mli_s_min",
        expression="mli_s_min",
        binning=(40, 1, 10000),
        log_x=True,
        x_title=r"$S_{min}$",
        aux={"overflow": True},
    )
