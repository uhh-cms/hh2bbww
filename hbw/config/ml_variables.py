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
        # ("", "central jets"),
        ("", "jets"),
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
            aux={
                "overflow": True,
                "x_min": 0.5,
            },
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
        name="b_score_sum_discrete",
        expression="mli_b_score_sum",
        binning=[
            0.0, 0.0246, 0.0492, 0.0738, 0.0984, 0.1272, 0.1518, 0.1764, 0.201, 0.2544,
            0.279, 0.3036, 0.3816, 0.4062, 0.4648, 0.4894, 0.5088, 0.514, 0.5386, 0.592,
            0.6166, 0.6298, 0.6412, 0.6544, 0.679, 0.7036, 0.7192, 0.7438, 0.757, 0.7816,
            0.8062, 0.8464, 0.8842, 0.9088, 0.9296, 0.9542, 0.9739, 0.9788, 0.9985, 1.0,
            1.0114, 1.0231, 1.0246, 1.0477, 1.0492, 1.0568, 1.0738, 1.0814, 1.0946, 1.1011,
            1.1192, 1.1257, 1.1272, 1.1438, 1.1503, 1.1518, 1.1764, 1.184, 1.2218, 1.2283,
            1.2464, 1.2529, 1.2544, 1.2596, 1.279, 1.2842, 1.3088, 1.349, 1.3555, 1.3816,
            1.3868, 1.3944, 1.4114, 1.419, 1.4387, 1.4633, 1.4648, 1.4879, 1.4894, 1.514,
            1.5216, 1.5594, 1.5659, 1.584, 1.5905, 1.592, 1.6037, 1.6166, 1.6283, 1.6298,
            1.6529, 1.6544, 1.679, 1.6866, 1.6931, 1.7192, 1.7244, 1.7309, 1.749, 1.7555,
            1.757, 1.7816, 1.8516, 1.8581, 1.8592, 1.8842, 1.8894, 1.9035, 1.914, 1.9281,
            1.9296, 1.9478, 1.9542, 1.9724, 1.9739, 1.997, 1.9985, 2.0, 2.0166, 2.0231,
            2.0242, 2.0246, 2.0307, 2.0492, 2.0568, 2.0685, 2.075, 2.0931, 2.0946, 2.0996,
            2.1011, 2.1192, 2.1257, 2.1272, 2.1518, 2.1892, 2.1957, 2.2022, 2.2218, 2.2283,
            2.2335, 2.2544, 2.2581, 2.2596, 2.2842, 2.3542, 2.3607, 2.3683, 2.3868, 2.3944,
            2.4126, 2.4372, 2.4387, 2.4633, 2.4648, 2.4894, 2.5192, 2.5333, 2.5398, 2.5594,
            2.5659, 2.5776, 2.592, 2.6022, 2.6037, 2.6283, 2.6298, 2.6544, 2.6983, 2.7048,
            2.7244, 2.7309, 2.757, 2.8633, 2.8774, 2.8894, 2.9035, 2.9217, 2.9296, 2.9463,
            2.9478, 2.9724, 2.9739, 2.9985, 3.0, 3.0246, 3.0424, 3.0489, 3.0685, 3.075,
            3.0946, 3.1011, 3.1272, 3.2074, 3.2335, 3.2596, 3.3865, 3.4126, 3.4387, 3.4648,
            3.5515, 3.5776, 3.6037, 3.6298, 3.8956, 3.9217, 3.9478, 3.9739, 4.0,
        ],
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
            binning=(40, 0, 8),
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
            aux={"overflow": True, "rebin": 2},
        )
        config.add_variable(
            name=f"mli_{prefix}vbf_mass",
            expression=f"mli_{prefix}vbf_mass",
            binning=(40, 0, 4000),
            unit="GeV",
            aux={"overflow": True, "rebin": 2},
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

    for obj in ["b1", "b2", "j1", "j2"]:
        for var in ["b_score"]:
            config.add_variable(
                name=f"check_{obj}_{var}",
                expression=f"mli_{obj}_{var}",
                binning=[0.0, 0.0246, 0.1272, 0.4648, 0.6298, 0.9739, 1.0],
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
