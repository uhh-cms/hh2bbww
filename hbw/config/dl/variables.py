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
        binning=(40, 0, 1200),
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
        x_title=r"$min_{b,l} \Delta R(l,b)$",
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
    config.add_variable(
        name="mli_mbbllMET_rebinned3",
        expression="mli_mbbllMET",
        binning=[
            100, 200, 230, 260, 290, 320, 350, 380, 410, 440, 470, 500, 530, 560, 590, 620,
            660, 700, 750, 800, 900, 1000, 1200,
        ],
        unit="GeV",
        x_title=r"$m_{HH}$",
    )
    config.add_variable(
        name="mli_bb_pt_rebinned3",
        expression="mli_bb_pt",
        binning=[
            0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 165, 180, 200,
            230, 260, 300,
        ],
        unit="GeV",
        x_title=r"$p_{T}^{bb}$",
    )
    config.add_variable(
        name="mli_b1_pt_rebinned3",
        expression="mli_b1_pt",
        binning=[
            0, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 180, 200,
            230, 260, 300,
        ],
        unit="GeV",
        x_title=r"$p_{T}$ of jet with highest b-tagging score",
        aux={
            "x_min": 25,
        },
    )
    config.add_variable(
        name="mli_mbb_rebinned3",
        expression="mli_mbb",
        binning=[
            0, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 370, 400,
            440, 500, 580, 660, 800,
        ],
        unit="GeV",
        x_title=r"$m_{bb}$",
    )

    # reconstructed variables
    config.add_variable(
        name="mli_dphi_bb_nu",
        expression="mli_dphi_bb_nu",
        binning=(40, 0, 3.2),
        aux={"overflow": True},
        x_title=r"$\Delta\Phi(bb,\nu)$",
    )
    config.add_variable(
        name="mli_mll",
        expression="mli_mll",
        binning=(40, 0, 160),
        aux={
            "overflow": True,
            "x_min": 20,
        },
        x_title=r"$m_{\ell\ell}$",
    )
    config.add_variable(
        name="mli_dr_ll",
        expression="mli_dr_ll",
        binning=(40, 0, 6),
        aux={"overflow": True},
        x_title=r"$\Delta R(\ell,\ell)$",
    )
    config.add_variable(
        name="mli_min_dr_llbb",
        expression="mli_min_dr_llbb",
        binning=(40, 0, 6),
        aux={"overflow": True},
        x_title=r"$min_{b,l} \Delta R(b,\ell)$",
    )
    config.add_variable(
        name="mli_dr_ll_bb",
        expression="mli_dr_ll_bb",
        binning=(40, 0, 6),
        aux={"overflow": True},
        x_title=r"$\Delta R(bb,\ell\ell)$",
    )
    config.add_variable(
        name="mli_mllMET",
        expression="mli_mllMET",
        binning=(40, 0, 600),
        aux={"overflow": True},
        x_title=r"$m_{\ell\ell MET}$",
    )
    config.add_variable(
        name="mli_dr_bb_llMET",
        expression="mli_dr_bb_llMET",
        binning=(40, 0, 6),
        aux={"overflow": True},
        x_title=r"$\Delta R(bb,\ell\ell MET)$",
    )
    config.add_variable(
        name="mli_dphi_bb_llMET",
        expression="mli_dphi_bb_llMET",
        binning=(64, 0, 3.2),
        aux={"overflow": True},
        x_title=r"$\Delta \phi(bb,\ell\ell MET)$",
    )
    config.add_variable(
        name="mli_mbbllMET",
        expression="mli_mbbllMET",
        binning=(40, 0, 1200),
        aux={"overflow": True},
        unit="GeV",
        # x_title=r"$m_{bb \ell\ell MET}$",
        x_title=r"$m_{HH}$",
    )
    config.add_variable(
        name="mli_dphi_ll",
        expression="mli_dphi_ll",
        binning=(64, 0, 3.2),
        aux={"overflow": True},
        x_title=r"$\Delta \phi_{\ell\ell}$",
    )
    config.add_variable(
        name="mli_deta_ll",
        expression="mli_deta_ll",
        binning=(40, 0, 6),
        aux={"overflow": True},
        x_title=r"$\Delta \eta_{\ell\ell}$",
    )
    config.add_variable(
        name="mli_ll_pt",
        expression="mli_ll_pt",
        binning=(40, 0, 200),
        aux={"overflow": True},
        unit="GeV",
        x_title=r"$p_{T}^{\ell\ell}$",
    )

    for obj in ["lep2"]:
        for var in ["pt", "eta"]:
            binning = default_var_binning[var]
            if obj == "lep2" and var == "pt":
                # TODO: change to 160? 120? idk.
                binning = (40, 0, 240)
            config.add_variable(
                name=f"mli_{obj}_{var}",
                expression=f"mli_{obj}_{var}",
                binning=binning,
                aux={"overflow": True},
                unit=default_var_unit.get(var, "1"),
                x_title="{obj} {var}".format(obj=obj, var=var),
            )

    config.add_variable(
        name="mli_lep_tag",
        expression="mli_lep_tag",
        binning=(2, -.5, 1.5),
        aux={"overflow": True},
        x_title="lepton 1 muon tag",
    )

    config.add_variable(
        name="mli_lep2_tag",
        expression="mli_lep2_tag",
        binning=(2, -0.5, 1.5),
        aux={"overflow": True},
        x_title="lepton 2 muon tag",
    )

    config.add_variable(
        name="mli_mixed_channel",
        expression="mli_mixed_channel",
        binning=(2, -0.5, 1.5),
        aux={"overflow": True},
        x_title="Mixed channel tag",
    )
