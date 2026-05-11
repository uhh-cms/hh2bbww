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
            "x_min": 10,
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


@call_once_on_config()
def add_hhh_dl_ml_variables(config: od.Config) -> None:
    config.add_variable(
        name="mli_mbb1",
        expression="mli_mbb1",
        binning=(40, 0, 1200),
        aux={"overflow": True},
        unit="GeV",
        x_title=r"$m_{bb1}$",
    )
    config.add_variable(
        name="mli_mbb2",
        expression="mli_mbb2",
        binning=(40, 0, 1200),
        aux={"overflow": True},
        unit="GeV",
        x_title=r"$m_{bb2}$",
    )
    config.add_variable(
        name="mli_dr_bb_bb",
        expression="mli_dr_bb_bb",
        binning=(40, 0, 6),
        aux={"overflow": True},
        x_title=r"$\Delta R(bb1, bb2)$",
    )
    config.add_variable(
        name="mli_dr_ll_bb1",
        expression="mli_dr_ll_bb1",
        binning=(40, 0, 6),
        aux={"overflow": True},
        x_title=r"$\Delta R(ll, bb1)$",
    )
    config.add_variable(
        name="mli_dr_ll_bb2",
        expression="mli_dr_ll_bb2",
        binning=(40, 0, 6),
        aux={"overflow": True},
        x_title=r"$\Delta R(ll, bb2)$",
    )
    config.add_variable(
        name="mli_mhhh",
        expression="mli_mhhh",
        binning=(40, 0, 1200),
        aux={"overflow": True},
        unit="GeV",
        x_title=r"$m_{HHH}$",
    )
    config.add_variable(
        name="mli_m4bllMET",
        expression="mli_m4bllMET",
        binning=(40, 0, 1200),
        aux={"overflow": True},
        unit="GeV",
        x_title=r"$m_{4b ll MET}$",
    )
    config.add_variable(
        name="mli_dr_bb1_llMET",
        expression="mli_dr_bb1_llMET",
        binning=(40, 0, 6),
        aux={"overflow": True},
        x_title=r"$\Delta R(bb1, ll MET)$",
    )
    config.add_variable(
        name="mli_dr_bb2_llMET",
        expression="mli_dr_bb2_llMET",
        binning=(40, 0, 6),
        aux={"overflow": True},
        x_title=r"$\Delta R(bb2, ll MET)$",
    )


@call_once_on_config()
def add_hhh_bjet_variables(config: od.Config) -> None:
    config.add_variable(
        name="hhh_dr_bb",
        expression="hhh_dr_bb",
        binning=(40, 0, 6),
        aux={"overflow": False},
        x_title=r"$ \Delta R(b,b)$",
    )
    config.add_variable(
        name="mli_mindr_bb",
        expression="mli_mindr_bb",
        binning=(40, 0, 6),
        aux={"overflow": False},
        x_title=r"$min_{b,b} \Delta R(b,b)$",
    )
    config.add_variable(
        name="mli_maxdr_bb",
        expression="mli_maxdr_bb",
        binning=(40, 0, 6),
        aux={"overflow": False},
        x_title=r"$max_{b,b} \Delta R(b,b)$",
    )
    config.add_variable(
        name="discrete_sum_b_score",
        expression="discrete_sum_b_score",
        binning=(40, 0, 6),
        aux={"overflow": False},
        x_title=r"$sum b score, discrete$",
    )
    config.add_variable(
        name="check_n_btag",
        expression="check_n_btag",
        binning=(40, 0, 6),
        aux={"overflow": False},
        x_title=r"$sum b score, discrete$",
    )
    config.add_variable(
        name="b_score_sum_check",
        expression="discrete_sum_b_score",
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
            3.5515, 3.5776, 3.6037, 3.6298, 3.8956, 3.9217, 3.9478, 3.9739, 4.0
        ],
        x_title="sum of btag scores",
        aux={"overflow": True},
    )
    for i in range(0, 2):
        config.add_variable(
            name=f"mli_bjet{i+1}_pt",
            expression=f"btag_jet{i+1}.pt",
            binning=(40, 0, 400),
            aux={"overflow": True},
            x_title=f"p_T(b{i+1})",
        )
        config.add_variable(
            name=f"mli_bjet{i+1}_eta",
            expression=f"btag_jet{i+1}.eta",
            binning=(40, -4, 4),
            aux={"overflow": True},
            x_title=f"eta(b{i+1})",
        )
        config.add_variable(
            name=f"mli_bjet{i+1}_phi",
            expression=f"btag_jet{i+1}.phi",
            binning=(40, -4, 4),
            aux={"overflow": True},
            x_title=f"phi(b{i+1})",
        )
        config.add_variable(
            name=f"mli_bjet{i+1}_btagUParTAK4B",
            expression=f"btag_jet{i+1}.btagUParTAK4B",
            binning=(40, 0, 1),
            aux={"overflow": True},
            x_title=f"$b{i+1} btagUParTAK4B$",
        )
        config.add_variable(
            name=f"mli_bjet{i+1}_discrete_b_score",
            expression=f"btag_jet{i+1}.discrete_b_score",
            binning=[0.0, 0.0246, 0.1272, 0.4648, 0.6298, 0.9739, 1.0],
            aux={"overflow": True},
            x_title=f"$b{i+1} b score$",
        )
    for i in range(0, 4):
        config.add_variable(
            name=f"mli_jet{i+1}_discrete_b_score",
            expression=f"Jet.discrete_b_score[:, {i}]",
            null_value=-9999,
            binning=[0.0, 0.0246, 0.1272, 0.4648, 0.6298, 0.9739, 1.0],
            aux={"overflow": True},
            x_title=f"$Jet{i+1} b score$",
        )


@call_once_on_config()
def add_hh_bjet_variables(config: od.Config) -> None:
    label_dict = {
        "": "",
        "_b": "(BJets)",
        "_l": "(LightJets)",
    }
    for obj in ["", "_b", "_l"]:
        config.add_variable(
            name=f"mli{obj}_discrete_b_score_sum",
            binning=(16, -0.5, 15.5),
            aux={"overflow": True},
            x_title=f"sum of discrete b-scores {label_dict[obj]}",
        )
    for obj in ["b1", "b2", "j1", "j2"]:
        config.add_variable(
            name=f"mli_{obj}_discrete_b_score",
            binning=(6, -0.5, 5.5),
            aux={"overflow": True},
            x_title=f"{obj} discrete b-score",
        )
    config.add_variable(
        name="test_discrete_sum_b_score",
        expression="mli_discrete_b_score_sum",
        binning=(16, 0, 16),
        aux={"overflow": True},
        x_title="sum of discrete b-scores (test)",
    )
    for i in range(4):
        config.add_variable(
            name=f"mli_jet{i+1}_discrete_b_score",
            expression=f"Jet.discrete_b_score[:, {i}]",
            null_value=-9999,
            binning=(6, -0.5, 5.5),
            aux={"overflow": True},
            x_title=f"Jet{i+1} b-score",
        )
    config.add_variable(
        name="btag_weights",
        expression="btag_weight",
        binning=(200, 0, 4),
        aux={"overflow": True},
        x_title="b-tagging weight",
    )
    config.add_variable(
        name="low_lep2_pt",
        expression=lambda events: events.Lepton.pt[:, 1],
        binning=(20, 0, 20),
        aux={
            "overflow": False,
            "inputs": {"{Electron,Muon}.{pt,eta,phi,mass}"},
        },
        x_title=r"Subleading lepton $p_T$ (below 20 GeV)",
    )
    config.add_variable(
        name="discrete_b_scores",
        expression="Jet.discrete_b_score",
        binning=(6, -0.5, 5.5),
        aux={"overflow": True},
        x_title="Jet discrete b-scores",
    )
