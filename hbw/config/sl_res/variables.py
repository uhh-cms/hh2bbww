# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")

from columnflow.columnar_util import EMPTY_FLOAT  # noqa
from hbw.util import call_once_on_config

# object of resonant HH analysis
#
#                           _    b
#                    Higgs_bb  /
#                  - - - - - -
#                /             \ _
#               /                b
#  Heavy_Higgs /                          l
#   - - - - - -                 Wlepton  /
#              \                ---------
#               \              /         \
#                \  Higgs_WW  /           nu
#                  - - - - - -
#                             \              q
#                              \  Whadron   /
#                               ------------
#                                           \
#                                            q'
#


@call_once_on_config()
def add_resonant_variables(config: od.Config) -> None:
    config.add_variable(  # Whadron
        name="pt_Whadron",
        expression="Whadron.pt",
        binning=(40, 0., 1500.),
        unit="GeV",
        x_title=r"$pt_{qq'}$",
    )
    config.add_variable(
        name="m_Whadron",
        binning=(40, 0., 1500.),
        unit="GeV",
        x_title=r"$m_{qq'}$",
    )
    config.add_variable(
        name="eta_Whadron",
        binning=(50, -2.5, 2.5),
        x_title=r"$eta_{qq'}$",
    )
    config.add_variable(
        name="phi_Whadron",
        binning=(40, -3.2, 3.2),
        x_title=r"$phi_{qq'}$",
    )
# Wlepton
    config.add_variable(
        name="pt_Wlepton",
        binning=(40, 0., 1500.),
        unit="GeV",
        x_title=r"$pt_{leptnu'}$",
    )
    config.add_variable(
        name="m_Wlepton",
        binning=(40, 0., 1500.),
        unit="GeV",
        x_title=r"$m_{leptnu}$",
    )
    config.add_variable(
        name="eta_Wlepton",
        binning=(50, -2.5, 2.5),
        x_title=r"$eta_{leptnu}$",
    )
    config.add_variable(
        name="phi_Wlepton",
        binning=(40, -3.2, 3.2),
        x_title=r"$phi_{leptnu}$",
    )
# Higgs_WW
    config.add_variable(
        name="pt_Higgs_WW",
        binning=(40, 0., 600.),
        unit="GeV",
        x_title=r"$pt_{Leptnu+qq'}$",
    )
    config.add_variable(
        name="m_Higgs_WW",
        binning=(40, 0., 1500.),
        unit="GeV",
        x_title=r"$m_{Leptnu+qq'}$",
    )
    config.add_variable(
        name="eta_Higgs_WW",
        binning=(50, -2.5, 2.5),
        x_title=r"$eta_{Leptnu+qq'}$",
    )
    config.add_variable(
        name="phi_Higgs_WW",
        binning=(40, -3.2, 3.2),
        x_title=r"$phi_{Leptnu+qq'}$",
    )
# Higgs_bb
    config.add_variable(
        name="pt_Higgs_bb",
        binning=(40, 0., 600.),
        unit="GeV",
        x_title=r"$pt_{bb}$",
    )
    config.add_variable(
        name="m_Higgs_bb",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"$m_{bb}$",
    )
    config.add_variable(
        name="eta_Higgs_bb",
        binning=(50, -2.5, 2.5),
        x_title=r"$eta_{bb}$",
    )
    config.add_variable(
        name="phi_Higgs_bb",
        binning=(40, -3.2, 3.2),
        x_title=r"$phi_{bb}$",
    )
# Heavy_Higgs
    config.add_variable(
        name="pt_Heavy_Higgs",
        binning=(40, 0., 600.),
        unit="GeV",
        x_title=r"$pt_{Leptnu+bb+qq'}$",
    )
    config.add_variable(
        name="m_Heavy_Higgs",
        binning=(40, 0., 1500.),
        unit="GeV",
        x_title=r"$m_{Leptnu+bb+qq'}$",
    )
    config.add_variable(
        name="eta_Heavy_Higgs",
        binning=(50, -2.5, 2.5),
        x_title=r"$eta_{Leptnu+bb+qq'}$",
    )
    config.add_variable(
        name="phi_Heavy_Higgs",
        binning=(40, -3.2, 3.2),
        x_title=r"$phi_{Leptnu+bb+qq'}$",
    )


@call_once_on_config()
def add_sl_res_ml_variables(config: od.Config) -> None:
    """
    Adds ML input variables to a *config*.
    """

    # reconstructed variables
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
    # W_jj features
    config.add_variable(
        name="mli_pt_jj",
        expression="mli_pt_jj",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"$W_{hadron}$ $p_{T}$",
    )
    config.add_variable(
        name="mli_eta_jj",
        expression="mli_eta_jj",
        binning=(50, -2.5, 2.5),
        x_title=r"$W_{hadron}$ $\eta$",
    )
    config.add_variable(
        name="mli_phi_jj",
        expression="mli_phi_jj",
        binning=(40, -3.2, 3.2),
        x_title=r"$W_{hadron}$ $\phi$ ",
    )
    # W_lnu features
    config.add_variable(
        name="mli_pt_lnu",
        expression="mli_pt_lnu",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"$W_{lepton}$ $p_{T}$",
    )
    config.add_variable(
        name="mli_eta_lnu",
        expression="mli_eta_lnu",
        binning=(50, -2.5, 2.5),
        x_title=r"$W_{lepton}$ $\eta$",
    )
    config.add_variable(
        name="mli_phi_lnu",
        expression="mli_phi_lnu",
        binning=(40, -3.2, 3.2),
        x_title=r"$W_{lepton}$ $\phi$ ",
    )
    # H_WW (all) features
    config.add_variable(
        name="mli_pt_jjlnu",
        expression="mli_pt_jjlnu",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"$H_{WW}$ $p_{T}$",
    )
    config.add_variable(
        name="mli_eta_jjlnu",
        expression="mli_eta_jjlnu",
        binning=(50, -2.5, 2.5),
        x_title=r"H_{WW}$ $\eta$",
    )
    config.add_variable(
        name="mli_phi_jjlnu",
        expression="mli_phi_jjlnu",
        binning=(40, -3.2, 3.2),
        x_title=r"$H_{WW}$ $\phi$ ",
    )
    # H_bb features
    config.add_variable(
        name="mli_eta_bb",
        expression="mli_eta_bb",
        binning=(50, -2.5, 2.5),
        x_title=r"H_{bb}$ $\eta$",
    )
    config.add_variable(
        name="mli_phi_bb",
        expression="mli_phi_bb",
        binning=(40, -3.2, 3.2),
        x_title=r"$H_{bb}$ $\phi$ ",
    )
