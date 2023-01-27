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

    # Event properties
    config.add_variable(
        name="n_jet",
        binning=(12, -0.5, 11.5),
        x_title="Number of jets",
    )
    config.add_variable(
        name="n_deepjet",
        binning=(11, -0.5, 10.5),
        x_title="Number of deepjets",
    )
    config.add_variable(
        name="n_fatjet",
        binning=(7, -0.5, 6.5),
        x_title="Number of fatjets",
    )
    config.add_variable(
        name="n_electron",
        binning=(3, -0.5, 2.5),
        x_title="Number of electrons",
    )
    config.add_variable(
        name="n_muon",
        binning=(3, -0.5, 2.5),
        x_title="Number of muons",
    )
    config.add_variable(
        name="ht",
        binning=(40, 0, 1500),
        x_title="HT",
    )
    config.add_variable(
        name="ht_rebin",
        expression="ht",
        binning=[0, 80, 120, 160, 200, 240, 280, 320, 400, 500, 600, 800, 1200],
        unit="GeV",
        x_title="HT",
    )

    # Object properties

    # Jets (4 pt-leading jets)
    for i in range(4):
        config.add_variable(
            name=f"jet{i+1}_pt",
            expression=f"Jet.pt[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0., 400.),
            unit="GeV",
            x_title=r"Jet%i $p_{T}$" % (i + 1),
        )
        config.add_variable(
            name=f"jet{i+1}_eta",
            expression=f"Jet.eta[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(50, -2.5, 2.5),
            x_title=r"Jet%i $\eta$" % (i + 1),
        )
        config.add_variable(
            name=f"jet{i+1}_phi",
            expression=f"Jet.phi[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, -3.2, 3.2),
            x_title=r"Jet%i $\phi$" % (i + 1),
        )
        config.add_variable(
            name=f"jet{i+1}_mass",
            expression=f"Jet.mass[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 200),
            unit="GeV",
            x_title=r"Jet%i mass" % (i + 1),
        )
        config.add_variable(
            name=f"jet{i+1}_btagDeepB",
            expression=f"Jet.btagDeepB[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 1),
            x_title=r"Jet%i DeepCSV b+bb tag" % (i + 1),
        )
        config.add_variable(
            name=f"jet{i+1}_btagDeepFlavB",
            expression=f"Jet.btagDeepFlavB[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 1),
            x_title=r"Jet%i DeepFlavour b+bb+lepb tag" % (i + 1),
        )

    # FatJets (4 pt-leading jets)
    for i in range(4):
        config.add_variable(
            name=f"fatjet{i+1}_pt",
            expression=f"FatJet.pt[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 170., 500.),
            unit="GeV",
            x_title=r"FatJet%i $p_{T}$" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_eta",
            expression=f"FatJet.eta[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(50, -2.5, 2.5),
            x_title=r"FatJet%i $\eta$" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_phi",
            expression=f"FatJet.phi[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, -3.2, 3.2),
            x_title=r"FatJet%i $\phi$" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_mass",
            expression=f"FatJet.mass[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 250),
            unit="GeV",
            x_title=r"FatJet%i mass" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_msoftdrop",
            expression=f"FatJet.msoftdrop[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 250),
            unit="GeV",
            x_title=r"FatJet%i softdrop mass" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_tau1",
            expression=f"FatJet.tau1[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 1),
            x_title=r"FatJet%i $\tau_1$" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_tau2",
            expression=f"FatJet.tau2[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 1),
            x_title=r"FatJet%i $\tau_2$" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_tau21",
            expression=f"FatJet.tau21[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 1),
            x_title=r"FatJet%i $\tau_{21}$" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_btagDeepB",
            expression=f"FatJet.btagDeepB[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 1),
            x_title=r"FatJet%i DeepCSV b+bb tag" % (i + 1),
        )

    # Bjets (2 b-score leading jets)
    for i in range(2):
        config.add_variable(
            name=f"bjet{i+1}_pt",
            expression=f"Bjet.pt[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0., 300.),
            unit="GeV",
            x_title=r"Bjet %i $p_{T}$" % (i + 1),
        )
        config.add_variable(
            name=f"bjet{i+1}_eta",
            expression=f"Bjet.eta[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(50, -2.5, 2.5),
            x_title=r"Bjet %i $\eta$" % (i + 1),
        )
        config.add_variable(
            name=f"bjet{i+1}_phi",
            expression=f"Jet.phi[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, -3.2, 3.2),
            x_title=r"Bjet %i $\phi$" % (i + 1),
        )
        config.add_variable(
            name=f"bjet{i+1}_mass",
            expression=f"Bjet.mass[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 200),
            x_title=r"Bjet %i mass" % (i + 1),
        )

    # Leptons
    for obj in ["Electron", "Muon"]:
        config.add_variable(
            name=f"{obj.lower()}_pt",
            expression=f"{obj}.pt[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0., 350.),
            unit="GeV",
            x_title=obj + r" $p_{T}$",
        )
        config.add_variable(
            name=f"{obj.lower()}_phi",
            expression=f"{obj}.phi[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(40, -3.2, 3.2),
            x_title=obj + r" $\phi$",
        )
        config.add_variable(
            name=f"{obj.lower()}_eta",
            expression=f"{obj}.eta[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(50, -2.5, 2.5),
            x_title=obj + r" $\eta$",
        )
        config.add_variable(
            name=f"{obj.lower()}_mass",
            expression=f"{obj}.mass[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 200),
            x_title=obj + " mass",
        )

    # MET
    config.add_variable(
        name="met_pt",
        expression="MET.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"MET $p_{T}$",
    )
    config.add_variable(
        name="met_phi",
        expression="MET.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"MET $\phi$",
    )

    # bb features
    config.add_variable(
        name="m_bb",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"$m_{bb}$",
    )
    config.add_variable(
        name="deltaR_bb",
        binning=(40, 0, 5),
        x_title=r"$\Delta R(b,b)$",
    )
    # jj features
    config.add_variable(
        name="m_jj",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"$m_{jj}$",
    )
    config.add_variable(
        name="deltaR_jj",
        binning=(40, 0, 5),
        x_title=r"$\Delta R(j_{1},j_{2})$",
    )
