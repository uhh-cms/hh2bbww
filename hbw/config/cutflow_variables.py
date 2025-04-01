# coding: utf-8

"""
Definition of variables that can be plotted via `PlotCutflowVariables` tasks.
"""

import order as od

from hbw.util import call_once_on_config
from hbw.config.styling import quick_addvar


@call_once_on_config()
def add_cutflow_variables(config: od.Config) -> None:
    """
    Function to add cutflow variables to the config. Variable names of objects follow the format
    'cf_{obj}{i}_{var}' (all lowercase). See `hbw.config.styling` for more information on the configuration.
    """

    config.add_variable(
        name="cf_loosejets_pt",
        expression="cutflow.LooseJet.pt",
        binning=(40, 0, 400),
        unit="GeV",
        x_title="$p_{T}$ of all jets",
    )

    # Jets
    for i in range(4):
        # loose jets
        for var in ("pt", "puId", "puIdDisc", "bRegRes", "bRegCorr"):
            quick_addvar(config, "LooseJet", i, var)
            quick_addvar(config, "Jet", i, var)
            quick_addvar(config, "JetPtBelow50", i, var)

    # Leptons
    for i in range(3):
        # veto leptons
        for var in ("pt", "eta", "dxy", "dz", "mvaTTH", "miniPFRelIso_all"):
            quick_addvar(config, "VetoLepton", i, var)
        # veto electrons
        for var in ("mvaFall17V2Iso", "mvaFall17V2noIso", "pfRelIso03_all"):
            quick_addvar(config, "VetoElectron", i, var)
        # veto muons
        for var in ("pfRelIso04_all", "mvaLowPt"):
            quick_addvar(config, "VetoMuon", i, var)

    # number of objects
    for obj in (
            "deepjet_med", "fatjet", "hbbjet", "electron", "muon", "lepton",
            "veto_electron", "veto_muon", "veto_lepton", "veto_tau",
    ):
        config.add_variable(
            name=f"cf_n_{obj}",
            expression=f"cutflow.n_{obj}",
            binning=(11, -0.5, 10.5),
            x_title=f"Number of {obj}s",
        )

    # FatJet cutflow variables
    config.add_variable(
        name="cf_HbbJet1_n_subjets".lower(),
        expression="HbbJet.n_subjets[:, 0]",
        binning=(5, -0.5, 4.5),
        x_title="HbbJet 1 number of subjets",
    )
    config.add_variable(
        name="cf_HbbJet1_n_separated_jets".lower(),
        expression="HbbJet.n_separated_jets[:, 0]",
        binning=(5, -0.5, 4.5),
        x_title=r"HbbJet 1 number of jets with $\Delta R > 1.2$",
    )
    config.add_variable(
        name="cf_HbbJet1_max_dr_ak4".lower(),
        expression="HbbJet.max_dr_ak4[:, 0]",
        binning=(40, 0, 4),
        x_title=r"max($\Delta R$ (HbbJet 1, AK4 jets))",
    )

    # Pileup variables
    config.add_variable(
        name="cf_npvs",
        expression="cutflow.npvs",
        binning=(101, -.5, 100.5),
        x_title="Number of reconstructed primary vertices",
    )
    config.add_variable(
        name="cf_npu",
        expression="cutflow.npu",
        binning=(101, -.5, 100.5),
        x_title="Number of pileup interactions",
    )
    config.add_variable(
        name="cf_npu_true",
        expression="cutflow.npu_true",
        binning=(101, -.5, 100.5),
        x_title="True mean number of poisson distribution from which number of interactions has been sampled",
    )


@call_once_on_config()
def add_gen_variables(config: od.Config) -> None:
    """
    defines gen-level cutflow variables
    """
    for gp in ["h1", "h2", "b1", "b2", "wlep", "whad", "l", "nu", "q1", "q2", "sec1", "sec2"]:
        config.add_variable(
            name=f"gen_{gp}_pt",
            expression=f"gen_hbw_decay.{gp}.pt",
            binning=(40, 0., 1000.),
            unit="GeV",
            x_title=r"$p_{T, %s}^{gen}$" % (gp),
            aux={"overflow": True,},
        )
        config.add_variable(
            name=f"gen_{gp}_mass",
            expression=f"gen_hbw_decay.{gp}.mass",
            binning=(40, 0., 1000.),
            unit="GeV",
            x_title=r"$m_{%s}^{gen}$" % (gp),
            aux={"overflow": True,},
        )
        config.add_variable(
            name=f"gen_{gp}_eta",
            expression=f"gen_hbw_decay.{gp}.eta",
            binning=(40, -6., 6.),
            unit="GeV",
            x_title=r"$\eta_{%s}^{gen}$" % (gp),
            aux={"overflow": True,},
        )
        config.add_variable(
            name=f"gen_{gp}_eta_barrel",
            expression=f"gen_hbw_decay.{gp}.eta",
            binning=(40, -6., 6.),
            selection=(lambda events: events.gen_hbw_decay[gp]['eta'] > 2.4),
            unit="GeV",
            aux={"overflow": True,},
            x_title=r"$\eta_{%s}^{gen}(barrel)$" % (gp),
        )
        config.add_variable(
            name=f"gen_{gp}_phi",
            expression=f"gen_hbw_decay.{gp}.phi",
            binning=(40, -4, 4),
            unit="GeV",
            aux={"overflow": True,},
            x_title=r"$\phi_{%s}^{gen}$" % (gp),
        )
    # config.add_variable(
    #     name=f"vbfpair.nevents",
    #     #expression=f"gen_hbw_decay.{gp}.phi",
    #     binning=(2, -0.5, 1.5),
    #     unit="GeV",
    #     x_title=r"$Nevents$" ,
    # )
    config.add_variable(
        name=f"vbfpair.dr",
        #expression=f"gen_hbw_decay.{gp}.phi",
        binning=(40, 0, 10),
        unit="GeV",
        x_title=r"$\Delta \, R_{gen}$" ,
        aux={"overflow": True,},
    )
    config.add_variable(
        name=f"vbfpair.deta",
        #expression=f"gen_hbw_decay.{gp}.phi",
        binning=(40, 0, 10),
        unit="GeV",
        x_title=r"$\Delta \, \eta_{gen}$",
        aux={"overflow": True,},
    )
    config.add_variable(
        name=f"vbfpair.mass",
        #expression=f"gen_hbw_decay.{gp}.phi",
        binning=(40, 0, 4000),
        unit="GeV",
        x_title=r"$mass(VBF pair)^{gen}$",
        aux={"overflow": True,},
    )

@call_once_on_config()
def add_gp_variables(config: od.Config) -> None:
    """
    defines gen-level cutflow variables
    """
    for gp in ["h1", "h2", "b1", "b2", "wlep", "whad", "l", "nu", "q1", "q2", "sec1", "sec2"]:
        config.add_variable(
            name=f"gp_{gp}_pt",
            expression=f"gp.{gp}_pt",
            binning=(40, 0., 400.),
            unit="GeV",
            x_title=r"$p_{T, %s}^{gen}$" % (gp),
        )
        config.add_variable(
            name=f"gp_{gp}_mass",
            expression=f"gp.{gp}_mass",
            binning=(40, 0., 400.),
            unit="GeV",
            x_title=r"$m_{%s}^{gen}$" % (gp),
        )
        config.add_variable(
            name=f"gp_{gp}_eta",
            expression=f"gp.{gp}_eta",
            binning=(12, -6., 6.),
            unit="GeV",
            x_title=r"$\eta_{%s}^{gen}$" % (gp),
        )
        config.add_variable(
            name=f"gp_{gp}_phi",
            expression=f"gp.{gp}_phi",
            binning=(8, -4, 4),
            unit="GeV",
            x_title=r"$\phi_{%s}^{gen}$" % (gp),
        )
