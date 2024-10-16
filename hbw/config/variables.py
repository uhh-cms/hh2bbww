# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")

from columnflow.columnar_util import EMPTY_FLOAT
from hbw.util import call_once_on_config
from hbw.config.styling import default_var_binning, default_var_unit


@call_once_on_config()
def add_feature_variables(config: od.Config) -> None:
    """
    Adds variables to a *config* that are produced as part of the `features` producer.
    """

    # Event properties
    config.add_variable(
        name="features_n_jet",
        expression=lambda events: ak.num(events.Jet.pt, axis=1),
        binning=(12, -0.5, 11.5),
        x_title="Number of jets",
        aux={"inputs": {"Jet.pt"}},
        discrete_x=True,
    )
    config.add_variable(
        name="features_n_deepjet",
        binning=(11, -0.5, 10.5),
        x_title="Number of deepjets",
        discrete_x=True,
    )
    config.add_variable(
        name="features_n_fatjet",
        binning=(7, -0.5, 6.5),
        x_title="Number of fatjets",
        discrete_x=True,
    )
    config.add_variable(
        name="features_n_hbbjet",
        binning=(4, -0.5, 3.5),
        x_title="Number of hbbjets",
        discrete_x=True,
    )
    config.add_variable(
        name="features_n_electron",
        binning=(4, -0.5, 3.5),
        x_title="Number of electrons",
        discrete_x=True,
    )
    config.add_variable(
        name="features_n_muon",
        binning=(4, -0.5, 3.5),
        x_title="Number of muons",
        discrete_x=True,
    )
    config.add_variable(
        name="features_n_bjet",
        binning=(4, -0.5, 3.5),
        x_title="Number of bjets",
        discrete_x=True,
    )
    config.add_variable(
        name="features_ht",
        binning=(40, 0, 1500),
        x_title="HT",
    )

    # bb features
    config.add_variable(
        name="m_bb",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"$m_{bb}$",
    )
    config.add_variable(
        name="m_bb_combined",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"$m_{bb}$ combined",
    )
    config.add_variable(
        name="bb_pt",
        binning=(40, 0., 350),
        x_title=r"$p_T^{bb}$",
        unit="GeV",
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
        name="jj_pt",
        binning=(40, 0., 350),
        x_title=r"$p_T^{jj}$",
        unit="GeV",
    )
    config.add_variable(
        name="deltaR_jj",
        binning=(40, 0, 5),
        x_title=r"$\Delta R(j_{1},j_{2})$",
    )

    # FatJet features
    for i in range(2):
        config.add_variable(
            name=f"fatjet{i+1}_tau21",
            expression=f"FatJet.tau21[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 1),
            x_title=r"FatJet %i $\tau_{21}$" % (i + 1),
        )


@call_once_on_config()
def add_neutrino_variables(config: od.Config) -> None:
    """
    Adds variables to a *config* that are produced as part of the `neutrino_reconstruction` producer.
    """

    for obj in ["Neutrino", "Neutrino1", "Neutrino2"]:
        # pt and phi should be the same as MET, mass should always be 0
        for var in ["pt", "eta", "phi", "mass"]:
            config.add_variable(
                name=f"{obj}_{var}",
                expression=f"{obj}.{var}",
                binning=default_var_binning[var],
                unit=default_var_unit.get(var, "1"),
                x_title="{obj} {var}".format(obj=obj, var=var),
            )


@call_once_on_config()
def add_top_reco_variables(config: od.Config) -> None:
    """
    Adds variables to a *config* that are produced as part of the `top_reconstruction` producer.
    """
    # add neutrino variables aswell since the neutrino needs to be reconstructed anyway
    add_neutrino_variables(config)

    # add reconstructed top variables
    for obj in ["tlep_hyp1", "tlep_hyp2"]:
        # pt and phi should be the same as MET, mass should always be 0
        for var in ["pt", "eta", "phi", "mass"]:
            config.add_variable(
                name=f"{obj}_{var}",
                expression=f"{obj}.{var}",
                binning=default_var_binning[var],
                unit=default_var_unit.get(var, "1"),
                x_title="{obj} {var}".format(obj=obj, var=var),
            )


def add_debug_variable(config: od.Config) -> None:
    """
    Variable for debugging, opens a debugger when the `expression` is evaluated, e.g. when calling:
    law run cf.CreateHistograms --variables debugger
    """
    def dbg(events):
        from hbw.util import debugger
        debugger()

    config.add_variable(
        name="debugger",
        expression=dbg,
        aux={"inputs": {"{Electron,Muon,Jet}.{pt,eta,phi,mass}"}},
        binning=(1, 0, 1),
    )


@call_once_on_config()
def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config* that are present after `ReduceEvents`
    without calling any producer
    """
    # from columnflow.columnar_util import set_ak_column

    # def with_behavior(custom_expression: callable) -> callable:
    #     def expression(events):
    #         from hbw.production.prepare_objects import custom_collections
    #         from columnflow.production.util import attach_coffea_behavior
    #         events = attach_coffea_behavior.call_func(None, events, collections=custom_collections)

    #         # add Lepton collection if possible
    #         if "Lepton" not in events.fields and "Electron" in events.fields and "Muon" in events.fields:
    #             lepton = ak.concatenate([events.Muon * 1, events.Electron * 1], axis=-1)
    #             events = set_ak_column(events, "Lepton", lepton[ak.argsort(lepton.pt, ascending=False)])

    #         return custom_expression(events)

    #     return expression

    # config.add_variable(
    #     name="mll_test",
    #     expression=lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
    #     binning=(40, 0., 200.),
    #     unit="GeV",
    #     x_title=r"$m_{ll}$",
    #     aux={"inputs": {"{Electron,Muon}.{pt,eta,phi,mass}"}},
    # )

    # (the "event", "run" and "lumi" variables are required for some cutflow plotting task,
    # and also correspond to the minimal set of columns that coffea's nano scheme requires)
    config.add_variable(
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    config.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    config.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )

    #
    # Weights
    #

    # TODO: implement tags in columnflow; meanwhile leave these variables commented out (as they only work for mc)
    """
    config.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=list(np.logspace(1.2, 5, 200)),
        log_x=True,
        x_title="MC weight",
        tags={"mc_only"},
    )
    config.add_variable(
        name="normalization_weight",
        binning=list(np.logspace(0, 6, 100)),
        log_x=True,
        x_title="normalization weight",
        tags={"mc_only"},
    )
    config.add_variable(
        name="event_weight",
        binning=list(np.logspace(0, 6, 100)),
        log_x=True,
        x_title="event weight",
        tags={"mc_only"},
    )

    for weight in ["pu", "pdf", "mur", "muf", "murf_envelope"]:
        # NOTE: would be nice to not use the event weight for these variables
        config.add_variable(
            name=f"{weight}_weight",
            expression=f"{weight}_weight",
            binning=(40, -2, 2),
            x_title=f"{weight} weight",
            tags={"mc_only"},
        )
        config.add_variable(
            name=f"{weight}_weight_log",
            expression=f"{weight}_weight",
            binning=list(np.logspace(-2, 2, 100)),
            log_x=True,
            x_title=f"{weight} weight",
            tags={"mc_only"},
        )
    """
    config.add_variable(
        name="npvs",
        expression="PV.npvs",
        binning=(51, -.5, 50.5),
        x_title="Number of primary vertices",
        discrete_x=True,
    )

    # some variables for testing of different axis types
    config.add_variable(
        name="high_jet_pt_strcat",
        # NOTE: for some reason passing the string directly produces ValueError due to different shapes, e.g.
        # ValueError: cannot broadcast RegularArray of size 7 with RegularArray of size 264
        expression=lambda events: ak.where(events.Jet.pt > 50, ["high_pt"], ["low_pt"]),
        aux={
            "inputs": {"Jet.pt"},
            "axis_type": "strcat",
        },
        x_title="Jet $p_{T}$ string category",
    )
    # NOTE: for IntCat, it is important to pick the correct bins via *hist.loc* because the order of bins can be random
    # h[{"high_jet_pt_intcat": 0}] picks the first bin, independent of which value the bin edge corresponds to
    # h[{"high_jet_pt_intcat": hist.loc(0)}] picks the bin with value 0
    config.add_variable(
        name="high_jet_pt_intcat",
        expression=lambda events: ak.where(events.Jet.pt > 50, 1, 0),
        aux={
            "inputs": {"Jet.pt"},
            "axis_type": "intcat",
        },
        x_title="Jet $p_{T}$ integer category",
    )
    config.add_variable(
        name="high_jet_pt_bool",
        expression=lambda events: events.Jet.pt > 50,
        aux={
            "inputs": {"Jet.pt"},
            "axis_type": "bool",
        },
        x_title="Jet $p_{T}$ bool category",
    )

    #
    # Simple event properties
    #

    config.add_variable(
        name="mll",
        binning=(40, 0., 200.),
        unit="GeV",
        x_title=r"$m_{ll}$",
    )
    config.add_variable(
        name="mll_manybins",
        expression="mll",
        binning=(2400, 0., 240.),
        unit="GeV",
        x_title=r"$m_{ll}$",
        aux={
            "rebin": 10,
            "x_max": 50,
        },
    )

    config.add_variable(
        # NOTE: only works when running `prepare_objects` in WeightProducer
        name="ptll",
        expression=lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"$p_{T}^{ll}$",
        aux={"inputs": {"{Electron,Muon}.{pt,eta,phi,mass}"}},
    )

    config.add_variable(
        name="n_jet",
        expression=lambda events: ak.num(events.Jet.pt, axis=1),
        aux={"inputs": {"Jet.pt"}},
        binning=(12, -0.5, 11.5),
        x_title="Number of jets",
        discrete_x=True,
    )
    deepjet_wps = config.x.btag_working_points.deepjet
    config.add_variable(
        name="n_deepjet_loose",
        expression=lambda events: ak.sum(events.Jet.btagDeepFlavB > deepjet_wps.loose, axis=1),
        aux={"inputs": {"Jet.pt", "Jet.btagDeepFlavB"}},
        binning=(7, -0.5, 6.5),
        x_title="Number of deepjets (loose WP)",
        discrete_x=True,
    )
    config.add_variable(
        name="n_deepjet_medium",
        expression=lambda events: ak.sum(events.Jet.btagDeepFlavB > deepjet_wps.medium, axis=1),
        aux={"inputs": {"Jet.pt", "Jet.btagDeepFlavB"}},
        binning=(7, -0.5, 6.5),
        x_title="Number of deepjets (medium WP)",
        discrete_x=True,
    )
    config.add_variable(
        name="n_deepjet_tight",
        expression=lambda events: ak.sum(events.Jet.btagDeepFlavB > deepjet_wps.tight, axis=1),
        aux={"inputs": {"Jet.pt", "Jet.btagDeepFlavB"}},
        binning=(7, -0.5, 6.5),
        x_title="Number of deepjets (tight WP)",
        discrete_x=True,
    )
    if config.x.run == 3:
        particlenet_wps = config.x.btag_working_points.particlenet
        config.add_variable(
            name="n_particlenet_loose",
            expression=lambda events: ak.sum(events.Jet.btagPNetB > particlenet_wps.loose, axis=1),
            aux={"inputs": {"Jet.pt", "Jet.btagPNetB"}},
            binning=(7, -0.5, 6.5),
            x_title="Number of pnet jets (loose WP)",
            discrete_x=True,
        )
        config.add_variable(
            name="n_particlenet_medium",
            expression=lambda events: ak.sum(events.Jet.btagPNetB > particlenet_wps.medium, axis=1),
            aux={"inputs": {"Jet.pt", "Jet.btagPNetB"}},
            binning=(7, -0.5, 6.5),
            x_title="Number of pnet jets (medium WP)",
            discrete_x=True,
        )
        config.add_variable(
            name="n_particlenet_tight",
            expression=lambda events: ak.sum(events.Jet.btagPNetB > particlenet_wps.tight, axis=1),
            aux={"inputs": {"Jet.pt", "Jet.btagPNetB"}},
            binning=(7, -0.5, 6.5),
            x_title="Number of pnet jets (tight WP)",
            discrete_x=True,
        )
    config.add_variable(
        name="n_fatjet",
        expression=lambda events: ak.num(events.FatJet.pt, axis=1),
        aux={"inputs": {"FatJet.pt"}},
        binning=(7, -0.5, 6.5),
        x_title="Number of fatjets",
        discrete_x=True,
    )
    config.add_variable(
        name="n_hbbjet",
        expression=lambda events: ak.num(events.HbbJet.pt, axis=1),
        aux={"inputs": {"HbbJet.pt"}},
        binning=(4, -0.5, 3.5),
        x_title="Number of hbbjets",
        discrete_x=True,
    )
    config.add_variable(
        name="n_electron",
        expression=lambda events: ak.num(events.Electron.pt, axis=1),
        aux={"inputs": {"Electron.pt"}},
        binning=(4, -0.5, 3.5),
        x_title="Number of electrons",
        discrete_x=True,
    )
    config.add_variable(
        name="n_muon",
        expression=lambda events: ak.num(events.Muon.pt, axis=1),
        aux={"inputs": {"Muon.pt"}},
        binning=(4, -0.5, 3.5),
        x_title="Number of muons",
        discrete_x=True,
    )
    config.add_variable(
        name="n_bjet",
        expression=lambda events: ak.num(events.Bjet.pt, axis=1),
        aux={"inputs": {"Bjet.pt"}},
        binning=(4, -0.5, 3.5),
        x_title="Number of bjets",
        discrete_x=True,
    )
    config.add_variable(
        name="ht",
        expression=lambda events: ak.sum(events.Jet.pt, axis=1),
        aux={"inputs": {"Jet.pt"}},
        binning=(40, 0, 1200),
        unit="GeV",
        x_title="HT",
    )
    config.add_variable(
        name="lt",
        expression=lambda events: (
            ak.sum(events.Muon.pt, axis=1) + ak.sum(events.Muon.pt, axis=1) + events.MET.pt
        ),
        aux={"inputs": {"Muon.pt", "Electron.pt", "MET.pt"}},
        binning=(40, 0, 1200),
        unit="GeV",
        x_title="LT",
    )
    config.add_variable(
        name="ht_bjet_norm",
        expression=lambda events: ak.sum(events.Jet.pt, axis=1),
        aux={"inputs": {"Jet.pt"}},
        binning=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1450, 1700, 2400],
        unit="GeV",
        x_title="HT",
    )

    #
    # Object properties
    #

    config.add_variable(
        name="jets_pt",
        expression="Jet.pt",
        binning=(40, 0, 400),
        unit="GeV",
        x_title="$p_{T}$ of all jets",
    )

    # Jets (4 pt-leading jets)
    for i in range(4):
        config.add_variable(
            name=f"jet{i+1}_pt",
            expression=f"Jet.pt[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0., 400.),
            unit="GeV",
            x_title=r"Jet %i $p_{T}$" % (i + 1),
        )
        config.add_variable(
            name=f"jet{i+1}_eta",
            expression=f"Jet.eta[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(50, -2.5, 2.5),
            x_title=r"Jet %i $\eta$" % (i + 1),
        )
        config.add_variable(
            name=f"jet{i+1}_phi",
            expression=f"Jet.phi[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, -3.2, 3.2),
            x_title=r"Jet %i $\phi$" % (i + 1),
        )
        config.add_variable(
            name=f"jet{i+1}_mass",
            expression=f"Jet.mass[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 200),
            unit="GeV",
            x_title=r"Jet %i mass" % (i + 1),
        )
        # config.add_variable(
        #     name=f"jet{i+1}_btagDeepB",
        #     expression=f"Jet.btagDeepB[:,{i}]",
        #     null_value=EMPTY_FLOAT,
        #     binning=(40, 0, 1),
        #     x_title=r"Jet %i DeepCSV b+bb tag" % (i + 1),
        # )
        config.add_variable(
            name=f"jet{i+1}_btagDeepFlavB",
            expression=f"Jet.btagDeepFlavB[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 1),
            x_title=r"Jet %i DeepFlavour b+bb+lepb tag" % (i + 1),
        )
        if config.x.run == 3:
            config.add_variable(
                name=f"jet{i+1}_btagPNetB",
                expression=f"Jet.btagPNetB[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(40, 0, 1),
                x_title=r"Jet %i ParticleNet score" % (i + 1),
            )

    # Bjets (2 b-score leading jets) and Lightjets (2 non-b pt-leading jets)
    for i in range(2):
        for obj in ["Bjet", "Lightjet"]:
            config.add_variable(
                name=f"{obj}{i+1}_pt".lower(),
                expression=f"{obj}.pt[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(40, 0., 300.),
                unit="GeV",
                x_title=obj + r" %i $p_{T}$" % (i + 1),
            )
            config.add_variable(
                name=f"{obj}{i+1}_eta".lower(),
                expression=f"{obj}.eta[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(50, -2.5, 2.5),
                x_title=obj + r" %i $\eta$" % (i + 1),
            )
            config.add_variable(
                name=f"{obj}{i+1}_phi".lower(),
                expression=f"{obj}.phi[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(40, -3.2, 3.2),
                x_title=obj + r" %i $\phi$" % (i + 1),
            )
            config.add_variable(
                name=f"{obj}{i+1}_mass".lower(),
                expression=f"{obj}.mass[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(40, 0, 200),
                x_title=obj + r" %i mass" % (i + 1),
            )
            if config.x.run == 3:
                config.add_variable(
                    name=f"{obj}{i+1}_btagPNetB",
                    expression=f"{obj}.btagPNetB[:,{i}]",
                    null_value=EMPTY_FLOAT,
                    binning=(40, 0, 1),
                    x_title=obj + r" %i ParticleNet score" % (i + 1),
                )

    # FatJets (2 pt-leading fatjets)
    for i in range(2):
        config.add_variable(
            name=f"fatjet{i+1}_pt",
            expression=f"FatJet.pt[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 170., 500.),
            unit="GeV",
            x_title=r"FatJet %i $p_{T}$" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_eta",
            expression=f"FatJet.eta[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(50, -2.5, 2.5),
            x_title=r"FatJet %i $\eta$" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_phi",
            expression=f"FatJet.phi[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, -3.2, 3.2),
            x_title=r"FatJet %i $\phi$" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_mass",
            expression=f"FatJet.mass[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 250),
            unit="GeV",
            x_title=r"FatJet %i mass" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_msoftdrop",
            expression=f"FatJet.msoftdrop[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 250),
            unit="GeV",
            x_title=r"FatJet %i softdrop mass" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_tau1",
            expression=f"FatJet.tau1[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 1),
            x_title=r"FatJet %i $\tau_1$" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_tau2",
            expression=f"FatJet.tau2[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 1),
            x_title=r"FatJet %i $\tau_2$" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_btagHbb",
            expression=f"FatJet.btagHbb[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 1),
            x_title=r"FatJet %i btagHbb" % (i + 1),
        )
        config.add_variable(
            name=f"fatjet{i+1}_deepTagMD_HbbvsQCD",
            expression=f"FatJet.deepTagMD_HbbvsQCD[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0, 1),
            x_title=r"FatJet %i deepTagMD_HbbvsQCD " % (i + 1),
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
        config.add_variable(
            name=f"{obj.lower()}_dxy",
            expression=f"{obj}.dxy[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(240, 0, 1.2),
            aux={"x_max": 0.2},
            x_title=obj + " dxy",
        )
        config.add_variable(
            name=f"{obj.lower()}_dz",
            expression=f"{obj}.dz[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(240, 0, 1.2),
            aux={"x_max": 0.6},
            x_title=obj + " dz",
        )

    # MET
    config.add_variable(
        name="met_pt",
        expression="MET.pt",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"MET $p_{T}$",
    )
    config.add_variable(
        name="met_phi",
        expression="MET.phi",
        binning=(40, -3.2, 3.2),
        x_title=r"MET $\phi$",
    )
