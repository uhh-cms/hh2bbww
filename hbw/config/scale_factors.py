# coding: utf-8

"""
Contains changes to the config needed for the calculation of trigger efficiencies and scale factors.
"""


import order as od

from columnflow.columnar_util import EMPTY_FLOAT

from hbw.util import call_once_on_config


@call_once_on_config
def configure_for_scale_factors(cfg: od.Config) -> None:
    """
    TODO: Document changes this function does to the config.

    NOTE (Mathis): this function should not be required anymore for DL SFs,
    I moved these config changes to the Reducer "triggersf" in hbw/reduction/default.py.
    """

    ####################################################################################################################
    # Changes for single lepton channel
    ####################################################################################################################
    if cfg.has_tag("is_sl") and cfg.has_tag("is_nonresonant"):
        # Set list of trigger paths to explore, this is only needed for triggers not fully added in the trigger config.
        # Single lepton triggers for non-resonant searches
        cfg.x.sl_triggers = {}
        cfg.x.sl_triggers["2022"] = {
            "e": [
                "Ele30_WPTight_Gsf",
                "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
                "Ele28_eta2p1_WPTight_Gsf_HT150",
                "Ele15_IsoVVVL_PFHT450",
                "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
            ],
            "m": [
                "IsoMu24",
                "Mu50",
                "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
                "Mu15_IsoVVVL_PFHT450",
            ],
        }
        cfg.x.sl_triggers["2023"] = {
            "e": [
                "Ele30_WPTight_Gsf",
                "PFHT280_QuadPFJet30_PNet2BTagMean0p55",
                "Ele28_eta2p1_WPTight_Gsf_HT150",
                "Ele15_IsoVVVL_PFHT450",
                "Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
                "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70",
            ],
            "m": [
                "IsoMu24",
                "Mu50",
                "PFHT280_QuadPFJet30_PNet2BTagMean0p55",
                "Mu15_IsoVVVL_PFHT450",
                "PFHT340_QuadPFJet70_50_40_40_PNet2BTagMean0p70",
                "Mu3er1p5_PFJet100er2p5_PFMETNoMu100_PFMHTNoMu100_IDTight",
            ],
        }
        # cfg.x.sl_orthogonal_triggers = {}

        # Add paths to keep after ReduceEvents
        for channel in cfg.x.sl_triggers[cfg.x.cpn_tag[:4]]:
            for path in cfg.x.sl_triggers[cfg.x.cpn_tag[:4]][channel]:
                if f"HLT.{path}" not in cfg.x.keep_columns["cf.ReduceEvents"]:
                    cfg.x.keep_columns["cf.ReduceEvents"] |= {f"HLT.{path}"}

    ####################################################################################################################
    # Changes for dilepton channel
    ####################################################################################################################
    if cfg.has_tag("is_dl") and cfg.has_tag("is_nonresonant"):
        # Double lepton triggers for non-resonant searches, only the orthgonal trigger needs to be set here
        # TODO: At some point this should probably time dependent as well
        cfg.x.dl_orthogonal_trigger = "PFMETNoMu120_PFMHTNoMu120_IDTight"
        cfg.x.dl_orthogonal_trigger2 = "PFMET120_PFMHT120_IDTight"
        cfg.x.hlt_L1_seeds = {
            "PFMETNoMu120_PFMHTNoMu120_IDTight": [
                "ETMHF90",
                "ETMHF100",
                "ETMHF110",
                "ETMHF120",
                "ETMHF130",
                "ETMHF140",
                "ETMHF150",
                "ETM150",
                "ETMHF90_SingleJet60er2p5_dPhi_Min2p1",
                "ETMHF90_SingleJet60er2p5_dPhi_Min2p6",
            ],
            "PFMET120_PFMHT120_IDTight": [
                "ETMHF90",
                "ETMHF100",
                "ETMHF110",
                "ETMHF120",
                "ETMHF130",
                "ETMHF140",
                "ETMHF150",
                "ETM150",
                "ETMHF90_SingleJet60er2p5_dPhi_Min2p1",
                "ETMHF90_SingleJet60er2p5_dPhi_Min2p6",
            ],
        }
        # Add paths to keep after ReduceEvents
        cfg.x.keep_columns["cf.ReduceEvents"] |= {f"HLT.{cfg.x.dl_orthogonal_trigger}"}
        cfg.x.keep_columns["cf.ReduceEvents"] |= {f"HLT.{cfg.x.dl_orthogonal_trigger2}"}
        for seed in cfg.x.hlt_L1_seeds[cfg.x.dl_orthogonal_trigger]:
            if f"L1.{seed}" not in cfg.x.keep_columns["cf.ReduceEvents"]:
                cfg.x.keep_columns["cf.ReduceEvents"] |= {f"L1.{seed}"}

        # Add additional datasets needed for orthogonal efficiency measurements.
        # data_jethtmet_eras = {
        #     "2022preEE": "cd",
        #     "2022postEE": "efg",
        #     "2023preBPix": "",
        #     "2023postBPix": "",  # TODO: add 2023 jetmet datasets in cmsdb
        # }[cfg.x.cpn_tag]

        # data_jethtmet_datasets = [
        #     f"data_jethtmet_{era}"
        #     for era in data_jethtmet_eras
        # ]

        # cfg.x.dataset_names.update({"data_jethtmet": data_jethtmet_datasets})
        # cfg.add_process(cfg.x.procs.n.data_jethtmet)
        # for dataset in data_jethtmet_datasets:
        #     cfg.add_dataset(cfg.campaign.get_dataset(dataset))
        #     if cfg.x.cpn_tag == "2022preEE":
        #         cfg.datasets.get(dataset).x.jec_era = "RunCD"

        #
        # Collection of smaller changes
        #
        # use histproducer without trigger scale factors
        cfg.x.default_hist_producer = "default"

        # add combined process with ttbar and drell-yan
        cfg.add_process(cfg.x.procs.n.tt_dy)

    ####################################################################################################################
    # Changes for both channels
    ####################################################################################################################
    # Change variables
    # need npvs as floats in the scale factor calculation
    cfg.add_variable(
        name="trg_npvs",
        expression=lambda events: events.PV.npvs * 1.0,
        aux={
            "inputs": {"PV.npvs"},
        },
        binning=(81, 0, 81),
        x_title=r"$\text{N}_{\text{PV}}$",
        # discrete_x=True,
    )
    # change lepton pt binning
    cfg.add_variable(
        name="trg_lepton0_pt",
        expression=lambda events: events.Lepton[:, 0].pt,
        aux=dict(
            inputs={"{Electron,Muon}.{pt,eta,phi,mass}"},
        ),
        binning=(400, 0., 400.),
        unit="GeV",
        null_value=EMPTY_FLOAT,
        x_title=r"Leading lepton $p_{{T}}$",
    )
    cfg.add_variable(
        name="trg_lepton1_pt",
        expression=lambda events: events.Lepton[:, 1].pt,
        aux=dict(
            inputs={"{Electron,Muon}.{pt,eta,phi,mass}"},
        ),
        binning=(400, 0., 400.),
        unit="GeV",
        null_value=EMPTY_FLOAT,
        x_title=r"Subleading lepton $p_{{T}}$",
    )
    cfg.add_variable(
        name="sf_lepton0_pt",
        expression=lambda events: events.Lepton[:, 0].pt,
        aux=dict(
            inputs={"{Electron,Muon}.{pt,eta,phi,mass}"},
        ),
        binning=[0., 15.] + [i for i in range(16, 76)] + [80., 90., 100., 110., 120., 150., 175., 200., 240., 400.],
        unit="GeV",
        null_value=EMPTY_FLOAT,
        x_title=r"Leading lepton $p_{{T}}$",
    )
    cfg.add_variable(
        name="sf_lepton1_pt",
        expression=lambda events: events.Lepton[:, 1].pt,
        aux=dict(
            inputs={"{Electron,Muon}.{pt,eta,phi,mass}"},
        ),
        binning=[0., 15.] + [i for i in range(16, 66)] + [100., 110., 120., 150., 175., 200., 240., 400.],
        unit="GeV",
        null_value=EMPTY_FLOAT,
        x_title=r"Subleading lepton $p_{{T}}$",
    )
    cfg.add_variable(
        name="sf_npvs",
        expression=lambda events: events.PV.npvs * 1.0,
        aux={
            "inputs": {"PV.npvs"},
        },
        binning=[0., 30.] + [i for i in range(31, 41)] + [50., 81.],
        x_title=r"$\text{N}_{\text{PV}}$",
    )
    # add trigger ids as variables
    cfg.add_variable(
        name="trigger_ids",  # these are the trigger IDs saved during the selection
        aux={"axis_type": "intcat"},
        x_title="Trigger IDs",
    )
    # trigger ids für scale factors
    cfg.add_variable(
        name="trig_ids",  # these are produced in the trigger_prod producer when building different combinations
        aux={"axis_type": "strcat"},
        x_title="Trigger IDs for scale factors",
    )
