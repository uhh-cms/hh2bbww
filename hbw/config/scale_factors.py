# coding: utf-8

"""
Contains changes to the config needed for the calculation of trigger efficiencies and scale factors.
"""


import order as od


from hbw.util import call_once_on_config


@call_once_on_config
def configure_for_scale_factors(cfg: od.Config) -> None:
    """
    TODO: Document changes this function does to the config.
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
        }
        # Add paths to keep after ReduceEvents
        cfg.x.keep_columns["cf.ReduceEvents"] |= {f"HLT.{cfg.x.dl_orthogonal_trigger}"}
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

        # Change variables
        # need npvs as floats in the scale factor calculation
        cfg.variables.remove("npvs")
        cfg.add_variable(
            name="npvs",
            expression=lambda events: events.PV.npvs * 1.0,
            aux={
                "inputs": {"PV.npvs"},
            },
            binning=(81, 0, 81),
            x_title="Number of primary vertices",
            discrete_x=True,
        )
        # change lepton pt binning
        for i in [0, 1]:
            cfg.variables.get(f"lepton{i}_pt").binning = (240, 0., 240.)
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

    ####################################################################################################################
    # Changes for both channels
    ####################################################################################################################
    # change process scale for signal processes
    for proc, _, _ in cfg.walk_processes():
        # unstack signal in plotting
        if "hh_" in proc.name.lower():
            proc.add_tag("is_signal")
            proc.unstack = True
            proc.scale = None
