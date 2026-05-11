# coding: utf-8

"""
Configuration of corrections for the m(ttbar) analysis.
"""

import order as od

from columnflow.util import DotDict
from columnflow.production.cms.btag import BTagSFConfig


def btag_sf_cfg(
    config: od.Config,
    year: int = None,
) -> list[tuple, list]:
    name = ("deepJet_shape") if year != 2024 else ("UParTAK4_kinfit")
    discr = "btagPNetB" if year != 2024 else "btagUParTAK4B"
    jec_sources = [
        "",  # same as "Total"
        "Absolute",
        "AbsoluteMPFBias",
        "AbsoluteScale",
        "AbsoluteStat",
        f"Absolute_{year}",
        "BBEC1",
        f"BBEC1_{year}",
        "EC2",
        f"EC2_{year}",
        "FlavorQCD",
        "Fragmentation",
        "HF",
        f"HF_{year}",
        "PileUpDataMC",
        "PileUpPtBB",
        "PileUpPtEC1",
        "PileUpPtEC2",
        "PileUpPtHF",
        "PileUpPtRef",
        "RelativeBal",
        "RelativeFSR",
        "RelativeJEREC1",
        "RelativeJEREC2",
        "RelativeJERHF",
        "RelativePtBB",
        "RelativePtEC1",
        "RelativePtEC2",
        "RelativePtHF",
        "RelativeSample",
        f"RelativeSample_{year}",
        "RelativeStatEC",
        "RelativeStatFSR",
        "RelativeStatHF",
        "SinglePionECAL",
        "SinglePionHCAL",
        "TimePtEta",
    ]

    btag_uncs = {
        # combined(?) uncertainties
        # uncertainties to b/c jets
        "down_bc": "bc_down",
        "up_bc": "bc_up",
        # uncertainties to light jets
        "down_light": "light_down",
        "up_light": "light_up",
        # split uncertainties(?) (all needed?)
        # uncertainties to b/c jets
        # "up_fsrdef_bc": "fsrdef_bc_up",
        # "up_isrdef_bc": "isrdef_bc_up",
        # "up_hdamp_bc": "hdamp_bc_up",
        # "up_jer_bc": "jer_bc_up",
        # "up_jes_bc": "jes_bc_up",
        # "up_mass_bc": "mass_bc_up",
        # "up_statistic_bc": "statistic_bc_up",
        # "up_tune_bc": "tune_bc_up",
        # "down_fsrdef_bc": "fsrdef_bc_down",
        # "down_isrdef_bc": "isrdef_bc_down",
        # "down_hdamp_bc": "hdamp_bc_down",
        # "down_jer_bc": "jer_bc_down",
        # "down_jes_bc": "jes_bc_down",
        # "down_mass_bc": "mass_bc_down",
        # "down_statistic_bc": "statistic_bc_down",
        # "down_tune_bc": "tune_bc_down",
        "up_bfragmentation_bc": "bfragmentation_bc_up",
        "up_pileup_bc": "pileup_bc_up",
        "up_type3_bc": "type3_bc_up",
        "up_statistic_bc": "statistic_bc_up",
        "down_bfragmentation_bc": "bfragmentation_bc_down",
        "down_pileup_bc": "pileup_bc_down",
        "down_type3_bc": "type3_bc_down",
        "down_statistic_bc": "statistic_bc_down",
        # uncertainties to light jets
        "down_correlated_light": "correlated_light_down",
        "up_correlated_light": "correlated_light_up",
        "down_uncorrelated_light": "uncorrelated_light_down",
        "up_uncorrelated_light": "uncorrelated_light_up",
    }
    if year == 2024:
        # TODO: use shape based BTagSFConfig when available
        # currently, one fixed WP is available for b tagging SF in 2024
        # implementation from hbt analysis:
        # https://github.com/uhh-cms/hh2bbtautau/blob/4b2f1bc57a9c2ada18776e5ac6f0372269e1e26c/hbt/config/configs_hbt.py#L1410 # noqa
        from columnflow.selection.cms.btag import BTagWPCountConfig
        btag_wp_count_config = BTagWPCountConfig(
            jet_name="Jet",
            btag_column=discr,
            btag_wps=config.x.btag_wp_names.UParTAK4,
            pt_edges=(0, 20, 30, 50, 70, 100, 140, 200, 300, 600, 10_000),
            abs_eta_edges=(0.0, 1.0, 1.5, 2.0, 5.0),
            # abs_eta_edges=(0.0, 1.5, 5.0),
        )

        from columnflow.production.cms.btag import BTagWPSFConfig

        def dataset_groups(dataset_inst: od.Dataset) -> list[od.Dataset]:
            # check which group the dataset belongs to
            for group_index in range(0, len(config.x.btag_wp_eff_groups)):
                group_tag = f"btag_wp_eff_group_{group_index}"
                if dataset_inst.has_tag(group_tag):
                    return [
                        _dataset_inst
                        for _dataset_inst in config.datasets
                        if _dataset_inst.has_tag(group_tag)
                    ]
            raise NotImplementedError(f"btag WP efficiency group not implemented for dataset {dataset_inst.name}")

        btag_wp_sf_config = BTagWPSFConfig(
            jet_name="Jet",
            btag_column=discr,
            correction_set="UParTAK4_merged",
            # btag_wps=config.x.btag_wp_names.UParTAK4,
            dataset_groups=dataset_groups,
            # pt_edges=(0, 10_000),
            pt_edges=(0, 20, 30, 50, 70, 100, 140, 200, 300, 600, 10_000),
            # abs_eta_edges=(0.0, 1.0, 1.5, 2.0, 5.0),
            systs=btag_uncs,
            wp_merging={
                "loose": ["loose"],
                "medium": ["medium"],
                "tight": ["tight"],
                "xtight": ["xtight", "xxtight"],
            },
            # further merge eta bins for sufficient statistics in each bin
            abs_eta_edges=(0.0, 5.0),
            btag_wps={
                "loose": 0.0246,
                "medium": 0.1272,
                "tight": 0.4648,
                "xtight": 0.6298,
                # "xxtight": 0.9739,
            },
        )
    else:
        raise NotImplementedError("B-tagging SFs for 2022 and 2023 not implemented yet.")

    configs = {
        "btag_wp_count_config": btag_wp_count_config,
        "btag_wp_sf_config": btag_wp_sf_config,
    }

    return configs
