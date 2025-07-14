# coding: utf-8

"""
Configuration of the HH -> bbWW config.
"""

from __future__ import annotations

import os

import yaml
from scinum import Number
import law
import order as od

from columnflow.util import DotDict
from columnflow.config_util import add_shift_aliases
from columnflow.columnar_util import ColumnCollection, skip_column
from hbw.config.styling import stylize_processes
from hbw.config.categories import add_categories_selection
from hbw.config.variables import add_variables
from hbw.config.datasets import add_hbw_processes_and_datasets, configure_hbw_datasets
from hbw.config.processes import configure_hbw_processes
from hbw.config.defaults_and_groups import set_config_defaults_and_groups
from hbw.config.sl_defaults_and_groups import set_sl_config_defaults_and_groups
from hbw.config.hist_hooks import add_hist_hooks
from hbw.config.scale_factors import configure_for_scale_factors
from hbw.util import timeit_multiple
from columnflow.production.cms.dy import DrellYanConfig

from columnflow.production.cms.electron import ElectronSFConfig
from columnflow.production.cms.muon import MuonSFConfig
from columnflow.production.cms.btag import BTagSFConfig
from columnflow.calibration.cms.egamma import EGammaCorrectionConfig
from columnflow.production.cms.jet import JetIdConfig

thisdir = os.path.dirname(os.path.abspath(__file__))

logger = law.logger.get_logger(__name__)


@timeit_multiple
def add_config(
    analysis: od.Analysis,
    campaign: od.Campaign,
    config_name: str | None = None,
    config_id: int | None = None,
    limit_dataset_files: int | None = None,
    add_dataset_extensions: bool = False,
) -> od.Config:
    # gather campaign data
    year = campaign.x.year
    year2 = year % 100

    corr_postfix = ""
    if campaign.x.year == 2016:
        if not campaign.has_tag("preVFP") and not campaign.has_tag("postVFP"):
            raise ValueError("2016 campaign must have the 'preVFP' or 'postVFP' tag")
        corr_postfix = "postVFP" if campaign.has_tag("postVFP") else "preVFP"
    elif campaign.x.year == 2022:
        if not campaign.has_tag("postEE") and not campaign.has_tag("preEE"):
            raise ValueError("2022 campaign must have the 'postEE' or 'preEE' tag")
        corr_postfix = "postEE" if campaign.has_tag("postEE") else "preEE"
    elif campaign.x.year == 2023:
        if not campaign.has_tag("postBPix") and not campaign.has_tag("preBPix"):
            raise ValueError("2023 campaign must have the 'postBPix' or 'preBPix' tag")
        corr_postfix = "postBPix" if campaign.has_tag("postBPix") else "preBPix"

    implemented_years = [2017, 2022, 2023]
    if campaign.x.year not in implemented_years:
        raise NotImplementedError(f"For now, only {', '.join(implemented_years)} years are implemented")

    # create a config by passing the campaign, so id and name will be identical
    # cfg = analysis.add_config(campaign, name=config_name, id=config_id, tags=analysis.tags)
    cfg = od.Config(name=config_name, id=config_id, campaign=campaign, tags=analysis.tags)

    # helper to enable processes / datasets only for a specific era
    def if_era(
        *,
        run: int | None = None,
        year: int | None = None,
        postfix: str | None = None,
        tag: str | None = None,
        cfg_tag: str | None = None,
        values: list[str] | None = None,
    ) -> list[str]:
        """
        Helper function to enable processes / datasets only for a specific era
        :param run: LHC Run, either 2 or 3
        :param year: Year of the data-taking campaign, e.g. 2017
        :param postfix: Additional postfix for the campaign, e.g. "EE"
        :param tag: Additional tag for the campaign, e.g. "postEE"
        :return: All values if the era matches, otherwise an empty list
        """
        match = (
            (run is None or campaign.x.run == run) and
            (year is None or campaign.x.year == year) and
            (postfix is None or campaign.x.postfix == postfix) and
            (tag is None or campaign.has_tag(tag)) and
            (cfg_tag is None or cfg.has_tag(cfg_tag))
        )
        return (values or []) if match else []

    cfg.x.if_era = if_era

    # add some important tags to the config
    # TODO: generalize and move to campaign
    cfg.x.cpn_tag = f"{year}{corr_postfix}"
    cfg.x.run = cfg.campaign.x.run

    if cfg.has_tag("is_sl"):
        cfg.x.lepton_tag = "sl"
    elif cfg.has_tag("is_dl"):
        cfg.x.lepton_tag = "dl"
    else:
        raise Exception(f"config {cfg.name} needs either the 'is_sl' or 'is_dl' tag")

    # add tag if used for scale factor calculation
    cfg.add_tag("is_for_sf")

    # define all resonant masspoints
    if cfg.has_tag("is_resonant"):
        cfg.x.graviton_masspoints = cfg.x.radion_masspoints = (
            250, 260, 270, 280, 300, 320, 350, 400, 450, 500,
            550, 600, 650, 700, 750, 800, 850, 900, 1000,
            1250, 1500, 1750, 2000, 2500, 3000,
        )

    ################################################################################################
    #
    # processes and datasets
    #
    ################################################################################################

    # add relevant processes and datasets to config
    add_hbw_processes_and_datasets(cfg, campaign)

    # configure processes in config
    configure_hbw_processes(cfg)

    # set color of some processes
    stylize_processes(cfg)

    # configure datasets in config
    configure_hbw_datasets(cfg, limit_dataset_files, add_dataset_extensions)

    # whether to validate the number of obtained LFNs in GetDatasetLFNs
    cfg.x.validate_dataset_lfns = limit_dataset_files is None

    ################################################################################################
    #
    # Luminosity
    #
    ################################################################################################

    # lumi values in inverse pb
    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2?rev=2#Combination_and_correlations
    if year == 2016:
        cfg.x.luminosity = Number(36310, {
            "lumi_13TeV_2016": 0.01j,
            "lumi_13TeV_correlated": 0.006j,
        })
    elif year == 2017:
        cfg.x.luminosity = Number(41480, {
            "lumi_13TeV_2017": 0.02j,
            "lumi_13TeV_1718": 0.006j,
            "lumi_13TeV_correlated": 0.009j,
        })
    elif year == 2018:
        cfg.x.luminosity = Number(59830, {
            "lumi_13TeV_2017": 0.015j,
            "lumi_13TeV_1718": 0.002j,
            "lumi_13TeV_correlated": 0.02j,
        })
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis
    elif year == 2022:
        if campaign.has_tag("preEE"):
            cfg.x.luminosity = Number(7971, {
                "lumi_13TeV_2022": 0.01j,
                "lumi_13TeV_correlated": 0.006j,
            })
        elif campaign.has_tag("postEE"):
            cfg.x.luminosity = Number(26337, {
                "lumi_13TeV_2022": 0.01j,
                "lumi_13TeV_correlated": 0.006j,
            })
    elif year == 2023:
        if campaign.has_tag("preBPix"):
            cfg.x.luminosity = Number(17794, {
                "lumi_13TeV_2023": 0.01j,
                "lumi_13TeV_correlated": 0.006j,
            })
        elif campaign.has_tag("postBPix"):
            cfg.x.luminosity = Number(9451, {
                "lumi_13TeV_2023": 0.01j,
                "lumi_13TeV_correlated": 0.006j,
            })
    else:
        raise NotImplementedError(f"Luminosity for year {year} is not defined.")

    # minimum bias cross section in mb (milli) for creating PU weights, values from
    # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData?rev=45#Recommended_cross_section
    # TODO: changes in Run3?
    cfg.x.minbias_xs = Number(69.2, 0.046j)

    ################################################################################################
    #
    # jet settings
    #
    ################################################################################################

    # JEC
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=201
    jerc_postfix = campaign.x.postfix
    if jerc_postfix not in ("", "APV", "EE", "BPix"):
        raise ValueError(f"Unknown JERC postfix '{jerc_postfix}'")

    if cfg.x.run == 2:
        jer_campaign = jec_campaign = f"Summer19UL{year2}{jerc_postfix}"
        jet_type = "AK4PFchs"
        fatjet_type = "AK8PFchs"
    elif cfg.x.run == 3:
        if year == 2022:
            jer_campaign = jec_campaign = f"Summer{year2}{jerc_postfix}_22Sep2023"
        elif year == 2023:
            # NOTE: this might be totally wrong, ask Daniel
            # TODO: fix for 2023postBPix....
            era = "Cv4" if campaign.has_tag("preBPix") else "D"
            jer_campaign = f"Summer{year2}{jerc_postfix}Prompt{year2}_Run{era}"
            jec_campaign = f"Summer{year2}{jerc_postfix}Prompt{year2}"
        jet_type = "AK4PFPuppi"
        fatjet_type = "AK8PFPuppi"

    jec_uncertainties = [
        # NOTE: there are many more sources available, but it is likely that we only need Total
        "Total",
        # "CorrelationGroupMPFInSitu",
        # "CorrelationGroupIntercalibration",
        # "CorrelationGroupbJES",
        # "CorrelationGroupFlavor",
        # "CorrelationGroupUncorrelated",
    ]

    cfg.x.jec = DotDict.wrap({
        # NOTE: currently, we set the uncertainty_sources in the calibrator itself
        "Jet": {
            "campaign": jec_campaign,
            "version": {2016: "V7", 2017: "V5", 2018: "V5", 2022: "V2", 2023: "V1"}[year],
            "jet_type": jet_type,
            "external_file_key": "jet_jerc",
            "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
            "levels_for_type1_met": ["L1FastJet"],
            "uncertainty_sources": jec_uncertainties,
        },
        "FatJet": {
            "campaign": jec_campaign,
            "version": {2016: "V7", 2017: "V5", 2018: "V5", 2022: "V2", 2023: "V1"}[year],
            "jet_type": fatjet_type,
            "external_file_key": "fat_jet_jerc",
            "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
            "levels_for_type1_met": ["L1FastJet"],
            "uncertainty_sources": jec_uncertainties,
        },
    })

    # JER
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107
    cfg.x.jer = DotDict.wrap({
        "Jet": {
            "campaign": jer_campaign,
            "version": {2016: "JRV3", 2017: "JRV2", 2018: "JRV2", 2022: "JRV1", 2023: "JRV1"}[year],
            "jet_type": jet_type,
            "external_file_key": "jet_jerc",
        },
        "FatJet": {
            "campaign": jer_campaign,
            "version": {2016: "JRV3", 2017: "JRV2", 2018: "JRV2", 2022: "JRV1", 2023: "JRV1"}[year],
            # "jet_type": "fatjet_type",
            # JER info only for AK4 jets, stored in AK4 file
            "jet_type": fatjet_type,
            "external_file_key": "jet_jerc",
        },
    })

    # JEC uncertainty sources propagated to btag scale factors
    # (names derived from contents in BTV correctionlib file)
    cfg.x.btag_sf_jec_sources = [
        "",  # total
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

    # b-tag working points
    # main source with all WPs: https://btv-wiki.docs.cern.ch/ScaleFactors/#sf-campaigns
    # 2016/17 sources with revision:
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16preVFP?rev=6
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16postVFP?rev=8
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=15
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=17
    cfg.x.btag_working_points = DotDict.wrap({
        "deepjet": {
            "loose": {"2016preVFP": 0.0508, "2016postVFP": 0.0480, "2017": 0.0532, "2018": 0.0490, "2022preEE": 0.0583, "2022postEE": 0.0614, "2023preBPix": 0.0479, "2023BPix": 0.048}.get(cfg.x.cpn_tag, 0.0),  # noqa
            "medium": {"2016preVFP": 0.2598, "2016postVFP": 0.2489, "2017": 0.3040, "2018": 0.2783, "2022preEE": 0.3086, "2022postEE": 0.3196, "2023preBPix": 0.2431, "2023BPix": 0.2435}.get(cfg.x.cpn_tag, 0.0),  # noqa
            "tight": {"2016preVFP": 0.6502, "2016postVFP": 0.6377, "2017": 0.7476, "2018": 0.7100, "2022preEE": 0.7183, "2022postEE": 0.73, "2023preBPix": 0.6553, "2023BPix": 0.6563}.get(cfg.x.cpn_tag, 0.0),  # noqa
        },
        "deepcsv": {
            "loose": {"2016preVFP": 0.2027, "2016postVFP": 0.1918, "2017": 0.1355, "2018": 0.1208}.get(cfg.x.cpn_tag, 0.0),  # noqa
            "medium": {"2016preVFP": 0.6001, "2016postVFP": 0.5847, "2017": 0.4506, "2018": 0.4168}.get(cfg.x.cpn_tag, 0.0),  # noqa
            "tight": {"2016preVFP": 0.8819, "2016postVFP": 0.8767, "2017": 0.7738, "2018": 0.7665}.get(cfg.x.cpn_tag, 0.0),  # noqa
        },
        "particlenet": {
            "loose": {"2022preEE": 0.047, "2022postEE": 0.0499, "2023preBPix": 0.0358, "2023postBPix": 0.359}.get(cfg.x.cpn_tag, 0.0),  # noqa
            "medium": {"2022preEE": 0.245, "2022postEE": 0.2605, "2023preBPix": 0.1917, "2023postBPix": 0.1919}.get(cfg.x.cpn_tag, 0.0),  # noqa
            "tight": {"2022preEE": 0.6734, "2022postEE": 0.6915, "2023preBPix": 0.6172, "2023postBPix": 0.6133}.get(cfg.x.cpn_tag, 0.0),  # noqa
        },
        # taken from preliminary studies from HH(4b)
        # source: https://indico.cern.ch/event/1372046/#2-run-3-particlenet-bb-sfs-sfb
        # different results here (0.8, 0.9, 0.95): https://indico.cern.ch/event/1428223/#21-calibration-of-run-3-partic
        "particlenet_xbb_vs_qcd": {
            "loose": {"2022preEE": 0.92, "2022postEE": 0.92, "2023preBPix": 0.92, "2023postBPix": 0.92}.get(cfg.x.cpn_tag, 0.0),  # noqa
            "medium": {"2022preEE": 0.95, "2022postEE": 0.95, "2023preBPix": 0.95, "2023postBPix": 0.95}.get(cfg.x.cpn_tag, 0.0),  # noqa
            "tight": {"2022preEE": 0.975, "2022postEE": 0.975, "2023preBPix": 0.975, "2023postBPix": 0.975}.get(cfg.x.cpn_tag, 0.0),  # noqa
        },
        "particlenet_hbb_vs_qcd": {
            # AK4 medium WP as placeholder (TODO: replace with actual values)
            "PLACEHOLDER": {"2022preEE": 0.245, "2022postEE": 0.2605, "2023preBPix": 0.1917, "2023postBPix": 0.1919}.get(cfg.x.cpn_tag, 0.0),  # noqa
        },
    })

    # b-tag configuration. Potentially overwritten by the jet Selector.
    if cfg.x.run == 2:
        cfg.x.b_tagger = "deepjet"
        cfg.x.btag_sf = BTagSFConfig(
            correction_set="deepJet_shape",
            jec_sources=cfg.x.btag_sf_jec_sources,
            discriminator="btagDeepFlavB",
            # corrector_kwargs=...,
        )
    elif cfg.x.run == 3:
        cfg.x.b_tagger = "particlenet"
        cfg.x.btag_sf = BTagSFConfig(
            correction_set="particleNet_shape",
            jec_sources=cfg.x.btag_sf_jec_sources,
            discriminator="btagPNetB",
            # corrector_kwargs=...,
        )

    cfg.x.btag_column = cfg.x.btag_sf.discriminator
    cfg.x.btag_wp = "medium"
    cfg.x.btag_wp_score = (
        cfg.x.btag_working_points[cfg.x.b_tagger][cfg.x.btag_wp]
    )
    if cfg.x.btag_wp_score == 0.0:
        raise ValueError(f"Unknown b-tag working point '{cfg.x.btag_wp}' for campaign {cfg.x.cpn_tag}")
    cfg.x.xbb_btag_wp_score = cfg.x.btag_working_points["particlenet_xbb_vs_qcd"]["medium"]
    if cfg.x.xbb_btag_wp_score == 0.0:
        raise ValueError(f"Unknown xbb b-tag working point 'medium' for campaign {cfg.x.cpn_tag}")

    # met configuration
    cfg.x.met_name = {
        2: "MET",
        3: "PuppiMET",
    }[cfg.x.run]
    cfg.x.raw_met_name = {
        2: "RawMET",
        3: "RawPuppiMET",
    }[cfg.x.run]

    # top pt reweighting parameters
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting#TOP_PAG_corrections_based_on_dat?rev=31
    cfg.x.top_pt_weight = {
        "a": 0.0615,
        "a_up": 0.0615 * 1.5,
        "a_down": 0.0615 * 0.5,
        "b": -0.0005,
        "b_up": -0.0005 * 1.5,
        "b_down": -0.0005 * 0.5,
    }

    # V+jets reweighting
    cfg.x.vjets_reweighting = DotDict.wrap({
        "z": {
            "value": "eej_pTV_kappa_NLO_EW",
            "ew": "eej_pTV_kappa_NLO_EW",
            "error": "eej_pTV_d1kappa_EW",  # NOTE: not sure if this is correct to use as error (d2,d3?)
            "d2": "eej_pTV_d2kappa_EW",
            "d3": "eej_pTV_d3kappa_EW",
        },
        "w": {
            "value": "aj_pTV_kappa_NLO_EW",
            "ew": "aj_pTV_kappa_NLO_EW",
            "error": "aj_pTV_d1kappa_EW",  # NOTE: not sure if this is correct to use as error (d2,d3?)
            "d2": "aj_pTV_d2kappa_EW",
            "d3": "aj_pTV_d3kappa_EW",
        },
        # "w": {
        #     "value": "wjets_kfactor_value",
        #     "error": "wjets_kfactor_error",
        # },
        # "z": {
        #     "value": "zjets_kfactor_value",
        #     "error": "zjets_kfactor_error",
        # },
    })

    ################################################################################################
    #
    # electron and muon calibrations and SFs (NOTE: we could add these config entries as part of the selector init)
    #
    ################################################################################################

    # electron calibrations
    cfg.x.eec = EGammaCorrectionConfig(
        correction_set=f"EGMScale_Compound_Ele_{cfg.x.cpn_tag}".replace("BPix", "BPIX"),
        value_type="scale",
        uncertainty_type="escale",
        compound=True,
    )
    # cfg.x.eec = EGammaCorrectionConfig(
    #     correction_set=f"EGMScale_ElePTsplit_{cfg.x.cpn_tag}",
    #     value_type="total_correction",
    #     uncertainty_type="escale",
    #     compound=False,
    # )
    cfg.x.eer = EGammaCorrectionConfig(
        correction_set=f"EGMSmearAndSyst_ElePTsplit_{cfg.x.cpn_tag}".replace("BPix", "BPIX"),
        value_type="smear",
        uncertainty_type="esmear",
    )

    if cfg.x.run == 2:
        # names of electron correction sets and working points
        # (used in the electron_sf producer)
        cfg.x.electron_sf_names = ElectronSFConfig(
            correction="UL-Electron-ID-SF",
            campaign=f"{cfg.x.cpn_tag}",
            working_point="Tight",
        )

        # names of muon correction sets and working points
        # (used in the muon producer)
        cfg.x.muon_id_sf_names = ("NUM_TightID_DEN_TrackerMuons", f"{cfg.x.cpn_tag}_UL")
        cfg.x.muon_iso_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", f"{cfg.x.cpn_tag}_UL")
    elif cfg.x.run == 3:
        electron_sf_campaign = {
            "2022postEE": "2022Re-recoE+PromptFG",
            "2022preEE": "2022Re-recoBCD",
            "2023postBPix": "2023PromptD",
            "2023preBPix": "2023PromptC",
        }[cfg.x.cpn_tag]

        cfg.x.electron_sf_names = ElectronSFConfig(
            correction="Electron-ID-SF",
            campaign=electron_sf_campaign,
            working_point="Tight",
        )
        # names of electron correction sets and working points
        # (used in the electron_sf producer)
        if cfg.x.cpn_tag == "2022postEE":
            # TODO: we might need to use different SFs for control regions
            cfg.x.electron_sf_names = ("Electron-ID-SF", "2022Re-recoE+PromptFG", "Tight")
        elif cfg.x.cpn_tag == "2022preEE":
            cfg.x.electron_sf_names = ("Electron-ID-SF", "2022Re-recoBCD", "Tight")

        # names of muon correction sets and working points
        # (used in the muon producer)
        # TODO: we might need to use different SFs for control regions
        cfg.x.muon_id_sf_names = MuonSFConfig(
            correction="NUM_TightID_DEN_TrackerMuons",
            campaign=f"{cfg.x.cpn_tag}",
        )
        cfg.x.muon_iso_sf_names = MuonSFConfig(
            correction="NUM_TightPFIso_DEN_TightID",
            campaign=f"{cfg.x.cpn_tag}",
        )

        # central trigger SF, only possible for SL
        if cfg.x.lepton_tag == "sl":
            # TODO: this should be year-dependent and setup in the selector
            cfg.x.muon_trigger_sf_names = ("NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight", f"{cfg.x.cpn_tag}")

    ################################################################################################
    #
    # shifts
    #
    ################################################################################################

    # register shifts
    cfg.add_shift(name="nominal", id=0)
    cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="mtop_up", id=5, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="mtop_down", id=6, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="minbias_xs_up", id=7, type="shape")
    cfg.add_shift(name="minbias_xs_down", id=8, type="shape")
    add_shift_aliases(
        cfg,
        "minbias_xs",
        {
            "normalized_pu_weight": "normalized_pu_weight_{name}",
            "pu_weight": "pu_weight_{name}",
        },
    )

    # top pt reweighting
    cfg.add_shift(name="top_pt_up", id=9, type="shape")
    cfg.add_shift(name="top_pt_down", id=10, type="shape")
    add_shift_aliases(cfg, "top_pt", {"top_pt_weight": "top_pt_weight_{direction}"})

    # V+jets reweighting
    cfg.add_shift(name="vjets_up", id=11, type="shape")
    cfg.add_shift(name="vjets_down", id=12, type="shape")
    add_shift_aliases(cfg, "vjets", {"vjets_weight": "vjets_weight_{direction}"})

    # electron scale factor uncertainties
    cfg.add_shift(name="e_sf_up", id=40, type="shape")
    cfg.add_shift(name="e_sf_down", id=41, type="shape")
    cfg.add_shift(name="e_trig_sf_up", id=42, type="shape")
    cfg.add_shift(name="e_trig_sf_down", id=43, type="shape")
    add_shift_aliases(cfg, "e_sf", {"electron_weight": "electron_weight_{direction}"})
    # add_shift_aliases(cfg, "e_trig_sf", {"electron_trigger_weight": "electron_trigger_weight_{direction}"})

    # cfg.add_shift(name="mu_sf_up", id=50, type="shape")
    # cfg.add_shift(name="mu_sf_down", id=51, type="shape")
    # add_shift_aliases(cfg, "mu_sf", {"muon_weight": "muon_weight_{direction}"})

    # muon scale factor uncertainties
    cfg.add_shift(name="mu_trig_sf_up", id=52, type="shape")
    cfg.add_shift(name="mu_trig_sf_down", id=53, type="shape")
    cfg.add_shift(name="mu_id_sf_up", id=54, type="shape")
    cfg.add_shift(name="mu_id_sf_down", id=55, type="shape")
    cfg.add_shift(name="mu_iso_sf_up", id=56, type="shape")
    cfg.add_shift(name="mu_iso_sf_down", id=57, type="shape")
    add_shift_aliases(cfg, "mu_id_sf", {"muon_id_weight": "muon_id_weight_{direction}"})
    add_shift_aliases(cfg, "mu_iso_sf", {"muon_iso_weight": "muon_iso_weight_{direction}"})
    # add_shift_aliases(cfg, "mu_trig_sf", {"muon_trigger_weight": "muon_trigger_weight_{direction}"})

    # trigger SFs
    cfg.add_shift(name="trigger_sf_up", id=60, type="shape")
    cfg.add_shift(name="trigger_sf_down", id=61, type="shape")
    add_shift_aliases(cfg, "trigger_sf", {"trigger_weight": "trigger_weight_{direction}"})

    # b-tagging scale factor uncertainties
    cfg.x.btag_uncs = [
        "hf", "lf", f"hfstats1_{year}", f"hfstats2_{year}",
        f"lfstats1_{year}", f"lfstats2_{year}", "cferr1", "cferr2",
    ]
    for i, unc in enumerate(cfg.x.btag_uncs):
        cfg.add_shift(name=f"btag_{unc}_up", id=100 + 2 * i, type="shape")
        cfg.add_shift(name=f"btag_{unc}_down", id=101 + 2 * i, type="shape")
        add_shift_aliases(
            cfg,
            f"btag_{unc}",
            {
                btag_weight: f"{btag_weight}_{unc}_" + "{direction}"
                for btag_weight in (
                    "btag_weight",
                    "normalized_btag_weight",
                    "normalized_njet_btag_weight",
                    "normalized_ht_njet_btag_weight",
                    "normalized_ht_njet_nhf_btag_weight",
                    "normalized_ht_btag_weight",
                )
            },
        )

    cfg.add_shift(name="mur_up", id=201, type="shape")
    cfg.add_shift(name="mur_down", id=202, type="shape")
    cfg.add_shift(name="muf_up", id=203, type="shape")
    cfg.add_shift(name="muf_down", id=204, type="shape")
    cfg.add_shift(name="murf_envelope_up", id=205, type="shape")
    cfg.add_shift(name="murf_envelope_down", id=206, type="shape")
    cfg.add_shift(name="pdf_up", id=207, type="shape")
    cfg.add_shift(name="pdf_down", id=208, type="shape")

    for unc in ["mur", "muf", "murf_envelope", "pdf"]:
        col = "murmuf_envelope" if unc == "murf_envelope" else unc
        add_shift_aliases(
            cfg,
            unc,
            {
                f"normalized_{col}_weight": f"normalized_{unc}_weight_" + "{direction}",
                f"{col}_weight": f"{unc}_weight_" + "{direction}",
            },
        )

    cfg.add_shift(name=f"dummy_{cfg.x.cpn_tag}_up", id=209, type="shape")
    cfg.add_shift(name=f"dummy_{cfg.x.cpn_tag}_down", id=210, type="shape")
    add_shift_aliases(
        cfg,
        f"dummy_{cfg.x.cpn_tag}",
        {
            "dummy_weight": f"dummy_{cfg.x.cpn_tag}_weight_" + "{direction}",
        },
    )
    # cfg.add_shift(name="dummy_2022postEE_up", id=209, type="shape")
    # cfg.add_shift(name="dummy_2022postEE_down", id=210, type="shape")
    # add_shift_aliases(
    #     cfg,
    #     "dummy_2022postEE",
    #     {
    #         "dummy_weight": "dummy_2022postEE_weight_" + "{direction}",
    #     },
    # )
    # cfg.add_shift(name="dummy_2022preEE_up", id=211, type="shape")
    # cfg.add_shift(name="dummy_2022preEE_down", id=212, type="shape")
    # add_shift_aliases(
    #     cfg,
    #     "dummy_2022preEE",
    #     {
    #         "dummy_weight": "dummy_2022preEE_weight_" + "{direction}",
    #     },
    # )

    with open(os.path.join(thisdir, "jec_sources.yaml"), "r") as f:
        all_jec_sources = yaml.load(f, yaml.Loader)["names"]

    for jec_source in cfg.x.jec.Jet["uncertainty_sources"]:
        idx = all_jec_sources.index(jec_source)
        cfg.add_shift(
            name=f"jec_{jec_source}_up",
            id=5000 + 2 * idx,
            type="shape",
            tags={"jec"},
            aux={"jec_source": jec_source},
        )
        cfg.add_shift(
            name=f"jec_{jec_source}_down",
            id=5001 + 2 * idx,
            type="shape",
            tags={"jec"},
            aux={"jec_source": jec_source},
        )
        add_shift_aliases(
            cfg,
            f"jec_{jec_source}",
            {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"},
        )

        if jec_source in ["Total", *cfg.x.btag_sf_jec_sources]:
            # when jec_source is a known btag SF source, add aliases for btag weight column
            add_shift_aliases(
                cfg,
                f"jec_{jec_source}",
                {
                    btag_weight: f"{btag_weight}_jec_{jec_source}_" + "{direction}"
                    for btag_weight in (
                        "btag_weight",
                        "normalized_btag_weight",
                        "normalized_njet_btag_weight",
                        "normalized_ht_njet_btag_weight",
                        "normalized_ht_njet_nhf_btag_weight",
                        "normalized_ht_btag_weight",
                    )
                },
            )

    cfg.add_shift(name="jer_up", id=6000, type="shape", tags={"jer"})
    cfg.add_shift(name="jer_down", id=6001, type="shape", tags={"jer"})
    add_shift_aliases(cfg, "jer", {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"})

    ################################################################################################
    #
    # external files
    #
    ################################################################################################

    cfg.x.external_files = DotDict()

    # helper
    def add_external(name, value):
        if isinstance(value, dict):
            value = DotDict.wrap(value)
        cfg.x.external_files[name] = value

    json_mirror = "/afs/cern.ch/user/m/mfrahm/public/mirrors/jsonpog-integration-a1ba637b"
    if cfg.x.run == 2:
        # json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c"
        corr_tag = f"{cfg.x.cpn_tag}_UL"
    elif cfg.x.run == 3:
        corr_tag = f"{year}_Summer{year2}{jerc_postfix}"

    # pileup weight correction
    add_external("pu_sf", (f"{json_mirror}/POG/LUM/{corr_tag}/puWeights.json.gz", "v1"))
    # jet energy correction
    add_external("jet_jerc", (f"{json_mirror}/POG/JME/{corr_tag}/jet_jerc.json.gz", "v1"))
    add_external("fat_jet_jerc", (f"{json_mirror}/POG/JME/{corr_tag}/fatJet_jerc.json.gz", "v1"))
    # jet veto map
    add_external("jet_veto_map", (f"{json_mirror}/POG/JME/{corr_tag}/jetvetomaps.json.gz", "v1"))
    # jet id fix
    add_external("jet_id", (f"{json_mirror}/POG/JME/{corr_tag}/jetid.json.gz", "v1"))
    cfg.x.jet_id = JetIdConfig(corrections={
        "AK4PUPPI_Tight": 2,
        "AK4PUPPI_TightLeptonVeto": 3,
    })
    cfg.x.fatjet_id = JetIdConfig(corrections={
        "AK8PUPPI_Tight": 2,
        "AK8PUPPI_TightLeptonVeto": 3,
    })
    # electron scale factors
    add_external("electron_sf", (f"{json_mirror}/POG/EGM/{corr_tag}/electron.json.gz", "v1"))
    add_external("electron_ss", (f"{json_mirror}/POG/EGM/{corr_tag}/electronSS_EtDependent.json.gz", "v1"))
    # muon scale factors
    add_external("muon_sf", (f"{json_mirror}/POG/MUO/{corr_tag}/muon_Z.json.gz", "v1"))

    # trigger_sf from Balduin
    trigger_sf_path = f"{json_mirror}/data/trig_sf_v1"

    add_external("trigger_sf_ee", (f"{trigger_sf_path}/sf_ee_mli_lep_pt-mli_lep2_pt-trig_idsv1.json", "v3"))
    add_external("trigger_sf_mm", (f"{trigger_sf_path}/sf_mm_mli_lep_pt-mli_lep2_pt-trig_idsv1.json", "v3"))
    add_external("trigger_sf_mixed", (f"{trigger_sf_path}/sf_mixed_mli_lep_pt-mli_lep2_pt-trig_idsv2.json", "v3"))  # noqa: E501

    # trigger configuration (can be overwritten in the Selector)
    from hbw.config.trigger import add_triggers
    add_triggers(cfg)

    # btag scale factor
    add_external("btag_sf_corr", (f"{json_mirror}/POG/BTV/{corr_tag}/btagging.json.gz", "v2"))
    # V+jets reweighting (derived for 13 TeV, custom json converted from ROOT, not centrally produced)
    # ROOT files (eej.root and aj.root) taken from here:
    # https://github.com/UHH2/2HDM/tree/ultra_legacy/data/ScaleFactors/VJetsCorrections
    add_external("vjets_reweighting", (f"{json_mirror}/data/json/vjets_pt.json.gz", "v1"))
    if cfg.x.run == 2:
        # met phi corrector (still unused and missing in Run3)
        add_external("met_phi_corr", (f"{json_mirror}/POG/JME/{corr_tag}/met.json.gz", "v1"))

    add_external("dy_weight_sf", (f"{json_mirror}/data/dy/DY_pTll_weights_v2.json.gz", "v1"))
    add_external("dy_recoil_sf", (f"{json_mirror}/data/dy/Recoil_corrections_v2.json.gz", "v2"))

    cfg.x.dy_weight_config = DrellYanConfig(
        era="2022postEE",
        order="NLO",  # only when using v2
        correction="DY_pTll_reweighting",
        unc_correction="DY_pTll_reweighting_N_uncertainty",
    )
    cfg.x.dy_recoil_config = DrellYanConfig(
        era="2022postEE",
        order="NLO",  # only when using v2
        correction="Recoil_correction_QuantileMapHist",
        unc_correction="Recoil_correction_Uncertainty",
    )

    # Louvain Transformer Model
    add_external("transformer_even", ("/afs/cern.ch/user/m/mfrahm/public/transformer/v1.2.3_even_model/model.onnx", "v1.2.3"))  # noqa: E501
    add_external("transformer_odd", ("/afs/cern.ch/user/m/mfrahm/public/transformer/v1.2.3_odd_model/model.onnx", "v1.2.3"))  # noqa: E501

    # documentation: https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2?rev=167
    if cfg.x.run == 2:
        cfg.x.met_filters = {
            "Flag.goodVertices",
            "Flag.globalSuperTightHalo2016Filter",
            "Flag.HBHENoiseFilter",
            "Flag.HBHENoiseIsoFilter",
            "Flag.EcalDeadCellTriggerPrimitiveFilter",
            "Flag.BadPFMuonFilter",
            "Flag.BadPFMuonDzFilter",  # this filter does not work with our EOY Signal samples
            "Flag.eeBadScFilter",
        }
    else:  # run == 3
        # NOTE: the "Flag.ecalBadCalibFilter" is currently missing in NanoAOD,
        # there is a receipe we might want to apply instead
        cfg.x.met_filters = {
            "Flag.goodVertices",
            "Flag.globalSuperTightHalo2016Filter",
            "Flag.EcalDeadCellTriggerPrimitiveFilter",
            "Flag.BadPFMuonFilter",
            "Flag.BadPFMuonDzFilter",
            "Flag.hfNoisyHitsFilter",
            "Flag.eeBadScFilter",
        }

    # external files with more complex year dependence
    if year not in (2017, 2022, 2023):
        raise NotImplementedError("TODO: generalize external files to different years than 2017")

    if year == 2017:
        add_external("lumi", {
            # files from TODO
            "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
            "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
        })
    elif year == 2022:
        add_external("lumi", {
            # files from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile
            "golden": ("https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
            "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
        })
    elif year == 2023:
        add_external("lumi", {
            # files from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile
            "golden": ("https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions23/Cert_Collisions2023_366442_370790_Golden.json", "v1"),  # noqa
            "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
        })
    elif year == 2024:
        add_external("lumi", {
            # files from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile
            # TODO: should be updated at the end of 2024 campaign
            "golden": ("https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions24/Cert_Collisions2024_378981_381417_Golden.json", "v1"),  # noqa
            "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
        })
    else:
        raise NotImplementedError(f"No lumi and pu files provided for year {year}")

    # columns to keep after certain steps
    # TODO: selector-dependent columns should either use the is_sl / is_dl tag or
    #       be implemented as part of the selectors itself (e.g. only SL needs the Lightjet column)
    cfg.x.keep_columns = DotDict.wrap({
        "cf.MergeSelectionMasks": {
            "mc_weight", "normalization_weight", "process_id", "category_ids", "cutflow.*",
            "HbbJet.n_subjets", "HbbJet.n_separated_jets", "HbbJet.max_dr_ak4", "gen_hbw_decay",
        },
    })

    cfg.x.keep_columns["cf.ReduceEvents"] = {
        # general event information, mandatory for reading files with coffea
        "run", "luminosityBlock", "event",
        ColumnCollection.MANDATORY_COFFEA,
        # columns added during selection, required in general
        "mc_weight", "PV.npvs", "process_id", "category_ids", "deterministic_seed",
        # Gen information (for categorization)
        "HardGenPart.pdgId",
        # Gen information for pt reweighting
        "GenPartonTop.pt", "GenVBoson.pt",
        "gen_dilepton_pt", "gen_dilepton_{vis,all}.{pt,phi}",
        # weight-related columns
        "pu_weight*", "pdf_weight*",
        "murmuf_envelope_weight*", "mur_weight*", "muf_weight*",
        "btag_weight*",
        # Gen particle information
        "gen_hbw_decay.*.*",
        # columns for btag reweighting crosschecks
        "njets", "ht", "nhf",
        # Jets (NOTE: we might want to store a local index to simplify selecting jet subcollections later on)
        "{Jet,ForwardJet,Bjet,Lightjet,VBFJet}.{pt,eta,phi,mass,btagDeepFlavB,btagPNetB,hadronFlavour,qgl}",
        # FatJets
        "{FatJet,HbbJet}.{pt,eta,phi,mass,msoftdrop,tau1,tau2,tau3,btagHbb,deepTagMD_HbbvsQCD}",
        # FatJet particleNet scores (all for now, should be reduced at some point)
        "FatJet.particleNet*",
        "{FatJet,HbbJet}.particleNet_{XbbVsQCD,massCorr}",
        "{FatJet,HbbJet}.particleNetWithMass_HbbvsQCD",
        # Leptons
        "{Electron,Muon}.{pt,eta,phi,mass,charge,pdgId,jetRelIso,is_tight,dxy,dz}",
        "Electron.{deltaEtaSC,r9,seedGain}", "mll",
        # isolations for testing
        "Electron.{pfRelIso03_all,miniPFRelIso_all,mvaIso,mvaTTH}",
        "Muon.{pfRelIso03_all,miniPFRelIso_all,mvaMuID,mvaTTH}",
        # Taus
        "VetoTau.{pt,eta,phi,mass,decayMode}",
        # MET
        "{MET,PuppiMET}.{pt,phi}",
        # steps
        "steps.*",
        # Trigger-related
        "trigger_ids", "trigger_data.*",
        "HLT.IsoMu24",
        "HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        "HLT.Ele30_WPTight_Gsf",
        "HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
        "HLT.Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        "HLT.Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        "HLT.Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
        # "TrigObj.{pt,eta,phi,mass,filterBits}",  # NOTE: this column is very large (~1/3 of final reduced events)
        # all columns added during selection using a ColumnCollection flag, but skip cutflow ones
        ColumnCollection.ALL_FROM_SELECTOR,
        skip_column("cutflow.*"),
    } | {
        "HLT.{trg.hlt_field}" for trg in cfg.get_aux("triggers", [])
    } | {
        *cfg.x.if_era(cfg_tag="is_sl", values=["Lightjet.btagPNetQvG"])
    }

    # Version of required tasks
    cfg.x.versions = {
        "cf.CalibrateEvents": law.config.get_expanded("analysis", "default_common_version", "common2"),
    }

    # add categories
    add_categories_selection(cfg)

    # add variables
    add_variables(cfg)

    # add hist hooks
    add_hist_hooks(cfg)

    # set some config defaults and groups
    # TODO: it might make sense to completely separate this for SL/DL
    if cfg.has_tag("is_sl"):
        set_sl_config_defaults_and_groups(cfg)
    elif cfg.has_tag("is_dl"):
        set_config_defaults_and_groups(cfg)

    # only produce cutflow features when number of dataset_files is limited (used in selection module)
    cfg.x.do_cutflow_features = bool(limit_dataset_files) and limit_dataset_files <= 10

    # customization based on the type of sub-analysis
    if cfg.has_tag("is_sl") and cfg.has_tag("is_nonresonant"):
        from hbw.config.sl import configure_sl
        configure_sl(cfg)
    if cfg.has_tag("is_dl") and cfg.has_tag("is_nonresonant"):
        from hbw.config.dl import configure_dl
        configure_dl(cfg)
    if cfg.has_tag("is_sl") and cfg.has_tag("is_resonant"):
        from hbw.config.sl_res import configure_sl_res
        configure_sl_res(cfg)

    # add configuration changes for scale factor calculations
    if cfg.has_tag("is_for_sf"):
        configure_for_scale_factors(cfg)

    # sanity check: sometimes the process is not the same as the one in the dataset
    p1 = cfg.get_process("dy_m50toinf")
    p2 = campaign.get_dataset("dy_m50toinf_amcatnlo").processes.get_first()
    # if repr(p1) != repr(p2):
    if p1 != p2:
        raise Exception(f"Processes are not the same: {repr(p1)} != {repr(p2)}")

    return cfg
