# coding: utf-8

"""
Configuration of the Run 2 HH -> bbWW analysis.
"""

from __future__ import annotations

import os
import re

import yaml
from scinum import Number
import law
import order as od

from columnflow.util import DotDict
from columnflow.config_util import get_root_processes_from_campaign
from hbw.config.styling import stylize_processes
from hbw.config.categories import add_categories_selection
from hbw.config.variables import add_variables
from hbw.config.datasets import get_dataset_lfns, get_custom_hh_datasets
from hbw.config.defaults_and_groups import set_config_defaults_and_groups
from hbw.util import four_vec


thisdir = os.path.dirname(os.path.abspath(__file__))

logger = law.logger.get_logger(__name__)


def add_config(
    analysis: od.Analysis,
    campaign: od.Campaign,
    config_name: str | None = None,
    config_id: int | None = None,
    limit_dataset_files: int | None = None,
) -> od.Config:
    # validations
    assert campaign.x.year in [2016, 2017, 2018]
    if campaign.x.year == 2016:
        assert campaign.x.vfp in ["pre", "post"]
    # gather campaign data
    year = campaign.x.year
    year2 = year % 100
    corr_postfix = f"{campaign.x.vfp}VFP" if year == 2016 else ""

    if year != 2017:
        raise NotImplementedError("For now, only 2017 campaign is fully implemented")

    # load custom produced datasets into campaign
    get_custom_hh_datasets(campaign)

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # create a config by passing the campaign, so id and name will be identical
    cfg = analysis.add_config(campaign, name=config_name, id=config_id, tags=analysis.tags)

    # use custom get_dataset_lfns function
    cfg.x.get_dataset_lfns = get_dataset_lfns

    # add processes we are interested in
    cfg.add_process(procs.n.data)
    cfg.add_process(procs.n.tt)
    cfg.add_process(procs.n.st)
    cfg.add_process(procs.n.w_lnu)
    cfg.add_process(procs.n.dy_lep)
    cfg.add_process(procs.n.qcd)
    cfg.add_process(procs.n.qcd_mu)
    cfg.add_process(procs.n.qcd_em)
    cfg.add_process(procs.n.qcd_bctoe)
    # cfg.add_process(procs.n.ttv)
    # cfg.add_process(procs.n.vv)
    # cfg.add_process(procs.n.vv)
    # cfg.add_process(procs.n.hh_ggf_bbtautau)

    if cfg.has_tag("is_sl") and cfg.has_tag("is_nonresonant"):
        cfg.add_process(procs.n.ggHH_kl_0_kt_1_sl_hbbhww)
        cfg.add_process(procs.n.ggHH_kl_1_kt_1_sl_hbbhww)
        cfg.add_process(procs.n.ggHH_kl_2p45_kt_1_sl_hbbhww)
        cfg.add_process(procs.n.ggHH_kl_5_kt_1_sl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_1_C2V_1_kl_1_sl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_1_C2V_1_kl_0_sl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_1_C2V_1_kl_2_sl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_1_C2V_0_kl_1_sl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_1_C2V_2_kl_1_sl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_0p5_C2V_1_kl_1_sl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_1p5_C2V_1_kl_1_sl_hbbhww)

    if cfg.has_tag("is_dl") and cfg.has_tag("is_nonresonant"):
        cfg.add_process(procs.n.ggHH_kl_0_kt_1_dl_hbbhww)
        cfg.add_process(procs.n.ggHH_kl_1_kt_1_dl_hbbhww)
        cfg.add_process(procs.n.ggHH_kl_2p45_kt_1_dl_hbbhww)
        cfg.add_process(procs.n.ggHH_kl_5_kt_1_dl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_1_C2V_1_kl_1_dl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_1_C2V_1_kl_0_dl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_1_C2V_1_kl_2_dl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_1_C2V_0_kl_1_dl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_1_C2V_2_kl_1_dl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_0p5_C2V_1_kl_1_dl_hbbhww)
        cfg.add_process(procs.n.qqHH_CV_1p5_C2V_1_kl_1_dl_hbbhww)

    # QCD process customization
    cfg.get_process("qcd_mu").label = "QCD Muon enriched"
    qcd_ele = cfg.add_process(
        name="qcd_ele",
        id=31199,
        xsecs={13: cfg.get_process("qcd_em").get_xsec(13) + cfg.get_process("qcd_bctoe").get_xsec(13)},
        label="QCD Electron enriched",
    )
    qcd_ele.add_process(cfg.get_process("qcd_em"))
    qcd_ele.add_process(cfg.get_process("qcd_bctoe"))

    # Custom v_lep process for ML Training, combining W+DY
    v_lep = cfg.add_process(
        name="v_lep",
        id=64575573,  # random number
        xsecs={13: cfg.get_process("w_lnu").get_xsec(13) + cfg.get_process("dy_lep").get_xsec(13)},
        label="W and DY",
    )
    v_lep.add_process(cfg.get_process("w_lnu"))
    v_lep.add_process(cfg.get_process("dy_lep"))
    
    # Custom t_bkg process for ML Training, combining tt + st
    t_bkg = cfg.add_process(
        name="t_bkg",
        id=81575573,  # random number
        xsecs={13: cfg.get_process("tt").get_xsec(13) + cfg.get_process("st").get_xsec(13)},
        label="tt and st",
    )
    t_bkg.add_process(cfg.get_process("tt"))
    t_bkg.add_process(cfg.get_process("st"))
    
    
    # Custom t_bkg process for ML Training, combining tt + st
    sg = cfg.add_process(
        name="sg",
        id=99975573,  # random number
        xsecs={13: cfg.get_process(procs.n.ggHH_kl_0_kt_1_dl_hbbhww).get_xsec(13) + cfg.get_process(procs.n.ggHH_kl_1_kt_1_dl_hbbhww).get_xsec(13) + cfg.get_process(procs.n.ggHH_kl_2p45_kt_1_dl_hbbhww).get_xsec(13)},
        label="HH",
    )
    sg.add_process(cfg.get_process(procs.n.ggHH_kl_0_kt_1_dl_hbbhww))
    sg.add_process(cfg.get_process(procs.n.ggHH_kl_1_kt_1_dl_hbbhww))
    sg.add_process(cfg.get_process(procs.n.ggHH_kl_2p45_kt_1_dl_hbbhww))
    
    '''
    # Custom all_sg process for ML Training, combining signal processes with different kl 
    all_kl = cfg.add_process(
        name="all_kl",
        id=91588473,  # random number
        xsecs={13: cfg.get_process("ggHH_kl_0_kt_1_dl_hbbhww").get_xsec(13) + cfg.get_process("ggHH_kl_1_kt_1_dl_hbbhww").get_xsec(13) + cfg.get_process("ggHH_kl_2p45_kt_1_dl_hbbhww").get_xsec(13)},
        label="HH (dl)",
    )
    all_kl.add_process(cfg.get_process("ggHH_kl_0_kt_1_dl_hbbhww"))
    all_kl.add_process(cfg.get_process("ggHH_kl_1_kt_1_dl_hbbhww"))
    all_kl.add_process(cfg.get_process("ggHH_kl_2p45_kt_1_dl_hbbhww"))
    '''

    # set color of some processes
    stylize_processes(cfg)

    # add datasets we need to study
    dataset_names = [
        # DATA
        "data_e_b",
        "data_e_c",
        "data_e_d",
        "data_e_e",
        "data_e_f",
        "data_mu_b",
        "data_mu_c",
        "data_mu_d",
        "data_mu_e",
        "data_mu_f",
        # TTbar
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        # SingleTop
        "st_tchannel_t_powheg",
        # "st_tchannel_tbar_powheg", # problems with weights in dproduce columns
        "st_twchannel_t_powheg",
        "st_twchannel_tbar_powheg",
        "st_schannel_lep_amcatnlo",
        # "st_schannel_had_amcatnlo",  # NOTE: this dataset produces some weird errors, so skip it for now
        # WJets
        "w_lnu_ht70To100_madgraph",
        "w_lnu_ht100To200_madgraph",
        "w_lnu_ht200To400_madgraph",
        "w_lnu_ht400To600_madgraph",
        "w_lnu_ht600To800_madgraph",
        "w_lnu_ht800To1200_madgraph",
        "w_lnu_ht1200To2500_madgraph",
        "w_lnu_ht2500_madgraph",
        # DY
        "dy_lep_m50_ht70to100_madgraph",
        "dy_lep_m50_ht100to200_madgraph",
        "dy_lep_m50_ht200to400_madgraph",
        "dy_lep_m50_ht400to600_madgraph",
        "dy_lep_m50_ht600to800_madgraph",
        "dy_lep_m50_ht800to1200_madgraph",
        "dy_lep_m50_ht1200to2500_madgraph",
        "dy_lep_m50_ht2500_madgraph",
        # QCD (no LHEScaleWeight)
        "qcd_mu_pt15to20_pythia", "qcd_mu_pt20to30_pythia",
        "qcd_mu_pt30to50_pythia", "qcd_mu_pt50to80_pythia",
        "qcd_mu_pt80to120_pythia", "qcd_mu_pt120to170_pythia",
        "qcd_mu_pt170to300_pythia", "qcd_mu_pt300to470_pythia",
        "qcd_mu_pt470to600_pythia", "qcd_mu_pt600to800_pythia",
        "qcd_mu_pt800to1000_pythia", "qcd_mu_pt1000_pythia",
        "qcd_em_pt15to20_pythia", "qcd_em_pt20to30_pythia",
        "qcd_em_pt30to50_pythia", "qcd_em_pt50to80_pythia",
        "qcd_em_pt80to120_pythia", "qcd_em_pt120to170_pythia",
        "qcd_em_pt170to300_pythia", "qcd_em_pt300toInf_pythia",
        "qcd_bctoe_pt15to20_pythia", "qcd_bctoe_pt20to30_pythia",
        "qcd_bctoe_pt30to80_pythia", "qcd_bctoe_pt80to170_pythia",
        "qcd_bctoe_pt170to250_pythia", "qcd_bctoe_pt250toInf_pythia",
        # TTV, VV -> ignore?; Higgs -> not used in Msc, but would be interesting
        # HH(bbtautau)
        # "hh_ggf_bbtautau_madgraph",
    ]

    if cfg.has_tag("is_sl") and cfg.has_tag("is_nonresonant"):
        # non-resonant HH -> bbWW(qqlnu) Signal
        if cfg.has_tag("custom_signals"):
            dataset_names += [
                "ggHH_kl_0_kt_1_sl_hbbhww_custom",
                "ggHH_kl_1_kt_1_sl_hbbhww_custom",
                "ggHH_kl_2p45_kt_1_sl_hbbhww_custom",
                "ggHH_kl_5_kt_1_sl_hbbhww_custom",
            ]
        else:
            dataset_names += [
                "ggHH_kl_0_kt_1_sl_hbbhww_powheg",
                "ggHH_kl_1_kt_1_sl_hbbhww_powheg",
                "ggHH_kl_2p45_kt_1_sl_hbbhww_powheg",
                "ggHH_kl_5_kt_1_sl_hbbhww_powheg",
            ]

        dataset_names += [
            "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww_madgraph",
            "qqHH_CV_1_C2V_1_kl_0_sl_hbbhww_madgraph",
            "qqHH_CV_1_C2V_1_kl_2_sl_hbbhww_madgraph",
            "qqHH_CV_1_C2V_0_kl_1_sl_hbbhww_madgraph",
            "qqHH_CV_1_C2V_2_kl_1_sl_hbbhww_madgraph",
            "qqHH_CV_0p5_C2V_1_kl_1_sl_hbbhww_madgraph",
            "qqHH_CV_1p5_C2V_1_kl_1_sl_hbbhww_madgraph",
        ]

    if cfg.has_tag("is_dl") and cfg.has_tag("is_nonresonant"):
        # non-resonant HH -> bbWW(lnulnu) Signal
        dataset_names += [
            "ggHH_kl_0_kt_1_dl_hbbhww_powheg",
            "ggHH_kl_1_kt_1_dl_hbbhww_powheg",
            "ggHH_kl_2p45_kt_1_dl_hbbhww_powheg",
            "ggHH_kl_5_kt_1_dl_hbbhww_powheg",
            "qqHH_CV_1_C2V_1_kl_1_dl_hbbhww_madgraph",
            "qqHH_CV_1_C2V_1_kl_0_dl_hbbhww_madgraph",
            "qqHH_CV_1_C2V_1_kl_2_dl_hbbhww_madgraph",
            "qqHH_CV_1_C2V_0_kl_1_dl_hbbhww_madgraph",
            "qqHH_CV_1_C2V_2_kl_1_dl_hbbhww_madgraph",
            "qqHH_CV_0p5_C2V_1_kl_1_dl_hbbhww_madgraph",
            "qqHH_CV_1p5_C2V_1_kl_1_dl_hbbhww_madgraph",
        ]
    if cfg.has_tag("is_resonant"):
        logger.warning(f"For analysis {analysis.name}: resonant samples still needs to be implemented")

    for dataset_name in dataset_names:
        dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

        if limit_dataset_files:
            # apply optional limit on the max. number of files per dataset
            for info in dataset.info.values():
                if info.n_files > limit_dataset_files:
                    info.n_files = limit_dataset_files

        # add aux info to datasets
        # TODO: switch from aux to tags for booleans
        if dataset.name.startswith(("st", "tt")):
            dataset.x.has_top = True
            dataset.add_tag("has_top")
        if dataset.name.startswith("tt"):
            dataset.x.is_ttbar = True
            dataset.add_tag("is_ttbar")
        if dataset.name.startswith("qcd"):
            dataset.x.is_qcd = True
            dataset.add_tag("is_qcd")
        if "HH" in dataset.name and "hbbhww" in dataset.name:
            # TODO: the is_hbw tag is used at times were we should ask for is_hbw_sl
            dataset.add_tag("is_hbw")
            dataset.x.is_hbw = True
            if "_sl_" in dataset.name:
                dataset.add_tag("is_hbw_sl")
            elif "_dl_" in dataset.name:
                dataset.add_tag("is_hbw_dl")

        if dataset.name.startswith("qcd") or dataset.name.startswith("qqHH_"):
            dataset.x.skip_scale = True
            dataset.x.skip_pdf = True
            dataset.add_tag("skip_scale")
            dataset.add_tag("skip_pdf")

        if dataset.has_tag("is_hbw") and "custom" in dataset.name:
            # No PDF weights and 6 scale weights in custom HH samples
            dataset.x.skip_scale = True
            dataset.x.skip_pdf = True
            dataset.add_tag("skip_scale")
            dataset.add_tag("skip_pdf")

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
    else:  # 2018
        cfg.x.luminosity = Number(59830, {
            "lumi_13TeV_2017": 0.015j,
            "lumi_13TeV_1718": 0.002j,
            "lumi_13TeV_correlated": 0.02j,
        })

    # minimum bias cross section in mb (milli) for creating PU weights, values from
    # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData?rev=45#Recommended_cross_section
    cfg.x.minbias_xs = Number(69.2, 0.046j)

    # whether to validate the number of obtained LFNs in GetDatasetLFNs
    cfg.x.validate_dataset_lfns = limit_dataset_files is None

    # jec configuration
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=201
    jerc_postfix = "APV" if year == 2016 and campaign.x.vfp == "post" else ""
    cfg.x.jec = DotDict.wrap({
        "campaign": f"Summer19UL{year2}{jerc_postfix}",
        "version": {2016: "V7", 2017: "V5", 2018: "V5"}[year],
        "jet_type": "AK4PFchs",
        "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
        "levels_for_type1_met": ["L1FastJet"],
        "uncertainty_sources": [
            # "AbsoluteStat",
            # "AbsoluteScale",
            # "AbsoluteSample",
            # "AbsoluteFlavMap",
            # "AbsoluteMPFBias",
            # "Fragmentation",
            # "SinglePionECAL",
            # "SinglePionHCAL",
            # "FlavorQCD",
            # "TimePtEta",
            # "RelativeJEREC1",
            # "RelativeJEREC2",
            # "RelativeJERHF",
            # "RelativePtBB",
            # "RelativePtEC1",
            # "RelativePtEC2",
            # "RelativePtHF",
            # "RelativeBal",
            # "RelativeSample",
            # "RelativeFSR",
            # "RelativeStatFSR",
            # "RelativeStatEC",
            # "RelativeStatHF",
            # "PileUpDataMC",
            # "PileUpPtRef",
            # "PileUpPtBB",
            # "PileUpPtEC1",
            # "PileUpPtEC2",
            # "PileUpPtHF",
            # "PileUpMuZero",
            # "PileUpEnvelope",
            # "SubTotalPileUp",
            # "SubTotalRelative",
            # "SubTotalPt",
            # "SubTotalScale",
            # "SubTotalAbsolute",
            # "SubTotalMC",
            "Total",
            # "TotalNoFlavor",
            # "TotalNoTime",
            # "TotalNoFlavorNoTime",
            # "FlavorZJet",
            # "FlavorPhotonJet",
            # "FlavorPureGluon",
            # "FlavorPureQuark",
            # "FlavorPureCharm",
            # "FlavorPureBottom",
            # "TimeRunA",
            # "TimeRunB",
            # "TimeRunC",
            # "TimeRunD",
            "CorrelationGroupMPFInSitu",
            "CorrelationGroupIntercalibration",
            "CorrelationGroupbJES",
            "CorrelationGroupFlavor",
            "CorrelationGroupUncorrelated",
        ],
    })

    # JER
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107
    cfg.x.jer = DotDict.wrap({
        "campaign": f"Summer19UL{year2}{jerc_postfix}",
        "version": "JR" + {2016: "V3", 2017: "V2", 2018: "V2"}[year],
        "jet_type": "AK4PFchs",
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
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16preVFP?rev=6
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16postVFP?rev=8
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=15
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=17
    btag_key = f"2016{campaign.x.vfp}" if year == 2016 else year
    cfg.x.btag_working_points = DotDict.wrap({
        "deepjet": {
            "loose": {"2016pre": 0.0508, "2016post": 0.0480, 2017: 0.0532, 2018: 0.0490}[btag_key],
            "medium": {"2016pre": 0.2598, "2016post": 0.2489, 2017: 0.3040, 2018: 0.2783}[btag_key],
            "tight": {"2016pre": 0.6502, "2016post": 0.6377, 2017: 0.7476, 2018: 0.7100}[btag_key],
        },
        "deepcsv": {
            "loose": {"2016pre": 0.2027, "2016post": 0.1918, 2017: 0.1355, 2018: 0.1208}[btag_key],
            "medium": {"2016pre": 0.6001, "2016post": 0.5847, 2017: 0.4506, 2018: 0.4168}[btag_key],
            "tight": {"2016pre": 0.8819, "2016post": 0.8767, 2017: 0.7738, 2018: 0.7665}[btag_key],
        },
    })

    # TODO: check e/mu/btag corrections and implement
    # btag weight configuration
    cfg.x.btag_sf = ("deepJet_shape", cfg.x.btag_sf_jec_sources)

    # names of electron correction sets and working points
    # (used in the electron_sf producer)
    cfg.x.electron_sf_names = ("UL-Electron-ID-SF", f"{year}{corr_postfix}", "wp80iso")

    # names of muon correction sets and working points
    # (used in the muon producer)
    cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", f"{year}{corr_postfix}_UL")

    # helper to add column aliases for both shifts of a source
    # TODO: switch to the columnflow function (but what happened to *selection_dependent*?)
    def add_shift_aliases(shift_source: str, aliases: dict[str], selection_dependent: bool):

        for direction in ["up", "down"]:
            shift = cfg.get_shift(od.Shift.join_name(shift_source, direction))
            # format keys and values
            inject_shift = lambda s: re.sub(r"\{([^_])", r"{_\1", s).format(**shift.__dict__)
            _aliases = {inject_shift(key): inject_shift(value) for key, value in aliases.items()}
            alias_type = "column_aliases_selection_dependent" if selection_dependent else "column_aliases"
            # extend existing or register new column aliases
            shift.set_aux(alias_type, shift.get_aux(alias_type, {})).update(_aliases)

    # register shifts
    # TODO: make shifts year-dependent
    cfg.add_shift(name="nominal", id=0)
    cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="minbias_xs_up", id=7, type="shape")
    cfg.add_shift(name="minbias_xs_down", id=8, type="shape")
    add_shift_aliases(
        "minbias_xs",
        {
            "pu_weight": "pu_weight_{name}",
            "normalized_pu_weight": "normalized_pu_weight_{name}",
        },
        selection_dependent=False)
    cfg.add_shift(name="top_pt_up", id=9, type="shape")
    cfg.add_shift(name="top_pt_down", id=10, type="shape")
    add_shift_aliases("top_pt", {"top_pt_weight": "top_pt_weight_{direction}"}, selection_dependent=False)

    cfg.add_shift(name="e_sf_up", id=40, type="shape")
    cfg.add_shift(name="e_sf_down", id=41, type="shape")
    cfg.add_shift(name="e_trig_sf_up", id=42, type="shape")
    cfg.add_shift(name="e_trig_sf_down", id=43, type="shape")
    add_shift_aliases("e_sf", {"electron_weight": "electron_weight_{direction}"}, selection_dependent=False)

    cfg.add_shift(name="mu_sf_up", id=50, type="shape")
    cfg.add_shift(name="mu_sf_down", id=51, type="shape")
    cfg.add_shift(name="mu_trig_sf_up", id=52, type="shape")
    cfg.add_shift(name="mu_trig_sf_down", id=53, type="shape")
    add_shift_aliases("mu_sf", {"muon_weight": "muon_weight_{direction}"}, selection_dependent=False)

    btag_uncs = [
        "hf", "lf", f"hfstats1_{year}", f"hfstats2_{year}",
        f"lfstats1_{year}", f"lfstats2_{year}", "cferr1", "cferr2",
    ]
    for i, unc in enumerate(btag_uncs):
        cfg.add_shift(name=f"btag_{unc}_up", id=100 + 2 * i, type="shape")
        cfg.add_shift(name=f"btag_{unc}_down", id=101 + 2 * i, type="shape")
        add_shift_aliases(
            f"btag_{unc}",
            {
                "normalized_btag_weight": f"normalized_btag_weight_{unc}_" + "{direction}",
                "normalized_njet_btag_weight": f"normalized_njet_btag_weight_{unc}_" + "{direction}",
            },
            selection_dependent=False,
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
        # add_shift_aliases(unc, {f"{unc}_weight": f"{unc}_weight_" + "{direction}"}, selection_dependent=False)
        add_shift_aliases(
            unc,
            {f"normalized_{unc}_weight": f"normalized_{unc}_weight_" + "{direction}"},
            selection_dependent=False,
        )

    with open(os.path.join(thisdir, "jec_sources.yaml"), "r") as f:
        all_jec_sources = yaml.load(f, yaml.Loader)["names"]
    for jec_source in cfg.x.jec["uncertainty_sources"]:
        idx = all_jec_sources.index(jec_source)
        cfg.add_shift(name=f"jec_{jec_source}_up", id=5000 + 2 * idx, type="shape")
        cfg.add_shift(name=f"jec_{jec_source}_down", id=5001 + 2 * idx, type="shape")
        add_shift_aliases(
            f"jec_{jec_source}",
            {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"},
            selection_dependent=True,
        )

    cfg.add_shift(name="jer_up", id=6000, type="shape", tags={"selection_dependent"})
    cfg.add_shift(name="jer_down", id=6001, type="shape", tags={"selection_dependent"})
    add_shift_aliases("jer", {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"}, selection_dependent=True)

    def make_jme_filename(jme_aux, sample_type, name, era=None):
        """
        Convenience function to compute paths to JEC files.
        """
        # normalize and validate sample type
        sample_type = sample_type.upper()
        if sample_type not in ("DATA", "MC"):
            raise ValueError(f"invalid sample type '{sample_type}', expected either 'DATA' or 'MC'")

        jme_full_version = "_".join(s for s in (jme_aux.campaign, era, jme_aux.version, sample_type) if s)

        return f"{jme_aux.source}/{jme_full_version}/{jme_full_version}_{name}_{jme_aux.jet_type}.txt"

    # external files
    json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-dfd90038"
    cfg.x.external_files = DotDict.wrap({
        # jet energy correction
        "jet_jerc": (f"{json_mirror}/POG/JME/{year}{corr_postfix}_UL/jet_jerc.json.gz", "v1"),

        # electron scale factors
        "electron_sf": (f"{json_mirror}/POG/EGM/{year}{corr_postfix}_UL/electron.json.gz", "v1"),

        # muon scale factors
        "muon_sf": (f"{json_mirror}/POG/MUO/{year}{corr_postfix}_UL/muon_Z.json.gz", "v1"),

        # btag scale factor
        "btag_sf_corr": (f"{json_mirror}/POG/BTV/{year}{corr_postfix}_UL/btagging.json.gz", "v1"),

        # met phi corrector
        "met_phi_corr": (f"{json_mirror}/POG/JME/{year}{corr_postfix}_UL/met.json.gz", "v1"),
    })

    # external files with more complex year dependence
    # TODO: generalize to different years
    if year != 2017:
        raise NotImplementedError("TODO: generalize external files to different years than 2017")

    cfg.x.external_files.update(DotDict.wrap({
        # files from TODO
        "lumi": {
            "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
            "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
        },

        # files from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=44#Pileup_JSON_Files_For_Run_II # noqa
        "pu": {
            "json": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/pileup_latest.txt", "v1"),  # noqa
            "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/435f0b04c0e318c1036a6b95eb169181bbbe8344/SimGeneral/MixingModule/python/mix_2017_25ns_UltraLegacy_PoissonOOTPU_cfi.py", "v1"),  # noqa
            "data_profile": {
                "nominal": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root", "v1"),  # noqa
                "minbias_xs_up": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-72400ub-99bins.root", "v1"),  # noqa
                "minbias_xs_down": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-66000ub-99bins.root", "v1"),  # noqa
            },
        },
    }))

    # columns to keep after certain steps
    # TODO: selector-dependent columns should either use the is_sl / is_dl tag or
    #       be implemented as part of the selectors itself (e.g. only SL needs the Lightjet column)
    cfg.x.keep_columns = DotDict.wrap({
        "cf.MergeSelectionMasks": {
            "mc_weight", "normalization_weight", "process_id", "category_ids", "cutflow.*",
            "HbbJet.n_subjets", "HbbJet.n_separated_jets", "HbbJet.max_dr_ak4",
        },
    })

    cfg.x.keep_columns["cf.ReduceEvents"] = (
        {
            # general event information
            "run", "luminosityBlock", "event",
            # columns added during selection, required in general
            "mc_weight", "PV.npvs", "process_id", "category_ids", "deterministic_seed",
            # weight-related columns
            "pu_weight*", "pdf_weight*",
            "murf_envelope_weight*", "mur_weight*", "muf_weight*",
            "btag_weight*",
        } | four_vec(  # Jets
            {"Jet", "Bjet", "VBFJet"},
            {"btagDeepFlavB", "hadronFlavour"},
        ) | four_vec(  # FatJets
            {"FatJet", "HbbJet"},
            {
                "msoftdrop", "tau1", "tau2", "tau3",
                "btagHbb", "deepTagMD_HbbvsQCD", "particleNet_HbbvsQCD",
            },
        ) | four_vec(  # Leptons
            {"Electron", "Muon"},
            {"charge", "pdgId"},
        ) | {"Electron.deltaEtaSC", "MET.pt", "MET.phi"}
    )

    # event weight columns as keys in an ordered dict, mapped to shift instances they depend on
    get_shifts = lambda *keys: sum(([cfg.get_shift(f"{k}_up"), cfg.get_shift(f"{k}_down")] for k in keys), [])
    cfg.x.event_weights = DotDict()

    cfg.x.event_weights["normalization_weight"] = []

    # for dataset in cfg.datasets:
    #     if dataset.x("is_ttbar", False):
    #         dataset.x.event_weights = {"top_pt_weight": get_shifts("top_pt")}

    # NOTE: which to use, njet_btag_weight or btag_weight?
    cfg.x.event_weights["normalized_btag_weight"] = get_shifts(*(f"btag_{unc}" for unc in btag_uncs))
    cfg.x.event_weights["normalized_pu_weight"] = get_shifts("minbias_xs")
    cfg.x.event_weights["electron_weight"] = get_shifts("e_sf")
    cfg.x.event_weights["muon_weight"] = get_shifts("mu_sf")

    for dataset in cfg.datasets:
        dataset.x.event_weights = DotDict()
        if not dataset.has_tag("skip_scale"):
            # pdf/scale weights for all non-qcd datasets
            dataset.x.event_weights["normalized_murf_envelope_weight"] = get_shifts("murf_envelope")
            dataset.x.event_weights["normalized_mur_weight"] = get_shifts("mur")
            dataset.x.event_weights["normalized_muf_weight"] = get_shifts("muf")

        if not dataset.has_tag("skip_pdf"):
            dataset.x.event_weights["normalized_pdf_weight"] = get_shifts("pdf")

    def reduce_version(cls, inst, params):
        # per default, use the version set on the command line
        version = inst.version  # same as params.get("version") ?

        if params.get("selector") == "sl_v1":
            # use a fixed version for the sl_v1 selector (NOTE: does not yet exist)
            version = "sl_v1"

        return version

    # Version of required tasks
    cfg.x.versions = {
        "cf.CalibrateEvents": "common1",
        "cf.SelectEvents": reduce_version,
        "cf.MergeSelectionStats": reduce_version,
        "cf.MergeSelectionMasks": reduce_version,
        "cf.ReduceEvents": reduce_version,
        "cf.MergeReductionStats": reduce_version,
        "cf.MergeReducedEvents": reduce_version,
    }

    # add categories
    add_categories_selection(cfg)

    # add variables
    add_variables(cfg)

    # set some config defaults and groups
    # TODO: it might make sense to completely separate this for SL/DL
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
    if cfg.has_tag("is_resonant"):
        # from hbw.config.resonant import configure_resonant
        configure_resonant(cfg)

    return cfg


def configure_resonant(config: od.Config):
    # TODO?
    return
