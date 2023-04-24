# coding: utf-8

"""
Configuration of the 2017 HH -> bbWW analysis.
"""

from __future__ import annotations

import os
import re
from typing import Set

import yaml
from scinum import Number
import order as od

from columnflow.util import DotDict
from columnflow.config_util import get_root_processes_from_campaign
from hbw.config.styling import stylize_processes
from hbw.config.categories import add_categories_selection
from hbw.config.variables import add_variables

from hbw.config.analysis_hbw import analysis_hbw

thisdir = os.path.dirname(os.path.abspath(__file__))


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

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # create a config by passing the campaign, so id and name will be identical
    cfg = analysis_hbw.add_config(campaign, name=config_name, id=config_id)

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
    cfg.add_process(procs.n.hh_ggf_bbtautau)

    cfg.get_process("qcd_mu").label = "QCD Muon enriched"
    qcd_ele = cfg.add_process(
        name="qcd_ele",
        id=31199,
        xsecs={13: Number(0.1)},  # TODO
        label="QCD Electron enriched",
    )
    qcd_ele.add_process(cfg.get_process("qcd_em"))
    qcd_ele.add_process(cfg.get_process("qcd_bctoe"))

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
        "st_tchannel_tbar_powheg",
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
        # Signal
        "ggHH_kl_0_kt_1_sl_hbbhww_powheg",
        "ggHH_kl_1_kt_1_sl_hbbhww_powheg",
        "ggHH_kl_2p45_kt_1_sl_hbbhww_powheg",
        "ggHH_kl_5_kt_1_sl_hbbhww_powheg",
        "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww_madgraph",
        "qqHH_CV_1_C2V_1_kl_0_sl_hbbhww_madgraph",
        "qqHH_CV_1_C2V_1_kl_2_sl_hbbhww_madgraph",
        "qqHH_CV_1_C2V_0_kl_1_sl_hbbhww_madgraph",
        "qqHH_CV_1_C2V_2_kl_1_sl_hbbhww_madgraph",
        "qqHH_CV_0p5_C2V_1_kl_1_sl_hbbhww_madgraph",
        "qqHH_CV_1p5_C2V_1_kl_1_sl_hbbhww_madgraph",
        # HH(bbtautau)
        "hh_ggf_bbtautau_madgraph",
    ]
    for dataset_name in dataset_names:
        dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

        if limit_dataset_files:
            # apply optional limit on the max. number of files per dataset
            for info in dataset.info.values():
                if info.n_files > limit_dataset_files:
                    info.n_files = limit_dataset_files

        # add aux info to datasets
        if dataset.name.startswith(("st", "tt")):
            dataset.x.has_top = True
        if dataset.name.startswith("tt"):
            dataset.x.is_ttbar = True
        if dataset.name.startswith("qcd"):
            dataset.x.is_qcd = True
        if "HH" in dataset.name and "hbbhww" in dataset.name:
            dataset.x.is_hbw = True

    # default calibrator, selector, producer, ml model and inference model
    cfg.x.default_calibrator = "skip_jecunc"
    cfg.x.default_selector = "default"
    cfg.x.default_producer = "ml_inputs"
    # cfg.x.default_ml_model = "default"
    cfg.x.default_ml_model = None
    cfg.x.default_inference_model = "default"
    cfg.x.default_categories = ["incl"]

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    cfg.x.process_groups = {
        "all": ["*"],
        "default": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "st", "w_lnu", "dy_lep"],
        "with_qcd": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "qcd", "st", "w_lnu", "dy_lep"],
        "much": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "qcd_mu", "st", "w_lnu", "dy_lep"],
        "ech": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "qcd_ele", "st", "w_lnu", "dy_lep"],
        "inference": ["ggHH_*", "tt", "st", "w_lnu", "dy_lep", "qcd_*"],
        "ml": ["ggHH_kl_1_*", "tt", "st", "w_lnu", "dy_lep"],
        "ml_test": ["ggHH_kl_1_*", "st", "w_lnu"],
        "test": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt_sl"],
        "small": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "st"],
        "bkg": ["tt", "st", "w_lnu", "dy_lep"],
        "signal": ["ggHH_*", "qqHH"], "gghh": ["ggHH_*"], "qqhh": ["qqHH_*"],
    }
    cfg.x.process_groups["dmuch"] = ["data_mu"] + cfg.x.process_groups["much"]
    cfg.x.process_groups["dech"] = ["data_e"] + cfg.x.process_groups["ech"]

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    cfg.x.dataset_groups = {
        "all": ["*"],
        "default": ["ggHH_kl_1*", "tt_*", "qcd_*", "st_*", "dy_*", "w_lnu_*"],
        "inference": ["ggHH_*", "tt_*", "qcd_*", "st_*", "dy_*", "w_lnu_*"],
        "test": ["ggHH_kl_1*", "tt_sl_powheg"],
        "small": ["ggHH_kl_1*", "tt_*", "st_*"],
        "bkg": ["tt_*", "st_*", "w_lnu_*", "dy_*"],
        "tt": ["tt_*"], "st": ["st_*"], "w": ["w_lnu_*"], "dy": ["dy_*"],
        "qcd": ["qcd_*"], "qcd_mu": ["qcd_mu*"], "qcd_ele": ["qcd_em*", "qcd_bctoe*"],
        "signal": ["ggHH_*", "qqHH_*"], "gghh": ["ggHH_*"], "qqhh": ["qqHH_*"],
        "ml": ["ggHH_kl_1*", "tt_*", "st_*", "dy_*", "w_lnu_*"],
    }

    # category groups for conveniently looping over certain categories
    # (used during plotting)
    cfg.x.category_groups = {
        "much": ["1mu", "1mu__resolved", "1mu__boosted"],
        "ech": ["1e", "1e__resolved", "1e__boosted"],
        "default": ["incl", "1e", "1mu"],
        "test": ["incl", "1e"],
    }

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    cfg.x.variable_groups = {
        "default": ["n_jet", "n_muon", "n_electron", "ht", "m_bb", "deltaR_bb", "jet1_pt"],  # n_deepjet, ....
        "test": ["n_jet", "n_electron", "jet1_pt"],
        "cutflow": ["cf_jet1_pt", "cf_jet4_pt", "cf_n_jet", "cf_n_electron", "cf_n_muon"],  # cf_n_deepjet
    }

    # shift groups for conveniently looping over certain shifts
    # (used during plotting)
    cfg.x.shift_groups = {
        "jer": ["nominal", "jer_up", "jer_down"],
    }

    # selector step groups for conveniently looping over certain steps
    # (used in cutflow tasks)
    cfg.x.selector_step_groups = {
        "resolved": ["Trigger", "Lepton", "VetoLepton", "Jet", "Bjet", "VetoTau"],
        "boosted": ["Trigger", "Lepton", "VetoLepton", "FatJet", "Boosted"],
        "default": ["Lepton", "VetoLepton", "Jet", "Bjet", "Trigger"],
        "thesis": ["Lepton", "Muon", "Jet", "Trigger", "Bjet"],  # reproduce master thesis cuts for checks
        "test": ["Lepton", "Jet", "Bjet"],
    }

    cfg.x.selector_step_labels = {
        "Jet": r"$N_{jets}^{AK4} \geq 3$",
        "Lepton": r"$N_{lepton} \geq 1$",
        "VetoLepton": r"$N_{lepton}^{veto} \leq 1$",
        "Bjet": r"$N_{jets}^{BTag} \geq 1$",
        "FatJet": r"$N_{H \rightarrow bb}^{AK8} \geq 1$",
        "Boosted": r"$N_{jets}^{AK4} \geq 1$",
    }

    # plotting settings groups
    cfg.x.general_settings_groups = {
        "test1": {"p1": True, "p2": 5, "p3": "text", "skip_legend": True},
        "default_norm": {"shape_norm": True, "yscale": "log"},
    }
    cfg.x.process_settings_groups = {
        "default": {"ggHH_kl_1_kt_1_sl_hbbhww": {"scale": 2000, "unstack": True}},
        "unstack_all": {proc.name: {"unstack": True} for proc in cfg.processes},
        "unstack_signal": {proc.name: {"unstack": True} for proc in cfg.processes if "HH" in proc.name},
        "scale_signal": {
            proc.name: {"unstack": True, "scale": 10000}
            for proc in cfg.processes if "HH" in proc.name
        },
    }
    # when drawing DY as a line, use a different type of yellow
    cfg.x.process_settings_groups["unstack_all"].update({"dy_lep": {"unstack": True, "color": "#e6d800"}})

    cfg.x.variable_settings_groups = {
        "test": {
            "mli_mbb": {"rebin": 2, "label": "test"},
            "mli_mjj": {"rebin": 2},
        },
    }

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
    def add_aliases(shift_source: str, aliases: Set[str], selection_dependent: bool):

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
    add_aliases("minbias_xs", {"pu_weight": "pu_weight_{name}"}, selection_dependent=False)
    cfg.add_shift(name="top_pt_up", id=9, type="shape")
    cfg.add_shift(name="top_pt_down", id=10, type="shape")
    add_aliases("top_pt", {"top_pt_weight": "top_pt_weight_{direction}"}, selection_dependent=False)

    cfg.add_shift(name="e_sf_up", id=40, type="shape")
    cfg.add_shift(name="e_sf_down", id=41, type="shape")
    cfg.add_shift(name="e_trig_sf_up", id=42, type="shape")
    cfg.add_shift(name="e_trig_sf_down", id=43, type="shape")
    add_aliases("e_sf", {"electron_weight": "electron_weight_{direction}"}, selection_dependent=False)

    cfg.add_shift(name="mu_sf_up", id=50, type="shape")
    cfg.add_shift(name="mu_sf_down", id=51, type="shape")
    cfg.add_shift(name="mu_trig_sf_up", id=52, type="shape")
    cfg.add_shift(name="mu_trig_sf_down", id=53, type="shape")
    add_aliases("mu_sf", {"muon_weight": "muon_weight_{direction}"}, selection_dependent=False)

    btag_uncs = [
        "hf", "lf", f"hfstats1_{year}", f"hfstats2_{year}",
        f"lfstats1_{year}", f"lfstats2_{year}", "cferr1", "cferr2",
    ]
    for i, unc in enumerate(btag_uncs):
        cfg.add_shift(name=f"btag_{unc}_up", id=100 + 2 * i, type="shape")
        cfg.add_shift(name=f"btag_{unc}_down", id=101 + 2 * i, type="shape")
        add_aliases(
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
        # add_aliases(unc, {f"{unc}_weight": f"{unc}_weight_" + "{direction}"}, selection_dependent=False)
        add_aliases(
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
        add_aliases(
            f"jec_{jec_source}",
            {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"},
            selection_dependent=True,
        )

    cfg.add_shift(name="jer_up", id=6000, type="shape", tags={"selection_dependent"})
    cfg.add_shift(name="jer_down", id=6001, type="shape", tags={"selection_dependent"})
    add_aliases("jer", {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"}, selection_dependent=True)

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

        # hh-btag repository (lightweight) with TF saved model directories
        "hh_btag_repo": ("https://github.com/hh-italian-group/HHbtag/archive/1dc426053418e1cab2aec021802faf31ddf3c5cd.tar.gz", "v1"),  # noqa
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
    cfg.x.keep_columns = DotDict.wrap({
        "cf.SelectEvents": {"mc_weight"},
        "cf.MergeSelectionMasks": {
            "mc_weight", "normalization_weight", "process_id", "category_ids", "cutflow.*",
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
        } | set(  # Jets
            f"{jet_obj}.{field}"
            for jet_obj in ["Jet", "Bjet", "Lightjet", "VBFJet"]
            # NOTE: if we run into storage troubles, skip Bjet and Lightjet
            for field in ["pt", "eta", "phi", "mass", "btagDeepFlavB", "hadronFlavour"]
        ) | set(  # H->bb FatJet
            f"{jet_type}.{field}"
            for jet_type in ["FatJet", "HbbJet"]
            for field in [
                "pt", "eta", "phi", "mass", "msoftdrop", "tau1", "tau2", "tau3",
                "btagHbb", "deepTagMD_HbbvsQCD", "particleNet_HbbvsQCD",
            ]
        ) | set(  # Leptons
            f"{lep}.{field}"
            for lep in ["Electron", "Muon"]
            for field in ["pt", "eta", "phi", "mass", "charge", "pdgId"]
        ) | {  # Electrons
            "Electron.deltaEtaSC",  # for SF calculation
        } | set(  # MET
            f"MET.{field}"
            for field in ["pt", "phi"]
        )
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
    # TODO: fix pu_weight; takes way too large values (from 0 to 160)
    # cfg.x.event_weights["normalized_pu_weight"] = get_shifts("minbias_xs")
    for dataset in cfg.datasets:
        dataset.x.event_weights = DotDict()
        if not dataset.x("is_qcd", False):
            # pdf/scale weights for all non-qcd datasets
            dataset.x.event_weights["normalized_murf_envelope_weight"] = get_shifts("murf_envelope")
            dataset.x.event_weights["normalized_mur_weight"] = get_shifts("mur")
            dataset.x.event_weights["normalized_muf_weight"] = get_shifts("muf")
            dataset.x.event_weights["normalized_pdf_weight"] = get_shifts("pdf")

    # versions per task family and optionally also dataset and shift
    # None can be used as a key to define a default value
    cfg.x.versions = {
        # None: "dev1",
        # "cf.SelectEvents": "dev1",
    }

    # add categories
    add_categories_selection(cfg)

    # add variables
    add_variables(cfg)

    # NOTE: this is only needed here since adding variable insts in producer inits does not work
    #       when submitting jobs; remove the following block as soon as this is fixed.
    #  add_ml_variables(cfg)
    # cfg.x.add_ml_variables = False
    # add_feature_variables(cfg)
    # cfg.x.add_feature_variables = False

    # only produce cutflow features when number of dataset_files is limited (used in selection module)
    cfg.x.do_cutflow_features = bool(limit_dataset_files) and limit_dataset_files <= 10

    return cfg
