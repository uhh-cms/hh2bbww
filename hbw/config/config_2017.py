# coding: utf-8

"""
Configuration of the 2017 HH -> bbWW analysis.
"""

import os
import re
from typing import Set

import yaml
from scinum import Number, REL
import order as od
import cmsdb
import cmsdb.campaigns.run2_2017_nano_v9

from columnflow.util import DotDict, get_root_processes_from_campaign
from hbw.config.categories import add_categories
from hbw.config.variables import add_variables
from hbw.config.ml_variables import add_ml_variables
from hbw.config.cutflow_variables import add_cutflow_variables, add_gen_variables

from hbw.config.analysis_hbw import analysis_hbw

thisdir = os.path.dirname(os.path.abspath(__file__))

#
# 2017 standard config
#

# copy the campaign, which in turn copies datasets and processes
campaign_run2_2017 = cmsdb.campaigns.run2_2017_nano_v9.campaign_run2_2017_nano_v9.copy()

# get all root processes
procs = get_root_processes_from_campaign(campaign_run2_2017)

# create a config by passing the campaign, so id and name will be identical
config_2017 = cfg = analysis_hbw.add_config(campaign_run2_2017)

# add processes we are interested in
cfg.add_process(procs.n.data)
cfg.add_process(procs.n.tt)
cfg.add_process(procs.n.st)
cfg.add_process(procs.n.w_lnu)
cfg.add_process(procs.n.dy_lep)
# cfg.add_process(procs.n.qcd)
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

# set color of some processes
colors = {
    "data": "#000000",  # black
    "tt": "#e41a1c",  # red
    "qcd": "#377eb8",  # blue
    "w_lnu": "#4daf4a",  # green
    "higgs": "#984ea3",  # purple
    "st": "#ff7f00",  # orange
    "dy_lep": "#ffff33",  # yellow
    "ttV": "#a65628",  # brown
    "VV": "#f781bf",  # pink
    "other": "#999999",  # grey
    "ggHH_kl_1_kt_1_sl_hbbhww": "#000000",  # black
    "ggHH_kl_0_kt_1_sl_hbbhww": "#1b9e77",  # green2
    "ggHH_kl_2p45_kt_1_sl_hbbhww": "#d95f02",  # orange2
    "ggHH_kl_5_kt_1_sl_hbbhww": "#e7298a",  # pink2
    "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww": "#e41a1c",  # red
    "qqHH_CV_1_C2V_1_kl_0_sl_hbbhww": "#377eb8",  # blue
    "qqHH_CV_1_C2V_1_kl_2_sl_hbbhww": "#4daf4a",  # green
    "qqHH_CV_1_C2V_0_kl_1_sl_hbbhww": "#984ea3",  # purple
    "qqHH_CV_1_C2V_2_kl_1_sl_hbbhww": "#ff7f00",  # orange
    "qqHH_CV_0p5_C2V_1_kl_1_sl_hbbhww": "#a65628",  # brown
    "qqHH_CV_1p5_C2V_1_kl_1_sl_hbbhww": "#f781bf",  # pink
}
for proc, color in colors.items():
    if proc in cfg.processes:
        cfg.get_process(proc).color1 = color

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
    "st_schannel_had_amcatnlo",
    # WJets (TODO: fix wjet datasets)
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
    # QCD (msc thesis samples not implemented)
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
]
for dataset_name in dataset_names:
    dataset = cfg.add_dataset(campaign_run2_2017.get_dataset(dataset_name))

    # reduce n_files to max. 2 for testing purposes (TODO switch to full dataset)
    for k in dataset.info.keys():
        # if dataset.name == "ggHH_kl_1_kt_1_sl_hbbhww_powheg":
        #    continue
        if dataset[k].n_files > 2:
            dataset[k].n_files = 2

    # add aux info to datasets
    if dataset.name.startswith(("st", "tt")):
        dataset.x.has_top = True
    if dataset.name.startswith("tt"):
        dataset.x.is_ttbar = True
    if "HH" in dataset.name and "hbbhww" in dataset.name:
        dataset.x.is_hbw = True


# default calibrator, selector, producer, ml model and inference model
cfg.set_aux("default_calibrator", "skip_jecunc")
cfg.set_aux("default_selector", "default")
cfg.set_aux("default_producer", "features")
cfg.set_aux("default_ml_model", None)
cfg.set_aux("default_inference_model", "default")
cfg.set_aux("default_categories", ["incl"])

# process groups for conveniently looping over certain processs
# (used in wrapper_factory and during plotting)
cfg.set_aux("process_groups", {
    "hh": ["ggHH_kl_1_kt_1_sl_hbbhww"],
    "default": ["ggHH_kl_1_kt_1_sl_hbbhww", "dy_lep", "w_lnu", "st", "tt"],
    "working": ["ggHH_kl_1_kt_1_sl_hbbhww", "dy_lep", "st", "tt"],
    "test": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt_sl"],
    "small": ["ggHH_kl_1_kt_1_sl_hbbhww", "st", "tt"],
    "signal": ["ggHH_*"],
    "bkg": ["tt", "st", "w_lnu", "dy_lep"],
})

# dataset groups for conveniently looping over certain datasets
# (used in wrapper_factory and during plotting)
cfg.set_aux("dataset_groups", {
    "all": ["*"],
    "working": ["tt_*", "st_*", "dy_*"],
    "small": ["ggHH_*", "tt_*", "st_*"],
    "default": ["ggHH_*", "tt_*", "st_*", "dy_*", "w_lnu_*"],
    "tt": ["tt_*"], "st": ["st_*"], "w": ["w_lnu_*"], "dy": ["dy_*"],
    "hh": ["ggHH_*"], "hhsm": ["ggHH_kl_1_kt_1_sl_hbbhww_powheg"],
})

# category groups for conveniently looping over certain categories
# (used during plotting)
cfg.set_aux("category_groups", {
    "default": ["incl", "1e", "1mu"],
    "test": ["incl", "1e"],
})

# variable groups for conveniently looping over certain variables
# (used during plotting)
cfg.set_aux("variable_groups", {
    "default": ["n_jet", "n_muon", "n_electron", "ht", "m_bb", "deltaR_bb", "jet1_pt"],  # n_deepjet, ....
    "test": ["n_jet", "n_electron", "jet1_pt"],
    "cutflow": ["cf_jet1_pt", "cf_jet4_pt", "cf_n_jet", "cf_n_electron", "cf_n_muon"],  # cf_n_deepjet
})

# shift groups for conveniently looping over certain shifts
# (used during plotting)
cfg.set_aux("shift_groups", {
    "jer": ["nominal", "jer_up", "jer_down"],
})

# selector step groups for conveniently looping over certain steps
# (used in cutflow tasks)
cfg.set_aux("selector_step_groups", {
    "default": ["Lepton", "VetoLepton", "Jet", "Bjet", "Trigger"],
    "thesis": ["Lepton", "Muon", "Jet", "Trigger", "Bjet"],  # reproduce master thesis cuts to check if everything works
    "test": ["Lepton", "Jet", "Bjet"],
})

cfg.set_aux("selector_step_labels", {
    "Jet": r"$N_{Jets} \geq 3$",
    "Lepton": r"$N_{Lepton} = 1$",
    "Bjet": r"$N_{Jets}^{BTag} \geq 1$",
})

# plotting settings groups
cfg.x.general_settings_groups = {
    "test1": {"p1": True, "p2": 5, "p3": "text", "skip_legend": True},
    "default_norm": {"shape_norm": True, "yscale": "log"},
}
cfg.x.process_settings_groups = {
    "default": [["ggHH_kl_1_kt_1_sl_hbbhww", "scale=2000", "unstack"]],
    "unstack_all": [[proc.name, "unstack"] for proc in cfg.processes],
    "unstack_signal": {proc.name: {"unstack": True} for proc in cfg.processes if "HH" in proc.name},
}
cfg.x.variable_settings_groups = {
    "test": {
        "mli_mbb": {"rebin": 2, "label": "test"},
        "mli_mjj": {"rebin": 2},
    },
}

# 2017 luminosity with values in inverse pb and uncertainties taken from
# https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM?rev=176#LumiComb
cfg.set_aux("luminosity", Number(41480, {
    "lumi_13TeV_2017": (REL, 0.02),
    "lumi_13TeV_1718": (REL, 0.006),
    "lumi_13TeV_correlated": (REL, 0.009),
}))

# 2017 minimum bias cross section in mb (milli) for creating PU weights, values from
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=44#Pileup_JSON_Files_For_Run_II
cfg.set_aux("minbiasxs", Number(69.2, (REL, 0.046)))

# 2017 b-tag working points
# https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=15
cfg.x.btag_working_points = DotDict.wrap({
    "deepjet": {
        "loose": 0.0532,
        "medium": 0.3040,
        "tight": 0.7476,
    },
    "deepcsv": {
        "loose": 0.1355,
        "medium": 0.4506,
        "tight": 0.7738,
    },
})

# TODO: check e/mu/btag corrections and implement
# name of the btag_sf correction set
cfg.x.btag_sf_correction_set = "deepJet_shape"

# names of electron correction sets and working points
# (used in the electron_sf producer)
cfg.x.electron_sf_names = ("UL-Electron-ID-SF", "2017", "wp80iso")

# names of muon correction sets and working points
# (used in the muon producer)
cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", "2017_UL")

# jec configuration
cfg.x.jec = DotDict.wrap({
    "campaign": "Summer19UL17",
    "version": "V5",
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

cfg.x.jer = DotDict.wrap({
    "campaign": "Summer19UL17",
    "version": "JRV2",
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
    "Absolute_2017",
    "BBEC1",
    "BBEC1_2017",
    "EC2",
    "EC2_2017",
    "FlavorQCD",
    "Fragmentation",
    "HF",
    "HF_2017",
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
    "RelativeSample_2017",
    "RelativeStatEC",
    "RelativeStatFSR",
    "RelativeStatHF",
    "SinglePionECAL",
    "SinglePionHCAL",
    "TimePtEta",
]


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
    "hf", "lf", "hfstats1_2017", "hfstats2_2017", "lfstats1_2017", "lfstats2_2017", "cferr1",
    "cferr2",
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
cfg.add_shift(name="scale_up", id=205, type="shape")
cfg.add_shift(name="scale_down", id=206, type="shape")
cfg.add_shift(name="pdf_up", id=207, type="shape")
cfg.add_shift(name="pdf_down", id=208, type="shape")
cfg.add_shift(name="alpha_up", id=209, type="shape")
cfg.add_shift(name="alpha_down", id=210, type="shape")

for unc in ["mur", "muf", "scale", "pdf", "alpha"]:
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
cfg.x.external_files = DotDict.wrap({
    # files from TODO
    "lumi": {
        "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
        "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
    },

    # files from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=44#Pileup_JSON_Files_For_Run_II
    "pu": {
        "json": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/pileup_latest.txt", "v1"),  # noqa
        "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/435f0b04c0e318c1036a6b95eb169181bbbe8344/SimGeneral/MixingModule/python/mix_2017_25ns_UltraLegacy_PoissonOOTPU_cfi.py", "v1"),  # noqa
        "data_profile": {
            "nominal": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root", "v1"),  # noqa
            "minbias_xs_up": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-72400ub-99bins.root", "v1"),  # noqa
            "minbias_xs_down": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-66000ub-99bins.root", "v1"),  # noqa
        },
    },

    # jet energy correction
    "jet_jerc": ("/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-f018adfb/POG/JME/2017_UL/jet_jerc.json.gz", "v1"),  # noqa

    # met phi corrector
    "met_phi_corr": ("/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-f018adfb/POG/JME/2017_UL/met.json.gz", "v1"),  # noqa

    # btag scale factor
    "btag_sf_corr": ("/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-f018adfb/POG/BTV/2017_UL/btagging.json.gz", "v1"),  # noqa

    # electron scale factors
    "electron_sf": ("/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-f018adfb/POG/EGM/2017_UL/electron.json.gz", "v1"),  # noqa

    # muon scale factors
    "muon_sf": ("/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-f018adfb/POG/MUO/2017_UL/muon_Z.json.gz", "v1"),  # noqa
})

reduce_events_columns = (
    {
        # general event information
        "run", "luminosityBlock", "event",
        # columns added during selection, required in general
        "mc_weight", "PV.npvs", "category_ids", "deterministic_seed",
        # weight-related columns
        "pu_weight*", "btag_weight*", "scale_weight*", "pdf_weight*",
    } | set(  # Jets
        f"{jet_obj}.{field}"
        for jet_obj in ["Jet", "Bjet", "Lightjet"]
        for field in ["pt", "eta", "phi", "mass", "btagDeepFlavB"]
    ) | set(  # Leptons
        f"{lep}.{field}"
        for lep in ["Electron", "Muon"]
        for field in ["pt", "eta", "phi", "mass", "charge", "pdgId"]
    ) | set(  # MET
        "MET.{field}"
        for field in ["pt", "phi"]
    )
)

# columns to keep after certain steps
cfg.set_aux("keep_columns", DotDict.wrap({
    "cf.SelectEvents": {"mc_weight"},
    "cf.ReduceEvents": reduce_events_columns,
    "cf.MergeSelectionMasks": {
        "mc_weight", "normalization_weight", "process_id", "category_ids", "cutflow.*",
    },
}))

# event weight columns as keys in an ordered dict, mapped to shift instances they depend on
get_shifts = lambda *names: sum(([cfg.get_shift(f"{name}_up"), cfg.get_shift(f"{name}_down")] for name in names), [])
cfg.x.event_weights = DotDict()
cfg.x.event_weights["normalization_weight"] = []
cfg.x.event_weights["pu_weight"] = get_shifts("minbias_xs")
cfg.x.event_weights["electron_weight"] = get_shifts("e_sf")
cfg.x.event_weights["muon_weight"] = get_shifts("mu_sf")

for dataset in cfg.datasets:
    if dataset.x("is_ttbar", False):
        dataset.x.event_weights = {"top_pt_weight": get_shifts("top_pt")}

# TODO: check that pdf/scale weights work for all cases
# for unc in ["mur", "muf", "scale", "pdf", "alpha"]:
#    cfg.x.event_weights[unc] = get_shifts(unc)

# TODO: normalized pu, scale, pdf weights; this also requires saving sum_mc_weights for shift variations
#       and producing the pdf/scale columns as part of the selection or calibration module

# cfg.x.event_weights["normalized_pu_weight"] = get_shifts("minbias_xs")
# cfg.x.event_weights["normalized_scale_weight"] = get_shifts("scale")
# cfg.x.event_weights["normalized_pdf_weight"] = get_shifts("pdf")

# versions per task family and optionally also dataset and shift
# None can be used as a key to define a default value
cfg.set_aux("versions", {
    # None: "dev1",
    # "cf.SelectEvents": "dev1",
})

# add categories
add_categories(cfg)

# add variables
add_variables(cfg)
add_ml_variables(cfg)

# add cutflow variables
add_cutflow_variables(cfg)
add_gen_variables(cfg)
