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
import cmsdb.campaigns.run2_2017

from columnflow.util import DotDict, get_root_processes_from_campaign
from hbw.config.categories import add_categories
from hbw.config.variables import add_variables

from hbw.config.analysis_hbw import analysis_hbw

thisdir = os.path.dirname(os.path.abspath(__file__))

#
# 2017 standard config
#

# copy the campaign, which in turn copies datasets and processes
campaign_run2_2017 = cmsdb.campaigns.run2_2017.campaign_run2_2017.copy()

# get all root processes
procs = get_root_processes_from_campaign(campaign_run2_2017)

# create a config by passing the campaign, so id and name will be identical
config_2017 = analysis_hbw.add_config(campaign_run2_2017)

# add processes we are interested in
config_2017.add_process(procs.n.data)
config_2017.add_process(procs.n.tt)
config_2017.add_process(procs.n.st)
config_2017.add_process(procs.n.w_lnu)
config_2017.add_process(procs.n.dy_lep)
# config_2017.add_process(procs.n.qcd)
# config_2017.add_process(procs.n.ttv)
# config_2017.add_process(procs.n.vv)
# config_2017.add_process(procs.n.vv)
config_2017.add_process(procs.n.hh_ggf_kt_1_kl_0_bbww_sl)
config_2017.add_process(procs.n.hh_ggf_kt_1_kl_1_bbww_sl)
config_2017.add_process(procs.n.hh_ggf_kt_1_kl_2p45_bbww_sl)
config_2017.add_process(procs.n.hh_ggf_kt_1_kl_5_bbww_sl)

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
    "hh_ggf_kt_1_kl_1_bbww_sl": "#000000",  # black
    "hh_ggf_kt_1_kl_0_bbww_sl": "#1b9e77",  # green2
    "hh_ggf_kt_1_kl_2p45_bbww_sl": "#d95f02",  # orange2
    "hh_ggf_kt_1_kl_5_bbww_sl": "#e7298a",  # pink2
}
for proc, color in colors.items():
    if proc in config_2017.processes:
        config_2017.get_process(proc).color = color

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
    "hh_ggf_kt_1_kl_0_bbww_sl_powheg",
    "hh_ggf_kt_1_kl_1_bbww_sl_powheg",
    "hh_ggf_kt_1_kl_2p45_bbww_sl_powheg",
    "hh_ggf_kt_1_kl_5_bbww_sl_powheg",
]
for dataset_name in dataset_names:
    dataset = config_2017.add_dataset(campaign_run2_2017.get_dataset(dataset_name))

    # reduce n_files to 2 for testing purposes (TODO switch to full dataset)
    for k in dataset.info.keys():
        if dataset.name == "hh_ggf_kt_1_kl_1_bbww_sl_powheg":  # full stats for HH
            continue
        dataset[k].n_files = 2

    # add aux info to datasets
    if dataset.name.startswith(("st", "tt")):
        dataset.x.has_top = True
    if dataset.name.startswith("tt"):
        dataset.x.is_ttbar = True
        dataset.x.event_weights = ["top_pt_weight"]
    if "hh" in dataset.name and "bbww" in dataset.name:
        dataset.x.is_hbw = True


# default calibrator, selector, producer, ml model and inference model
config_2017.set_aux("default_calibrator", "default")
config_2017.set_aux("default_selector", "default")
config_2017.set_aux("default_producer", "features")
config_2017.set_aux("default_ml_model", None)
config_2017.set_aux("default_inference_model", "test")
config_2017.set_aux("default_process_settings", [["hh_ggf_kt_1_kl_1_bbww_sl", "scale=2000", "unstack"]])

# process groups for conveniently looping over certain processs
# (used in wrapper_factory and during plotting)
config_2017.set_aux("process_groups", {
    "hh": ["hh_ggf_kt_1_kl_1_bbww_sl"],
    "default": ["hh_ggf_kt_1_kl_1_bbww_sl", "dy_lep", "w_lnu", "st", "tt"],
    "working": ["hh_ggf_kt_1_kl_1_bbww_sl", "dy_lep", "st", "tt"],
    "test": ["hh_ggf_kt_1_kl_1_bbww_sl", "tt_sl"],
    "small": ["hh_ggf_kt_1_kl_1_bbww_sl", "st", "tt"],
    "signal": ["hh_ggf_kt_1_kl_0_bbww_sl", "hh_ggf_kt_1_kl_1_bbww_sl",
               "hh_ggf_kt_1_kl_2p45_bbww_sl", "hh_ggf_kt_1_kl_5_bbww_sl"],
    "bkg": ["tt", "st", "w_lnu", "dy_lep"],
})

# dataset groups for conveniently looping over certain datasets
# (used in wrapper_factory and during plotting)
config_2017.set_aux("dataset_groups", {
    "all": ["*"],
    "working": ["tt_*", "st_*", "dy_*"],
    "tt": ["tt_*"], "st": ["st_*"], "w": ["w_lnu*"], "dy": ["dy_*"],
    "hh": ["hh_*"], "hhsm": ["hh_ggf_kt_1_kl_1_bbww_sl_powheg"],
})

# category groups for conveniently looping over certain categories
# (used during plotting)
config_2017.set_aux("category_groups", {
    "default": ["incl", "1e", "1mu"],
    "test": ["incl", "1e"],
})

# variable groups for conveniently looping over certain variables
# (used during plotting)
config_2017.set_aux("variable_groups", {
    "default": ["n_jet", "n_muon", "n_electron", "ht", "m_bb", "deltaR_bb", "jet1_pt"],  # n_deepjet, ....
    "test": ["n_jet", "n_electron", "jet1_pt"],
    "cutflow": ["cf_jet1_pt", "cf_jet4_pt", "cf_n_jet", "cf_n_electron", "cf_n_muon"],  # cf_n_deepjet
})

# shift groups for conveniently looping over certain shifts
# (used during plotting)
config_2017.set_aux("shift_groups", {
    "jer": ["nominal", "jer_up", "jer_down"],
})

# selector step groups for conveniently looping over certain steps
# (used in cutflow tasks)
config_2017.set_aux("selector_step_groups", {
    "default": ["Lepton", "VetoLepton", "Jet", "Bjet", "Trigger"],
    "thesis": ["Lepton", "Jet", "Trigger", "Bjet"],  # reproduce master thesis cuts to check if everything works
    "test": ["Lepton", "Jet", "Bjet"],
})

config_2017.set_aux("selector_step_labels", {
    "Jet": r"$N_{Jets} \geq 3$",
    "Lepton": r"$N_{Lepton} = 1$",
    "Bjet": r"$N_{Jets}^{BTag} \geq 1$",
})


# process settings groups to quickly define settings for ProcessPlots
config_2017.set_aux("process_settings_groups", {
    "default": [["hh_ggf_kt_1_kl_1_bbww_sl", "scale=2000", "unstack"]],
    "unstack_all": [[proc, "unstack"] for proc in config_2017.processes],
})

# 2017 luminosity with values in inverse pb and uncertainties taken from
# https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM?rev=176#LumiComb
config_2017.set_aux("luminosity", Number(41480, {
    "lumi_13TeV_2017": (REL, 0.02),
    "lumi_13TeV_1718": (REL, 0.006),
    "lumi_13TeV_correlated": (REL, 0.009),
}))

# 2017 minimum bias cross section in mb (milli) for creating PU weights, values from
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=44#Pileup_JSON_Files_For_Run_II
config_2017.set_aux("minbiasxs", Number(69.2, (REL, 0.046)))

# 2017 b-tag working points
# https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=15
config_2017.x.btag_working_points = DotDict.wrap({
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

# location of JEC txt files
config_2017.set_aux("jec", DotDict.wrap({
    "source": "https://raw.githubusercontent.com/cms-jet/JECDatabase/master/textFiles",
    "campaign": "Summer19UL17",
    "version": "V6",
    "jet_type": "AK4PFchs",
    "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
    "data_eras": ["RunB", "RunC", "RunD", "RunE", "RunF"],
    "uncertainty_sources": [
        # comment out most for now to prevent large file sizes
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
}))

config_2017.set_aux("jer", DotDict.wrap({
    "source": "https://raw.githubusercontent.com/cms-jet/JRDatabase/master/textFiles",
    "campaign": "Summer19UL17",
    "version": "JRV2",
    "jet_type": "AK4PFchs",
}))


# helper to add column aliases for both shifts of a source
def add_aliases(shift_source: str, aliases: Set[str], selection_dependent: bool):
    for direction in ["up", "down"]:
        shift = config_2017.get_shift(od.Shift.join_name(shift_source, direction))
        # format keys and values
        inject_shift = lambda s: re.sub(r"\{([^_])", r"{_\1", s).format(**shift.__dict__)
        _aliases = {inject_shift(key): inject_shift(value) for key, value in aliases.items()}
        alias_type = "column_aliases_selection_dependent" if selection_dependent else "column_aliases"
        # extend existing or register new column aliases
        shift.set_aux(alias_type, shift.get_aux(alias_type, {})).update(_aliases)


# register shifts
config_2017.add_shift(name="nominal", id=0)
config_2017.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
config_2017.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})
config_2017.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal"})
config_2017.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal"})
config_2017.add_shift(name="minbias_xs_up", id=7, type="shape")
config_2017.add_shift(name="minbias_xs_down", id=8, type="shape")
add_aliases("minbias_xs", {"pu_weight": "pu_weight_{name}"}, selection_dependent=False)
config_2017.add_shift(name="top_pt_up", id=9, type="shape")
config_2017.add_shift(name="top_pt_down", id=10, type="shape")
add_aliases("top_pt", {"top_pt_weight": "top_pt_weight_{direction}"}, selection_dependent=False)

config_2017.add_shift(name="mur_up", id=101, type="shape")
config_2017.add_shift(name="mur_down", id=102, type="shape")
config_2017.add_shift(name="muf_up", id=103, type="shape")
config_2017.add_shift(name="muf_down", id=104, type="shape")
config_2017.add_shift(name="scale_up", id=105, type="shape")
config_2017.add_shift(name="scale_down", id=106, type="shape")
config_2017.add_shift(name="pdf_up", id=107, type="shape")
config_2017.add_shift(name="pdf_down", id=108, type="shape")
config_2017.add_shift(name="alpha_up", id=109, type="shape")
config_2017.add_shift(name="alpha_down", id=110, type="shape")

for unc in ["mur", "muf", "scale", "pdf", "alpha"]:
    add_aliases(unc, {f"{unc}_weight": unc + "_weight_{direction}"}, selection_dependent=False)

with open(os.path.join(thisdir, "jec_sources.yaml"), "r") as f:
    all_jec_sources = yaml.load(f, yaml.Loader)["names"]
for jec_source in config_2017.x.jec["uncertainty_sources"]:
    idx = all_jec_sources.index(jec_source)
    config_2017.add_shift(name=f"jec_{jec_source}_up", id=5000 + 2 * idx, type="shape")
    config_2017.add_shift(name=f"jec_{jec_source}_down", id=5001 + 2 * idx, type="shape")
    add_aliases(
        f"jec_{jec_source}",
        {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"},
        selection_dependent=True,
    )

config_2017.add_shift(name="jer_up", id=6000, type="shape", tags={"selection_dependent"})
config_2017.add_shift(name="jer_down", id=6001, type="shape", tags={"selection_dependent"})
add_aliases("jer", {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"}, selection_dependent=True)


def make_jme_filenames(jme_aux, sample_type, names, era=None):
    """Convenience function to compute paths to JEC files."""

    # normalize and validate sample type
    sample_type = sample_type.upper()
    if sample_type not in ("DATA", "MC"):
        raise ValueError(f"Invalid sample type '{sample_type}'. Expected either 'DATA' or 'MC'.")

    jme_full_version = "_".join(s for s in (jme_aux.campaign, era, jme_aux.version, sample_type) if s)

    return [
        f"{jme_aux.source}/{jme_full_version}/{jme_full_version}_{name}_{jme_aux.jet_type}.txt"
        for name in names
    ]


# TODO check names
# external files
config_2017.set_aux("external_files", DotDict.wrap({
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
    "jec": {
        "mc": [
            (fname, "v1")
            for fname in make_jme_filenames(config_2017.x.jec, "mc", names=config_2017.x.jec.levels)
        ],
        "data": {
            era: [
                (fname, "v1")
                for fname in make_jme_filenames(config_2017.x.jec, "data", names=config_2017.x.jec.levels, era=era)
            ]
            for era in config_2017.x.jec.data_eras
        },
    },

    # jec energy correction uncertainties
    "junc": {
        "mc": [(make_jme_filenames(config_2017.x.jec, "mc", names=["UncertaintySources"])[0], "v1")],
        "data": {
            era: [(make_jme_filenames(config_2017.x.jec, "data", names=["UncertaintySources"], era=era)[0], "v1")]
            for era in config_2017.x.jec.data_eras
        },
    },

    # jet energy resolution (pt resolution)
    "jer": {
        "mc": [(make_jme_filenames(config_2017.x.jer, "mc", names=["PtResolution"])[0], "v1")],
    },

    # jet energy resolution (data/mc scale factors)
    "jersf": {
        "mc": [(make_jme_filenames(config_2017.x.jer, "mc", names=["SF"])[0], "v1")],
    },

}))

# columns to keep after certain steps
config_2017.set_aux("keep_columns", DotDict.wrap({
    "cf.SelectEvents": {"mc_weight"},
    "cf.ReduceEvents": {
        # general event information
        "run", "luminosityBlock", "event",
        # weights
        "LHEWeight.*",
        "LHEPdfWeight", "nLHEPdfWeight", "LHEScaleWeight", "nLHEScaleWeight",
        # object properties
        "nJet", "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB",
        "Bjet.pt", "Bjet.eta", "Bjet.phi", "Bjet.mass", "Bjet.btagDeepFlavB",
        # "Muon.*", "Electron.*", "MET.*",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "MET.pt", "MET.phi",
        # columns added during selection, required in general
        "mc_weight", "PV.npvs", "category_ids", "deterministic_seed",
    },
    "cf.MergeSelectionMasks": {
        "mc_weight", "normalization_weight", "process_id", "category_ids", "cutflow.*",
    },
}))

# event weight columns
config_2017.set_aux("event_weights", ["normalization_weight", "pu_weight"])
# TODO: enable different cases for number of pdf/scale weights
# config_2017.set_aux("event_weights", ["normalization_weight", "pu_weight", "scale_weight", "pdf_weight"])

# versions per task family and optionally also dataset and shift
# None can be used as a key to define a default value
config_2017.set_aux("versions", {
})

# add categories
add_categories(config_2017)

# add variables
add_variables(config_2017)
