# coding: utf-8

"""
Configuration of the Run 2 HH -> bbWW datasets.
"""

from __future__ import annotations

import itertools

import law
import order as od
from scinum import Number

import cmsdb.processes as cmsdb_procs
from columnflow.util import DotDict
from columnflow.tasks.external import GetDatasetLFNs
from columnflow.config_util import get_root_processes_from_campaign


logger = law.logger.get_logger(__name__)


#
# Collection of process names and their corresponding datasets for each year
#


data_mu = {
    "2017": [
        # "data_mu_b",  # missing triggers in DL
        "data_mu_c",
        "data_mu_d",
        "data_mu_e",
        "data_mu_f",
    ],
    "2022preEE": [
        "data_mu_c",
        "data_mu_d",
    ],
    "2022postEE": [
        "data_mu_e",
        "data_mu_f",
        "data_mu_g",
    ],
}

data_e = {
    "2017": [
        # "data_e_b",  # missing triggers in DL
        "data_e_c",
        "data_e_d",
        "data_e_e",
        "data_e_f",
    ],
}

data_egamma = {
    "2022preEE": [
        "data_egamma_c",
        "data_egamma_d",
    ],
    "2022postEE": [
        "data_egamma_e",
        "data_egamma_f",
        "data_egamma_g",
    ],
}

# commented out because of empty datasets
data_muoneg = {
    "2022preEE": [
        "data_muoneg_c",
        "data_muoneg_d",
    ],
    "2022postEE": [
        "data_muoneg_e",
        "data_muoneg_f",
        "data_muoneg_g",
    ],
}

tt = {
    "2017": [
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
    ],
    "2022preEE": [
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
    ],
    "2022postEE": [
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
    ],
}

st = {
    "2017": [
        "st_tchannel_t_4f_powheg",
        "st_tchannel_tbar_4f_powheg",
        "st_twchannel_t_powheg",
        "st_twchannel_tbar_powheg",
        "st_schannel_lep_4f_amcatnlo",
        "st_schannel_had_4f_amcatnlo",
    ],
    "2022preEE": [
        "st_tchannel_t_4f_powheg",
        "st_tchannel_tbar_4f_powheg",
        "st_twchannel_t_sl_powheg",
        "st_twchannel_tbar_sl_powheg",
        "st_twchannel_t_dl_powheg",
        "st_twchannel_tbar_dl_powheg",
        "st_twchannel_t_fh_powheg",
        "st_twchannel_tbar_fh_powheg",
        # "st_schannel_lep_4f_amcatnlo",
        # "st_schannel_had_4f_amcatnlo",
    ],
    "2022postEE": [
        "st_tchannel_t_4f_powheg",
        "st_tchannel_tbar_4f_powheg",
        "st_twchannel_t_sl_powheg",
        "st_twchannel_tbar_sl_powheg",
        "st_twchannel_t_dl_powheg",
        "st_twchannel_tbar_dl_powheg",
        "st_twchannel_t_fh_powheg",
        "st_twchannel_tbar_fh_powheg",
        # "st_schannel_lep_4f_amcatnlo",
        # "st_schannel_had_4f_amcatnlo",
    ],
}

w_lnu = {
    "2017": [
        "w_lnu_ht70to100_madgraph",
        "w_lnu_ht100to200_madgraph",
        "w_lnu_ht200to400_madgraph",
        "w_lnu_ht400to600_madgraph",
        "w_lnu_ht600to800_madgraph",
        "w_lnu_ht800to1200_madgraph",
        "w_lnu_ht1200to2500_madgraph",
        "w_lnu_ht2500toinf_madgraph",
    ],
    "2022postEE": [
        "w_lnu_amcatnlo",
    ],
    "2022preEE": [
        "w_lnu_amcatnlo",
    ],
}

dy = {  # TODO: stitching
    "2017": [
        "dy_m50toinf_ht70to100_madgraph",
        "dy_m50toinf_ht100to200_madgraph",
        "dy_m50toinf_ht200to400_madgraph",
        "dy_m50toinf_ht400to600_madgraph",
        "dy_m50toinf_ht600to800_madgraph",
        "dy_m50toinf_ht800to1200_madgraph",
        "dy_m50toinf_ht1200to2500_madgraph",
        "dy_m50toinf_ht2500toinf_madgraph",
    ],
    "2022postEE": [
        "dy_m50toinf_amcatnlo",
        "dy_m10to50_amcatnlo",
        "dy_m4to10_amcatnlo",
        "dy_m50toinf_0j_amcatnlo",
        "dy_m50toinf_1j_amcatnlo",
        "dy_m50toinf_2j_amcatnlo",
    ],
    "2022preEE": [
        "dy_m50toinf_amcatnlo",
        "dy_m10to50_amcatnlo",
        "dy_m4to10_amcatnlo",
        "dy_m50toinf_0j_amcatnlo",
        "dy_m50toinf_1j_amcatnlo",
        "dy_m50toinf_2j_amcatnlo",
    ],
}

qcd_mu = {
    "2017": [
        "qcd_mu_pt15to20_pythia",
        "qcd_mu_pt20to30_pythia",
        "qcd_mu_pt30to50_pythia",
        "qcd_mu_pt50to80_pythia",
        "qcd_mu_pt80to120_pythia",
        "qcd_mu_pt120to170_pythia",
        "qcd_mu_pt170to300_pythia",
        "qcd_mu_pt300to470_pythia",
        "qcd_mu_pt470to600_pythia",
        "qcd_mu_pt600to800_pythia",
        "qcd_mu_pt800to1000_pythia",
        "qcd_mu_pt1000toinf_pythia",
    ],
    "2022postEE": [
        # "qcd_mu_pt15to20_pythia",  # empty after selection
        "qcd_mu_pt20to30_pythia",
        "qcd_mu_pt30to50_pythia",
        "qcd_mu_pt50to80_pythia",
        "qcd_mu_pt80to120_pythia",
        "qcd_mu_pt120to170_pythia",
        "qcd_mu_pt170to300_pythia",
        "qcd_mu_pt300to470_pythia",
        "qcd_mu_pt470to600_pythia",
        "qcd_mu_pt600to800_pythia",
        "qcd_mu_pt800to1000_pythia",
        "qcd_mu_pt1000toinf_pythia",
    ],
    "2022preEE": [
        "qcd_mu_pt15to20_pythia",
        # "qcd_mu_pt20to30_pythia",  # file stuck
        "qcd_mu_pt30to50_pythia",
        "qcd_mu_pt50to80_pythia",
        "qcd_mu_pt80to120_pythia",
        "qcd_mu_pt120to170_pythia",
        "qcd_mu_pt170to300_pythia",
        "qcd_mu_pt300to470_pythia",
        "qcd_mu_pt470to600_pythia",
        "qcd_mu_pt600to800_pythia",
        "qcd_mu_pt800to1000_pythia",
        "qcd_mu_pt1000toinf_pythia",
    ],
}

qcd_em = {
    "2017": [
        "qcd_em_pt15to20_pythia",
        "qcd_em_pt20to30_pythia",
        "qcd_em_pt30to50_pythia",
        "qcd_em_pt50to80_pythia",
        "qcd_em_pt80to120_pythia",
        "qcd_em_pt120to170_pythia",
        "qcd_em_pt170to300_pythia",
        "qcd_em_pt300toinf_pythia",
    ],
    "2022postEE": [
        # "qcd_em_pt10to30_pythia",  # missing process + probably empty anyways
        "qcd_em_pt30to50_pythia",  # empty after selection
        "qcd_em_pt50to80_pythia",
        "qcd_em_pt80to120_pythia",
        "qcd_em_pt120to170_pythia",
        "qcd_em_pt170to300_pythia",
        "qcd_em_pt300toinf_pythia",
    ],
    "2022preEE": [
        # "qcd_em_pt10to30_pythia",  # missing process + probably empty anyways
        "qcd_em_pt30to50_pythia",
        "qcd_em_pt50to80_pythia",
        "qcd_em_pt80to120_pythia",
        "qcd_em_pt120to170_pythia",
        "qcd_em_pt170to300_pythia",
        "qcd_em_pt300toinf_pythia",
    ],
}

qcd_bctoe = {
    "2017": [
        "qcd_bctoe_pt15to20_pythia",
        "qcd_bctoe_pt20to30_pythia",
        "qcd_bctoe_pt30to80_pythia",
        "qcd_bctoe_pt80to170_pythia",
        "qcd_bctoe_pt170to250_pythia",
        "qcd_bctoe_pt250toinf_pythia",
    ],
    "2022postEE": [
        # empty for now
    ],
}

h = {
    "2017": [
        # empty for now
    ],
    "2022postEE": [
        "h_ggf_hbb_powheg",
        "h_ggf_hww2l2nu_powheg",
        "h_vbf_hbb_powheg",
        "h_vbf_hww2l2nu_powheg",
        "zh_zqq_hbb_powheg",
        "zh_zll_hbb_powheg",
        "zh_zll_hcc_powheg",
        "zh_hww2l2nu_powheg",
        "zh_gg_zll_hbb_powheg",
        "zh_gg_zqq_hbb_powheg",
        "zh_gg_znunu_hbb_powheg",
        "zh_gg_zll_hcc_powheg",
        "wph_wqq_hbb_powheg",
        "wph_wlnu_hbb_powheg",
        "wph_wqq_hcc_powheg",
        "wph_wlnu_hcc_powheg",
        "wph_hzg_zll_powheg",
        "wmh_wqq_hbb_powheg",
        "wmh_wlnu_hbb_powheg",
        "wmh_wqq_hcc_powheg",
        "wmh_wlnu_hcc_powheg",
        "wmh_hzg_zll_powheg",
        "tth_hbb_powheg",
        "tth_hnonbb_powheg",  # overlap with other samples, so be careful
        "ttzh_madgraph",
        "ttwh_madgraph",
    ],
    "2022preEE": [
        "h_ggf_hbb_powheg",
        "h_ggf_hww2l2nu_powheg",
        "h_vbf_hbb_powheg",
        "h_vbf_hww2l2nu_powheg",
        "zh_zqq_hbb_powheg",
        "zh_zll_hbb_powheg",
        "zh_zll_hcc_powheg",
        "zh_hww2l2nu_powheg",
        "zh_gg_zll_hbb_powheg",
        "zh_gg_zqq_hbb_powheg",
        "zh_gg_znunu_hbb_powheg",
        "zh_gg_zll_hcc_powheg",
        "wph_wqq_hbb_powheg",
        "wph_wlnu_hbb_powheg",
        "wph_wqq_hcc_powheg",
        "wph_wlnu_hcc_powheg",
        "wph_hzg_zll_powheg",
        "wmh_wqq_hbb_powheg",
        "wmh_wlnu_hbb_powheg",
        "wmh_wqq_hcc_powheg",
        "wmh_wlnu_hcc_powheg",
        "wmh_hzg_zll_powheg",
        "tth_hbb_powheg",
        "tth_hnonbb_powheg",  # overlap with other samples, so be careful
        "ttzh_madgraph",
        "ttwh_madgraph",
    ],
}

# cross sections still missing
vv = {
    "2017": [
        # empty for now
    ],
    "2022preEE": [
        "ww_pythia",
        "wz_pythia",
        "zz_pythia",
    ],
    "2022postEE": [
        "ww_pythia",
        "wz_pythia",
        "zz_pythia",
    ],
}

ttv = {
    "2017": [
        # empty for now
    ],
    "2022postEE": [
        # empty for now
    ],
}

hh_ggf_hbb_hvv = {
    "2017": [
        # SL
        "hh_ggf_hbb_hvvqqlnu_kl0_kt1_powheg",
        "hh_ggf_hbb_hvvqqlnu_kl1_kt1_powheg",
        "hh_ggf_hbb_hvvqqlnu_kl2p45_kt1_powheg",
        "hh_ggf_hbb_hvvqqlnu_kl5_kt1_powheg",
        # DL
        "hh_ggf_hbb_hvv2l2nu_kl0_kt1_powheg",
        "hh_ggf_hbb_hvv2l2nu_kl1_kt1_powheg",
        "hh_ggf_hbb_hvv2l2nu_kl2p45_kt1_powheg",
        "hh_ggf_hbb_hvv2l2nu_kl5_kt1_powheg",
    ],
    "2022preEE": [
        "hh_ggf_hbb_hvvqqlnu_kl1_kt1_powheg",  # SL
        "hh_ggf_hbb_hvv2l2nu_kl1_kt1_powheg",  # DL
        "hh_ggf_hbb_hvv_kl1_kt1_powheg",  # incl

    ],
    "2022postEE": [
        "hh_ggf_hbb_hvvqqlnu_kl1_kt1_powheg",  # SL
        "hh_ggf_hbb_hvv2l2nu_kl1_kt1_powheg",  # DL
        "hh_ggf_hbb_hvv_kl1_kt1_powheg",  # incl
    ],
}

hh_vbf_hbb_hvvqqlnu = {
    "2017": [
        "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl1_madgraph",
        "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl0_madgraph",
        "hh_vbf_hbb_hvvqqlnu_kv1_k2v1_kl2_madgraph",
        "hh_vbf_hbb_hvvqqlnu_kv1_k2v0_kl1_madgraph",
        "hh_vbf_hbb_hvvqqlnu_kv1_k2v2_kl1_madgraph",
        "hh_vbf_hbb_hvvqqlnu_kv0p5_k2v1_kl1_madgraph",
        "hh_vbf_hbb_hvvqqlnu_kv1p5_k2v1_kl1_madgraph",
    ],
    "2022postEE": [
        # empty for now
    ],
}

hh_vbf_hbb_hvv2l2nu = {
    "2017": [
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl1_madgraph",
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl0_madgraph",
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v1_kl2_madgraph",
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v0_kl1_madgraph",
        "hh_vbf_hbb_hvv2l2nu_kv1_k2v2_kl1_madgraph",
        "hh_vbf_hbb_hvv2l2nu_kv0p5_k2v1_kl1_madgraph",
        "hh_vbf_hbb_hvv2l2nu_kv1p5_k2v1_kl1_madgraph",
    ],
    "2022postEE": [
        # empty for now
    ],
}

graviton_hh_ggf_bbww = {
    "2017": [f"graviton_hh_ggf_bbww_m{mass}_madgraph" for mass in [
        250, 260, 270, 280, 300, 320, 350, 400, 450, 500,
        550, 600, 650, 700, 750, 800, 850, 900, 1000,
        1250, 1500, 1750, 2000, 2500, 3000,
    ]],
    "2022postEE": [
        # empty for now
    ],
}
radion_hh_ggf_bbww = {
    "2017": [f"radion_hh_ggf_bbww_m{mass}_madgraph" for mass in [
        250, 260, 270, 280, 300, 320, 350, 400, 450, 500,
        550, 600, 650, 700, 750, 800, 850, 900, 1000,
        1250, 1500, 1750, 2000, 2500, 3000,
    ]],
    "2022postEE": [
    ],
}


def get_dataset_names(cpn_tag: int | str, as_list: bool = False) -> DotDict[str: list[str]] | list[str]:
    """
    Central definition of datasets used in the hbb_hvv analysis based on the *cpn_tag*.
    As a default, it creates one DotDict of process names mapped to the corresponding dataset names.
    When *as_list* is True, a single list of all datasets is returned

    :param cpn_tag: String or integer of which data-taking campaign is used.
    :param as_list: Bool parameter to decide whether to return a list or a DotDict.
    :return: List or DotDict containing all dataset names.
    """

    cpn_tag = str(cpn_tag)

    #
    # Combine all process datasets into a single DotDict
    #

    dataset_names = DotDict.wrap(
        data_mu=data_mu.get(cpn_tag, []),
        data_e=data_e.get(cpn_tag, []),
        data_egamma=data_egamma.get(cpn_tag, []),
        data_muoneg=data_muoneg.get(cpn_tag, []),
        tt=tt.get(cpn_tag, []),
        st=st.get(cpn_tag, []),
        w_lnu=w_lnu.get(cpn_tag, []),
        dy=dy.get(cpn_tag, []),
        qcd_mu=qcd_mu.get(cpn_tag, []),
        qcd_em=qcd_em.get(cpn_tag, []),
        qcd_bctoe=qcd_bctoe.get(cpn_tag, []),
        h=h.get(cpn_tag, []),
        vv=vv.get(cpn_tag, []),
        ttv=ttv.get(cpn_tag, []),
        hh_ggf_hbb_hvv=hh_ggf_hbb_hvv.get(cpn_tag, []),
        hh_vbf_hbb_hvvqqlnu=hh_vbf_hbb_hvvqqlnu.get(cpn_tag, []),
        hh_vbf_hbb_hvv2l2nu=hh_vbf_hbb_hvv2l2nu.get(cpn_tag, []),
        graviton_hh_ggf_bbww=graviton_hh_ggf_bbww.get(cpn_tag, []),
        radion_hh_ggf_bbww=radion_hh_ggf_bbww.get(cpn_tag, []),
    )
    if as_list:
        return list(itertools.chain(*dataset_names.values()))

    return dataset_names


def get_dataset_names_for_config(config: od.Config, as_list: bool = False):
    """
    get all relevant dataset names and modify them based on the config and campaign
    """

    cpn_tag = str(config.x.cpn_tag)
    dataset_names = get_dataset_names(cpn_tag, as_list)
    # optionally switch to custom signal processes (only implemented for hh_ggf_sl)
    if config.has_tag("custom_signals"):
        dataset_names.hh_ggf_hbb_hvvqqlnu = [dataset_name.replace("powheg", "custom") for dataset_name in dataset_names]

    if not config.has_tag("is_resonant"):
        # remove all resonant signal processes/datasets
        dataset_names.pop("graviton_hh_ggf_bbww")
        dataset_names.pop("radion_hh_ggf_bbww")

    if not config.has_tag("is_sl"):
        # remove qcd datasets from DL
        dataset_names.pop("qcd_mu")
        dataset_names.pop("qcd_em")
        dataset_names.pop("qcd_bctoe")

    if not config.has_tag("is_nonresonant"):
        # remove all nonresonant signal processes/datasets
        for hh_proc in ("hh_ggf_hbb_hvv", "qHH_hbb_hvvqqlnu", "hh_vbf_hbb_hvv2l2nu"):
            dataset_names.pop(hh_proc)

    return dataset_names


def add_synchronization_dataset(config: od.Config):
    radion_hh_ggf_dl_bbww_m450 = config.add_process(
        name="radion_hh_ggf_dl_bbww_m450",
        id=24563574,  # random number
        xsecs={13: Number(0.1), 13.6: Number(0.1)},  # TODO
    )

    config.add_dataset(
        name="radion_hh_ggf_dl_bbww_m450_magraph",
        id=14876684,
        processes=[radion_hh_ggf_dl_bbww_m450],
        keys=[
            "/GluGlutoRadiontoHHto2B2Vto2B2L2Nu_M-450_narrow_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM",  # noqa
        ],
        n_files=19,
        n_events=87308,
    )


def add_hbw_processes_and_datasets(config: od.Config, campaign: od.Campaign):
    if config.x.cpn_tag == "2022postEE":
        add_synchronization_dataset(config)

    # if campaign.x.year == 2017:
    #     # load custom produced datasets into campaign (2017 only!)
    #     get_custom_hh_2017_datasets(campaign)

    #     # use custom get_dataset_lfns function
    #     config.x.get_dataset_lfns = get_dataset_lfns_2017

    dataset_names = get_dataset_names_for_config(config)

    # get all process names with corresponding datasets
    process_names = [proc_name for proc_name, _dataset_names in dataset_names.items() if _dataset_names]

    # get all root processes
    config.x.procs = procs = get_root_processes_from_campaign(campaign)

    # add processes to config
    for proc_name in process_names:
        config.add_process(procs.n(proc_name))

    # # add leaf signal processes directly to the config
    # for signal_proc in ("hh_ggf_hbb_hvvqqlnu", "hh_ggf_hbb_hvv2l2nu", "hh_vbf_hbb_hvvqqlnu", "hh_vbf_hbb_hvv2l2nu"):
    #     for dataset_name in dataset_names[signal_proc]:
    #         config.add_process(procs.n(dataset_name.replace("_powheg", "").replace("_madgraph", "")))

    # loop over all dataset names and add them to the config
    for dataset_name in list(itertools.chain(*dataset_names.values())):
        config.add_dataset(campaign.get_dataset(dataset_name))


def add_dataset_extension_to_nominal(dataset: od.Dataset) -> None:
    """
    Adds relevant keys from "extension" DatasetInfo to the "nominal" DatasetInfo
    """
    if "extension" in dataset.info.keys():
        dataset_info_ext = dataset.info["extension"]

        # add info from extension dataset to the nominal one
        dataset.info["nominal"].keys = dataset.info["nominal"].keys + dataset_info_ext.keys
        dataset.info["nominal"].n_files = dataset.info["nominal"].n_files + dataset_info_ext.n_files
        dataset.info["nominal"].n_events = dataset.info["nominal"].n_events + dataset_info_ext.n_events

        # remove the extension dataset info, since it is now included in "nominal"
        dataset.info.pop("extension")


def configure_hbw_datasets(
    config: od.Config,
    limit_dataset_files: int | None = None,
    add_dataset_extensions: bool = False,
):
    for dataset in config.datasets:
        if add_dataset_extensions:
            add_dataset_extension_to_nominal(dataset)

        if limit_dataset_files:
            # apply optional limit on the max. number of files per dataset
            for info in dataset.info.values():
                if info.n_files > limit_dataset_files:
                    info.n_files = limit_dataset_files

        # add aux info to datasets
        if dataset.name.startswith(("st", "tt")):
            dataset.add_tag("has_top")
        if dataset.name.startswith("tt"):
            dataset.add_tag("is_ttbar")

        if dataset.name.startswith("dy_"):
            dataset.add_tag("is_v_jets")
            dataset.add_tag("is_z_jets")
        if dataset.name.startswith("w_"):
            dataset.add_tag("is_v_jets")
            dataset.add_tag("is_w_jets")

        if dataset.name.startswith("qcd"):
            dataset.add_tag("is_qcd")

        if "hh" in dataset.name and "hbb_hvv" in dataset.name:
            # add HH signal tags
            dataset.add_tag("is_hbv")
            if "ggf" in dataset.name:
                dataset.add_tag("is_hbv_ggf")

            elif "vbf" in dataset.name:
                dataset.add_tag("is_hbv_vbf")
            if "qqlnu" not in dataset.name:
                dataset.add_tag("is_hbv_dl")
            elif "2l2nu" not in dataset.name:
                dataset.add_tag("is_hbv_sl")
            else:
                dataset.add_tag("is_hbv_incl")

        if dataset.name.endswith("_pythia") or "hh_vbf" in dataset.name:
            dataset.add_tag("skip_scale")
            dataset.add_tag("skip_pdf")
            dataset.add_tag("no_lhe_weights")

        if dataset.has_tag("is_hbv") and "custom" in dataset.name:
            # No PDF weights and 6 scale weights in custom HH samples
            dataset.add_tag("skip_scale")
            dataset.add_tag("skip_pdf")

        elif config.campaign.x.year == 2017:
            # our default Run2 signal samples are EOY, so we have to skip golden json, certain met filter
            dataset.add_tag("is_eoy")

        if dataset.is_data:
            if config.x.cpn_tag == "2022preEE":
                dataset.x.jec_era = "RunCD"


def get_custom_hh_2017_datasets(
    campaign: od.Campaign,
) -> None:
    """
    Add custom HH datasets to campaign
    """
    campaign.add_dataset(
        name="hh_ggf_hbb_hvvqqlnu_kl0_kt1_custom",
        id=10 ** 8 + 14057341,
        processes=[cmsdb_procs.hh_ggf_hbb_hvvqqlnu_kl0_kt1],
        keys=[
            "chhh0",
        ],
        n_files=2,
        n_events=493996,
        aux={"custom": True},
    )

    campaign.add_dataset(
        name="hh_ggf_hbb_hvvqqlnu_kl1_kt1_custom",
        id=10 ** 8 + 14065482,
        processes=[cmsdb_procs.hh_ggf_hbb_hvvqqlnu_kl1_kt1],
        keys=[
            "chhh1",
        ],
        n_files=2,
        n_events=498499,
        aux={"custom": True},
    )

    campaign.add_dataset(
        name="hh_ggf_hbb_hvvqqlnu_kl2p45_kt1_custom",
        id=10 ** 8 + 14066581,
        processes=[cmsdb_procs.hh_ggf_hbb_hvvqqlnu_kl2p45_kt1],
        keys=[
            "chhh2p45",
        ],
        n_files=2,
        n_events=498496,
        aux={"custom": True},
    )

    campaign.add_dataset(
        name="hh_ggf_hbb_hvvqqlnu_kl5_kt1_custom",
        id=10 ** 8 + 14058363,
        processes=[cmsdb_procs.hh_ggf_hbb_hvvqqlnu_kl5_kt1],
        keys=[
            "chhh5",
        ],
        n_files=2,
        n_events=496495,
        aux={"custom": True},
    )


def get_dataset_lfns_2017(
    dataset_inst: od.Dataset,
    shift_inst: od.Shift,
    dataset_key: str,
) -> list[str]:
    """
    Custom method to obtain custom NanoAOD datasets
    """

    if not dataset_inst.x("custom", None):
        return GetDatasetLFNs.get_dataset_lfns_dasgoclient(
            GetDatasetLFNs, dataset_inst=dataset_inst, shift_inst=shift_inst, dataset_key=dataset_key,
        )
    print("dataset name:", dataset_inst.name)
    print("dataset_key:", dataset_key)

    # NOTE: this currently simply takes samples from a hard-coded local path. Should be improved
    #       when all files are stored somewhere remote
    lfn_base = law.LocalDirectoryTarget(
        f"/nfs/dust/cms/user/paaschal/WorkingArea/MCProduction/sgnl_production/NanoAODs/{dataset_key}/",
        fs="local_fs",
    )

    # loop though files and interpret paths as lfns
    return [
        lfn_base.child(basename, type="f").path
        for basename in lfn_base.listdir(pattern="*.root")
    ]
