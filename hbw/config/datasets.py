# coding: utf-8

"""
Configuration of the Run 2 HH -> bbWW datasets.
"""

from __future__ import annotations

import itertools

import law
import order as od

import cmsdb.processes as procs
from columnflow.util import DotDict
from columnflow.tasks.external import GetDatasetLFNs
from columnflow.config_util import get_root_processes_from_campaign


logger = law.logger.get_logger(__name__)


#
# Collection of process names and their corresponding datasets for each year
#


data_mu = {
    "2017": [
        "data_mu_b",
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
        "data_e_b",
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
        "st_tchannel_t_powheg",
        "st_tchannel_tbar_powheg",
        "st_twchannel_t_powheg",
        "st_twchannel_tbar_powheg",
        "st_schannel_lep_amcatnlo",
        "st_schannel_had_amcatnlo",
    ],
    "2022preEE": [
        "st_tchannel_t_powheg",
        "st_tchannel_tbar_powheg",
        "st_twchannel_t_sl_powheg",
        "st_twchannel_tbar_sl_powheg",
        "st_twchannel_t_dl_powheg",
        "st_twchannel_tbar_dl_powheg",
        "st_twchannel_t_fh_powheg",
        "st_twchannel_tbar_fh_powheg",
        # "st_schannel_lep_amcatnlo",
        # "st_schannel_had_amcatnlo",
    ],
    "2022postEE": [
        "st_tchannel_t_powheg",
        "st_tchannel_tbar_powheg",
        "st_twchannel_t_sl_powheg",
        "st_twchannel_tbar_sl_powheg",
        "st_twchannel_t_dl_powheg",
        "st_twchannel_tbar_dl_powheg",
        "st_twchannel_t_fh_powheg",
        "st_twchannel_tbar_fh_powheg",
        # "st_schannel_lep_amcatnlo",
        # "st_schannel_had_amcatnlo",
    ],
}

w_lnu = {
    "2017": [
        "w_lnu_ht70To100_madgraph",
        "w_lnu_ht100To200_madgraph",
        "w_lnu_ht200To400_madgraph",
        "w_lnu_ht400To600_madgraph",
        "w_lnu_ht600To800_madgraph",
        "w_lnu_ht800To1200_madgraph",
        "w_lnu_ht1200To2500_madgraph",
        "w_lnu_ht2500_madgraph",
    ],
    "2022postEE": [
        "w_lnu_amcatnlo",
    ],
    "2022preEE": [
        "w_lnu_amcatnlo",
    ],
}

dy_lep = {
    "2017": [
        "dy_lep_m50_ht70to100_madgraph",
        "dy_lep_m50_ht100to200_madgraph",
        "dy_lep_m50_ht200to400_madgraph",
        "dy_lep_m50_ht400to600_madgraph",
        "dy_lep_m50_ht600to800_madgraph",
        "dy_lep_m50_ht800to1200_madgraph",
        "dy_lep_m50_ht1200to2500_madgraph",
        "dy_lep_m50_ht2500_madgraph",
    ],
    "2022postEE": [
        "dy_lep_m50_madgraph",
    ],
    "2022preEE": [
        "dy_lep_m50_madgraph",
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
        "qcd_mu_pt1000_pythia",
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
        "qcd_mu_pt1000_pythia",
    ],
    "2022preEE": [
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
        "qcd_mu_pt1000_pythia",
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
        "qcd_em_pt300toInf_pythia",
    ],
    "2022postEE": [
        # "qcd_em_pt10to30_pythia",  # missing process + probably empty anyways
        # "qcd_em_pt30to50_pythia",  # empty after selection
        "qcd_em_pt50to80_pythia",
        "qcd_em_pt80to120_pythia",
        "qcd_em_pt120to170_pythia",
        "qcd_em_pt170to300_pythia",
        "qcd_em_pt300toInf_pythia",
    ],
    "2022preEE": [
        # "qcd_em_pt10to30_pythia",  # missing process + probably empty anyways
        "qcd_em_pt30to50_pythia",
        "qcd_em_pt50to80_pythia",
        "qcd_em_pt80to120_pythia",
        "qcd_em_pt120to170_pythia",
        "qcd_em_pt170to300_pythia",
        "qcd_em_pt300toInf_pythia",
    ],
}

qcd_bctoe = {
    "2017": [
        "qcd_bctoe_pt15to20_pythia",
        "qcd_bctoe_pt20to30_pythia",
        "qcd_bctoe_pt30to80_pythia",
        "qcd_bctoe_pt80to170_pythia",
        "qcd_bctoe_pt170to250_pythia",
        "qcd_bctoe_pt250toInf_pythia",
    ],
    "2022postEE": [
        # empty for now
    ],
}

single_h = {
    "2017": [
        # empty for now
    ],
    "2022postEE": [
        # empty for now
    ],
}

# cross sections still missing
vv = {
    "2017": [
        # empty for now
    ],
    # "2022preEE": [
    #     "ww_pythia",
    #     "wz_pythia",
    #     "zz_pythia",
    # ],
    # "2022postEE": [
    #     "ww_pythia",
    #     "wz_pythia",
    #     "zz_pythia",
    # ],
}

ttv = {
    "2017": [
        # empty for now
    ],
    "2022postEE": [
        # empty for now
    ],
}

ggHH_sl_hbbhww = {
    "2017": [
        "ggHH_kl_0_kt_1_sl_hbbhww_powheg",
        "ggHH_kl_1_kt_1_sl_hbbhww_powheg",
        "ggHH_kl_2p45_kt_1_sl_hbbhww_powheg",
        "ggHH_kl_5_kt_1_sl_hbbhww_powheg",
    ],
    "2022preEE": [
        "ggHH_kl_1_kt_1_sl_hbbhww_powheg",
    ],
    "2022postEE": [
        "ggHH_kl_1_kt_1_sl_hbbhww_powheg",
    ],
}

ggHH_dl_hbbhww = {
    "2017": [
        "ggHH_kl_0_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_1_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_2p45_kt_1_dl_hbbhww_powheg",
        "ggHH_kl_5_kt_1_dl_hbbhww_powheg",
    ],
    "2022preEE": [
        "ggHH_kl_1_kt_1_dl_hbbhww_powheg",
    ],
    "2022postEE": [
        "ggHH_kl_1_kt_1_dl_hbbhww_powheg",
    ],
}

qqHH_sl_hbbhww = {
    "2017": [
        "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww_madgraph",
        "qqHH_CV_1_C2V_1_kl_0_sl_hbbhww_madgraph",
        "qqHH_CV_1_C2V_1_kl_2_sl_hbbhww_madgraph",
        "qqHH_CV_1_C2V_0_kl_1_sl_hbbhww_madgraph",
        "qqHH_CV_1_C2V_2_kl_1_sl_hbbhww_madgraph",
        "qqHH_CV_0p5_C2V_1_kl_1_sl_hbbhww_madgraph",
        "qqHH_CV_1p5_C2V_1_kl_1_sl_hbbhww_madgraph",
    ],
    "2022postEE": [
        # empty for now
    ],
}

qqHH_dl_hbbhww = {
    "2017": [
        "qqHH_CV_1_C2V_1_kl_1_dl_hbbhww_madgraph",
        "qqHH_CV_1_C2V_1_kl_0_dl_hbbhww_madgraph",
        "qqHH_CV_1_C2V_1_kl_2_dl_hbbhww_madgraph",
        "qqHH_CV_1_C2V_0_kl_1_dl_hbbhww_madgraph",
        "qqHH_CV_1_C2V_2_kl_1_dl_hbbhww_madgraph",
        "qqHH_CV_0p5_C2V_1_kl_1_dl_hbbhww_madgraph",
        "qqHH_CV_1p5_C2V_1_kl_1_dl_hbbhww_madgraph",
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
        # empty for now
    ],
}


def get_dataset_names(cpn_tag: int | str, as_list: bool = False) -> DotDict[str: list[str]] | list[str]:
    """
    Central definition of datasets used in the hbbhww analysis based on the *cpn_tag*.
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
        tt=tt.get(cpn_tag, []),
        st=st.get(cpn_tag, []),
        w_lnu=w_lnu.get(cpn_tag, []),
        dy_lep=dy_lep.get(cpn_tag, []),
        qcd_mu=qcd_mu.get(cpn_tag, []),
        qcd_em=qcd_em.get(cpn_tag, []),
        qcd_bctoe=qcd_bctoe.get(cpn_tag, []),
        single_h=single_h.get(cpn_tag, []),
        vv=vv.get(cpn_tag, []),
        ttv=ttv.get(cpn_tag, []),
        ggHH_sl_hbbhww=ggHH_sl_hbbhww.get(cpn_tag, []),
        ggHH_dl_hbbhww=ggHH_dl_hbbhww.get(cpn_tag, []),
        qqHH_sl_hbbhww=qqHH_sl_hbbhww.get(cpn_tag, []),
        qqHH_dl_hbbhww=qqHH_dl_hbbhww.get(cpn_tag, []),
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
    # optionally switch to custom signal processes (only implemented for ggHH_sl)
    if config.has_tag("custom_signals"):
        dataset_names.ggHH_sl_hbbhww = [dataset_name.replace("powheg", "custom") for dataset_name in dataset_names]

    if not config.has_tag("is_resonant"):
        # remove all resonant signal processes/datasets
        dataset_names.pop("graviton_hh_ggf_bbww")
        dataset_names.pop("radion_hh_ggf_bbww")

    if not config.has_tag("is_nonresonant"):
        # remove all nonresonant signal processes/datasets
        for hh_proc in ("ggHH_sl_hbbhww", "ggHH_dl_hbbhww", "qHH_sl_hbbhww", "qqHH_dl_hbbhww"):
            dataset_names.pop(hh_proc)

    return dataset_names


def add_hbw_processes_and_datasets(config: od.Config, campaign: od.Campaign):
    if campaign.x.year == 2017:
        # load custom produced datasets into campaign (2017 only!)
        get_custom_hh_2017_datasets(campaign)

        # use custom get_dataset_lfns function
        config.x.get_dataset_lfns = get_dataset_lfns_2017

    dataset_names = get_dataset_names_for_config(config)

    # get all process names with corresponding datasets
    process_names = [proc_name for proc_name, _dataset_names in dataset_names.items() if _dataset_names]

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # add processes to config
    for proc_name in process_names:
        config.add_process(procs.n(proc_name))
    # loop over all dataset names and add them to the config
    for dataset_name in list(itertools.chain(*dataset_names.values())):
        config.add_dataset(campaign.get_dataset(dataset_name))


def configure_hbw_datasets(config: od.Config, limit_dataset_files: int | None = None):
    for dataset in config.datasets:
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
        elif config.campaign.x.year == 2017:
            # our default Run2 signal samples are EOY, so we have to skip golden json, certain met filter
            dataset.add_tag("is_eoy")


def get_custom_hh_2017_datasets(
    campaign: od.Campaign,
) -> None:
    """
    Add custom HH datasets to campaign
    """
    campaign.add_dataset(
        name="ggHH_kl_0_kt_1_sl_hbbhww_custom",
        id=10 ** 8 + 14057341,
        processes=[procs.ggHH_kl_0_kt_1_sl_hbbhww],
        keys=[
            "chhh0",
        ],
        n_files=2,
        n_events=493996,
        aux={"custom": True},
    )

    campaign.add_dataset(
        name="ggHH_kl_1_kt_1_sl_hbbhww_custom",
        id=10 ** 8 + 14065482,
        processes=[procs.ggHH_kl_1_kt_1_sl_hbbhww],
        keys=[
            "chhh1",
        ],
        n_files=2,
        n_events=498499,
        aux={"custom": True},
    )

    campaign.add_dataset(
        name="ggHH_kl_2p45_kt_1_sl_hbbhww_custom",
        id=10 ** 8 + 14066581,
        processes=[procs.ggHH_kl_2p45_kt_1_sl_hbbhww],
        keys=[
            "chhh2p45",
        ],
        n_files=2,
        n_events=498496,
        aux={"custom": True},
    )

    campaign.add_dataset(
        name="ggHH_kl_5_kt_1_sl_hbbhww_custom",
        id=10 ** 8 + 14058363,
        processes=[procs.ggHH_kl_5_kt_1_sl_hbbhww],
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
