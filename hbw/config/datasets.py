# coding: utf-8

"""
Configuration of the Run 2 HH -> bbWW datasets.
"""

from __future__ import annotations

import law
import order as od

import cmsdb.processes as procs
from columnflow.tasks.external import GetDatasetLFNs


logger = law.logger.get_logger(__name__)


def add_hbw_datasets(config: od.Config, campaign: od.Campaign):
    # load custom produced datasets into campaign
    get_custom_hh_datasets(campaign)

    # use custom get_dataset_lfns function
    config.x.get_dataset_lfns = get_dataset_lfns

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

    if config.has_tag("is_sl") and config.has_tag("is_nonresonant"):
        # non-resonant HH -> bbWW(qqlnu) Signal
        if config.has_tag("custom_signals"):
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

    if config.has_tag("is_dl") and config.has_tag("is_nonresonant"):
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
    if config.has_tag("is_sl") and config.has_tag("is_resonant"):
        for mass in config.x.graviton_masspoints:
            dataset_names.append(f"graviton_hh_ggf_bbww_m{mass}_madgraph")
        for mass in config.x.radion_masspoints:
            dataset_names.append(f"radion_hh_ggf_bbww_m{mass}_madgraph")

    if config.has_tag("is_dl") and config.has_tag("is_resonant"):
        logger.warning(
            f"For analysis {config.analysis.name}: dileptonic resonant samples still needs to be implemented",
        )

    # loop over all dataset names and add them to the config
    for dataset_name in dataset_names:
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
        else:
            # our default Run2 signal samples are EOY, so we have to skip golden json, certain met filter
            dataset.add_tag("is_eoy")


def get_custom_hh_datasets(
    campaign: od.Campaign,
) -> None:
    """
    Add custom HH datasets to campaign
    """
    # TODO: change the n_files and n_events if available

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


def get_dataset_lfns(
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
