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


def hbw_dataset_names(config: od.Config, as_list: bool = False) -> DotDict[str: list[str]] | list[str]:
    # define data streams based on the run
    # NOTE: the order of the streams is important for data trigger filtering (removing double counting)
    nano_version = config.campaign.x.version
    if config.campaign.x.run == 2:
        config.x.data_streams = ["mu", "e"]
    elif config.campaign.x.run == 3 and nano_version == 14:
        config.x.data_streams = ["mu", "e", "muoneg"]
    elif config.campaign.x.run == 3:
        config.x.data_streams = ["mu", "egamma", "muoneg"]

    data_eras = {
        "2017": list("cdef"),
        "2022preEE": list("cd"),
        "2022postEE": list("efg"),
        "2023preBPix": ["c1", "c2", "c3", "c4"],
        "2023postBPix": ["d1", "d2"],
    }[config.x.cpn_tag]

    data_datasets = [
        f"data_{stream}_{era}"
        for era in data_eras
        for stream in config.x.data_streams
    ]

    # Add additional datasets needed for orthogonal efficiency measurements.
    data_jethtmet_eras = {
        "2022preEE": "cd",
        "2022postEE": "efg",
        "2023preBPix": "",
        "2023postBPix": "",  # TODO: add 2023 jetmet datasets in cmsdb
    }[config.x.cpn_tag]

    data_jethtmet_datasets = [
        f"data_jethtmet_{era}"
        for era in data_jethtmet_eras
    ]

    ggf_samples = lambda hhdecay: [
        f"hh_ggf_{hhdecay}_kl0_kt1_powheg",
        f"hh_ggf_{hhdecay}_kl1_kt1_powheg",
        f"hh_ggf_{hhdecay}_kl2p45_kt1_powheg",
        f"hh_ggf_{hhdecay}_kl5_kt1_powheg",
    ]
    vbf_samples = lambda hhdecay: [
        f"hh_vbf_{hhdecay}_kv1_k2v1_kl1_madgraph",
        f"hh_vbf_{hhdecay}_kv1_k2v0_kl1_madgraph",
        f"hh_vbf_{hhdecay}_kv1p74_k2v1p37_kl14p4_madgraph",
        f"hh_vbf_{hhdecay}_kvm0p012_k2v0p03_kl10p2_madgraph",
        f"hh_vbf_{hhdecay}_kvm0p758_k2v1p44_klm19p3_madgraph",
        f"hh_vbf_{hhdecay}_kvm0p962_k2v0p959_klm1p43_madgraph",
        f"hh_vbf_{hhdecay}_kvm1p21_k2v1p94_klm0p94_madgraph",
        f"hh_vbf_{hhdecay}_kvm1p6_k2v2p72_klm1p36_madgraph",
        f"hh_vbf_{hhdecay}_kvm1p83_k2v3p57_klm3p39_madgraph",
        f"hh_vbf_{hhdecay}_kvm2p12_k2v3p87_klm5p96_madgraph",
    ] if config.x.run == 3 else [
        f"hh_vbf_{hhdecay}_kv1_k2v1_kl1_madgraph",
        f"hh_vbf_{hhdecay}_kv1_k2v1_kl0_madgraph",
        f"hh_vbf_{hhdecay}_kv1_k2v1_kl2_madgraph",
        f"hh_vbf_{hhdecay}_kv1_k2v0_kl1_madgraph",
        f"hh_vbf_{hhdecay}_kv1_k2v2_kl1_madgraph",
        f"hh_vbf_{hhdecay}_kv0p5_k2v1_kl1_madgraph",
        f"hh_vbf_{hhdecay}_kv1p5_k2v1_kl1_madgraph",
    ]

    dataset_names = DotDict.wrap({
        # **data_datasets,
        "data": data_datasets,
        "data_jethtmet": config.x.if_era(cfg_tag="is_for_sf", values=data_jethtmet_datasets),
        "tt": ["tt_sl_powheg", "tt_dl_powheg", "tt_fh_powheg"],
        "st": [
            "st_schannel_tbar_lep_4f_amcatnlo",
            "st_schannel_t_lep_4f_amcatnlo",
            "st_tchannel_t_4f_powheg",
            "st_tchannel_tbar_4f_powheg",
            *config.x.if_era(run=2, values=[
                "st_twchannel_t_powheg",
                "st_twchannel_tbar_powheg",
            ]),
            *config.x.if_era(run=3, cfg_tag="is_sl", values=[
                "st_twchannel_t_fh_powheg",
                "st_twchannel_tbar_fh_powheg",
                "st_twchannel_t_sl_powheg",
                "st_twchannel_tbar_sl_powheg",
                "st_twchannel_t_dl_powheg",
                "st_twchannel_tbar_dl_powheg",
            ]),
            *config.x.if_era(run=3, cfg_tag="is_dl", values=[
                # "st_twchannel_t_fh_powheg",  # (almost) empty in DL
                # "st_twchannel_tbar_fh_powheg",  # (almost) empty in DL
                "st_twchannel_t_sl_powheg",
                "st_twchannel_tbar_sl_powheg",
                "st_twchannel_t_dl_powheg",
                "st_twchannel_tbar_dl_powheg",
            ]),
        ],
        "dy": [
            *config.x.if_era(run=2, values=[  # TODO: update to amcatnlo aswell
                "dy_m50toinf_ht70to100_madgraph",
                "dy_m50toinf_ht100to200_madgraph",
                "dy_m50toinf_ht200to400_madgraph",
                "dy_m50toinf_ht400to600_madgraph",
                "dy_m50toinf_ht600to800_madgraph",
                "dy_m50toinf_ht800to1200_madgraph",
                "dy_m50toinf_ht1200to2500_madgraph",
                "dy_m50toinf_ht2500toinf_madgraph",
            ]),
            *config.x.if_era(run=3, values=[
                # NLO samples
                "dy_m50toinf_amcatnlo",
                "dy_m10to50_amcatnlo",
                "dy_m4to10_amcatnlo",
                "dy_m50toinf_0j_amcatnlo",
                "dy_m50toinf_1j_amcatnlo",
                "dy_m50toinf_2j_amcatnlo",
            ]),
        ],
        "w_lnu": [
            *config.x.if_era(run=2, values=[  # TODO: update to amcatnlo aswell
                "w_lnu_ht70to100_madgraph",
                "w_lnu_ht100to200_madgraph",
                "w_lnu_ht200to400_madgraph",
                "w_lnu_ht400to600_madgraph",
                "w_lnu_ht600to800_madgraph",
                "w_lnu_ht800to1200_madgraph",
                "w_lnu_ht1200to2500_madgraph",
                "w_lnu_ht2500toinf_madgraph",
            ]),
            *config.x.if_era(run=3, values=[
                "w_lnu_amcatnlo",
            ]),
        ],
        "vv": [
            *config.x.if_era(run=3, values=[
                "ww_pythia",
                "wz_pythia",
                "zz_pythia",
            ]),
        ],
        "vvv": [
            "www_4f_amcatnlo",
            "wwz_4f_amcatnlo",
            "wzz_amcatnlo",
            "zzz_amcatnlo",
        ],
        "ttv": [
            # missing pdf weights in 2022postEE uhh samples
            "ttw_wlnu_amcatnlo",
            # "ttw_wqq_amcatnlo",  # not existing in 2022postEE uhh samples
            *config.x.if_era(run=3, values=[
                "ttz_zll_m4to50_amcatnlo",
                "ttz_zll_m50toinf_amcatnlo",
                "ttz_znunu_amcatnlo",
                "ttz_zqq_amcatnlo",
            ]),
        ],
        # NOTE: top + gamma is not used since it is already included in ttbar or single top samples
        # "ttg": [
        #     "ttg_pt10to100_amcatnlo",
        #     "ttg_pt100to200_amcatnlo",
        #     "ttg_pt200toinf_amcatnlo",
        # ],
        # "tg": ["tgqb_4f_amcatnlo"],
        "ttvv": [
            "ttww_madgraph",
            "ttwz_madgraph",
            "ttzz_madgraph",
        ],
        "tttt": ["tttt_amcatnlo"],
        "h": [
            *config.x.if_era(run=3, values=[
                # TODO: remove whatever is not really necessary
                # "h_ggf_hbb_powheg",  # empty in DL (< 0.01 events in postEE)
                "h_ggf_hww2l2nu_powheg",
                "h_vbf_hbb_powheg",
                "h_vbf_hww2l2nu_powheg",
                # "h_ggf_hzg_zll_powheg",  # probably empty in DL SR
                "zh_zqq_hbb_powheg",
                "zh_zll_hbb_powheg",
                # "zh_zll_hcc_powheg",  # 0.18 events in DL postEE analysis region
                "zh_hww2l2nu_powheg",
                "zh_gg_zll_hbb_powheg",
                "zh_gg_zqq_hbb_powheg",
                # "zh_gg_znunu_hbb_powheg",  # empty in DL (< 0.01 events in postEE)
                # "zh_gg_zll_hcc_powheg",  # 0.05 events in DL postEE analysis region
                # "wph_wqq_hbb_powheg",  # basically empty in DL (< 0.01 events in postEE)
                "wph_wlnu_hbb_powheg",
                # "wph_wqq_hcc_powheg",  # basically empty in DL (< 0.01 events in postEE)
                # "wph_wlnu_hcc_powheg",  # basically empty in DL (< 0.01 events in postEE)
                # "wph_hzg_zll_powheg",  # basically empty in DL (< 0.01 events in postEE)
                # "wmh_wqq_hbb_powheg",  # basically empty in DL (< 0.01 events in postEE)
                "wmh_wlnu_hbb_powheg",
                # "wmh_wqq_hcc_powheg",  # basically empty in DL (< 0.01 events in postEE)
                # "wmh_wlnu_hcc_powheg",  # basically empty in DL (< 0.01 events in postEE)
                # "wmh_hzg_zll_powheg",  # basically empty in DL (< 0.01 events in postEE)
                "tth_hbb_powheg",
                "tth_hnonbb_powheg",  # overlap with other samples, so be careful
                # TODO: preliminary cross sections for ttzh, ttwh
                "ttzh_madgraph",
                "ttwh_madgraph",
                # htt
                "h_ggf_htt_powheg",
                "h_vbf_htt_powheg",
                "zh_htt_powheg",
                "wph_htt_powheg",
                "wmh_htt_powheg",
                # thq, thw
                "thq_4f_madgraph",
                "thw_madgraph",
            ]),
        ],
        "hh_ggf": [
            *ggf_samples("hbb_hvvqqlnu"),
            *ggf_samples("hbb_hvv2l2nu"),
            *ggf_samples("hbb_htt"),
            *config.x.if_era(run=3, values=ggf_samples("hbb_hvv")),
        ],
        "hh_vbf": [
            *vbf_samples("hbb_hvvqqlnu"),
            *vbf_samples("hbb_hvv2l2nu"),
            *vbf_samples("hbb_htt"),
            *config.x.if_era(run=3, values=vbf_samples("hbb_hvv")),
        ],
        "graviton_hh_ggf_bbww": [
            *config.x.if_era(run=2, cfg_tag="is_resonant", values=[
                f"graviton_hh_ggf_bbww_m{mass}_madgraph"
                for mass in [
                    250, 260, 270, 280, 300, 320, 350, 400, 450, 500,
                    550, 600, 650, 700, 750, 800, 850, 900, 1000,
                    1250, 1500, 1750, 2000, 2500, 3000,
                ]
            ]),
        ],
        "radion_hh_ggf_bbww": [
            *config.x.if_era(run=2, cfg_tag="is_resonant", values=[
                f"radion_hh_ggf_bbww_m{mass}_madgraph"
                for mass in [
                    250, 260, 270, 280, 300, 320, 350, 400, 450, 500,
                    550, 600, 650, 700, 750, 800, 850, 900, 1000,
                    1250, 1500, 1750, 2000, 2500, 3000,
                ]
            ]),
        ],
        "qcd_mu": [
            *config.x.if_era(cfg_tag="is_sl", values=[
                # "qcd_mu_pt15to20_pythia",
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
            ]),
        ],
        "qcd_em": [
            *config.x.if_era(cfg_tag="is_sl", values=[
                # "qcd_em_pt15to20_pythia",
                # "qcd_em_pt20to30_pythia",
                "qcd_em_pt30to50_pythia",
                "qcd_em_pt50to80_pythia",
                "qcd_em_pt80to120_pythia",
                "qcd_em_pt120to170_pythia",
                "qcd_em_pt170to300_pythia",
                "qcd_em_pt300toinf_pythia",
            ]),
        ],
        "qcd_bctoe": [
            *config.x.if_era(run=2, cfg_tag="is_sl", values=[
                # "qcd_bctoe_pt15to20_pythia",
                "qcd_bctoe_pt20to30_pythia",
                "qcd_bctoe_pt30to80_pythia",
                "qcd_bctoe_pt80to170_pythia",
                "qcd_bctoe_pt170to250_pythia",
                "qcd_bctoe_pt250toinf_pythia",
            ]),
        ],

    })
    if as_list:
        return list(itertools.chain(*dataset_names.values()))

    return dataset_names


def get_dataset_names_for_config(config: od.Config, as_list: bool = False):
    """
    get all relevant dataset names and modify them based on the config and campaign
    """

    config.x.dataset_names = dataset_names = hbw_dataset_names(config)

    # optionally switch to custom signal processes (only implemented for hh_ggf_sl)
    if config.has_tag("custom_signals"):
        dataset_names.hh_ggf_hbb_hvvqqlnu = [dataset_name.replace("powheg", "custom") for dataset_name in dataset_names]

    if not config.has_tag("is_resonant"):
        # remove all resonant signal processes/datasets
        dataset_names.pop("graviton_hh_ggf_bbww", None)
        dataset_names.pop("radion_hh_ggf_bbww", None)

    if not config.has_tag("is_sl"):
        # remove qcd datasets from DL
        dataset_names.pop("qcd_mu", None)
        dataset_names.pop("qcd_em", None)
        dataset_names.pop("qcd_bctoe", None)

    if not config.has_tag("is_nonresonant"):
        # remove all nonresonant signal processes/datasets
        for hh_proc in ("hh_ggf", "hh_vbf"):
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
    # if config.x.cpn_tag == "2022postEE":
    #     add_synchronization_dataset(config)

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
    missing_datasets = set()
    for dataset_name in list(itertools.chain(*dataset_names.values())):
        if campaign.has_dataset(dataset_name):
            config.add_dataset(campaign.get_dataset(dataset_name))
        else:
            logger.warning(
                f"Dataset '{dataset_name}' not found in config '{config.name}', "
                "skipping it. Please check the campaign configuration.",
            )
            missing_datasets.add(dataset_name)
    if missing_datasets:
        print(f"Missing datasets in config {config.name}:\n" + "\n".join(sorted(missing_datasets)))


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
    # allow usage of UHH campaign
    enable_uhh_campaign_usage(config)

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

        if dataset.name.startswith("tt_"):
            dataset.add_tag("is_ttbar")

        if dataset.name.startswith("dy_"):
            dataset.add_tag("is_v_jets")
            dataset.add_tag("is_z_jets")
            dataset.add_tag("is_dy")
        if dataset.name.startswith("w_"):
            dataset.add_tag("is_v_jets")
            dataset.add_tag("is_w_jets")

        if dataset.name.startswith("qcd"):
            dataset.add_tag("is_qcd")
        if "hh" in dataset.name:
            dataset.add_tag("is_hh")
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

        if (
            dataset.name.endswith("_pythia") or
            "hh_vbf" in dataset.name or
            dataset.name == "ttw_wlnu_amcatnlo" or
            dataset.name == "zzz_amcatnlo"  # due to one broken file in 2022postEE uhh samples
        ):
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
        f"/data/dust/user/paaschal/WorkingArea/MCProduction/sgnl_production/NanoAODs/{dataset_key}/",
        fs="local_fs",
    )

    # loop though files and interpret paths as lfns
    return [
        lfn_base.child(basename, type="f").path
        for basename in lfn_base.listdir(pattern="*.root")
    ]


def enable_uhh_campaign_usage(cfg: od.Config) -> None:
    # custom lfn retrieval method in case the underlying campaign is custom uhh
    def get_dataset_lfns_uhh(
        dataset_inst: od.Dataset,
        shift_inst: od.Shift,
        dataset_key: str,
    ) -> list[str]:
        if "uhh" not in dataset_inst.x("campaign", ""):
            # for non-uhh datasets, use default GetDatasetLFNs method
            return GetDatasetLFNs.get_dataset_lfns_dasgoclient(
                GetDatasetLFNs, dataset_inst=dataset_inst, shift_inst=shift_inst, dataset_key=dataset_key,
            )
        cpn_name = dataset_inst.x.campaign
        # destructure dataset_key into parts and create the lfn base directory
        dataset_id, full_campaign, tier = dataset_key.split("/")[1:]
        main_campaign, sub_campaign = full_campaign.split("-", 1)
        lfn_base = law.wlcg.WLCGDirectoryTarget(
            f"/store/{dataset_inst.data_source}/{main_campaign}/{dataset_id}/{tier}/{sub_campaign}/0",
            # fs=f"wlcg_fs_{cfg.campaign.x.custom['name']}",
            fs=f"wlcg_fs_{cpn_name}",
        )

        broken_files = dataset_inst[shift_inst.name].get_aux("broken_files", [])

        # loop though files and interpret paths as lfns
        lfns = [
            lfn_base.child(basename, type="f").path
            for basename in lfn_base.listdir(pattern="*.root")
        ]
        return [lfn for lfn in lfns if lfn not in broken_files]

    if any("uhh" in cpn_name for cpn_name in cfg.campaign.x("campaigns", [])):
        # define the lfn retrieval function
        cfg.x.get_dataset_lfns = get_dataset_lfns_uhh

        # define custom remote fs's to look at
        cfg.x.get_dataset_lfns_remote_fs = lambda dataset_inst: (
            None if "uhh" not in dataset_inst.x("campaign", "") else [
                f"wlcg_fs_{dataset_inst.x.campaign}",
                f"local_fs_{dataset_inst.x.campaign}",
            ])
