# coding: utf-8

import law
import order as od

import cmsdb.processes as procs
from columnflow.tasks.external import GetDatasetLFNs


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
