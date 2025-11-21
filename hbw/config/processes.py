# coding: utf-8

"""
Configuration of the Run 2 HH -> bbWW processes.
"""

import order as od
import law

from scinum import Number
from hbw.config.styling import color_palette

logger = law.logger.get_logger(__name__)


def create_parent_process(child_proces: list[od.Process], **kwargs):
    """
    Helper function to create processes from multiple processes *child_procs*
    """
    if "id" not in kwargs:
        raise ValueError("A field 'id' is required to create a process")
    if "name" not in kwargs:
        raise ValueError("A field 'name' is required to create a process")

    proc_kwargs = kwargs.copy()

    if "xsecs" not in kwargs:
        # set the xsec as sum of all xsecs when the ecm key exists for all processes
        valid_ecms = set.intersection(*[set(proc.xsecs.keys()) for proc in child_proces])
        proc_kwargs["xsecs"] = {ecm: sum([proc.get_xsec(ecm) for proc in child_proces]) for ecm in valid_ecms}

    parent_process = od.Process(**proc_kwargs)

    # add child processes to parent
    for child_proc in child_proces:
        parent_process.add_process(child_proc)

    return parent_process


def add_parent_process(config: od.Config, child_procs: list[od.Process], **kwargs):
    """
    Helper function to create a parent process and add it to the config instance
    """
    parent_process = config.add_process(create_parent_process(child_procs, **kwargs))
    return parent_process


def add_dummy_xsecs(config: od.Config, dummy_xsec: float = 0.1):
    """ Helper that adds some dummy  xsecs when missing for the campaign's correspondign ecm """
    ecm = config.campaign.ecm

    process_insts = [
        process_inst
        for process_inst, _, _ in config.walk_processes()
        if process_inst.is_mc
    ]
    for process_inst in process_insts:
        if not process_inst.xsecs.get(ecm, None):
            # print(f"TODO: xsecs for {ecm} TeV, process {process_inst.name}")
            process_inst.xsecs[ecm] = Number(dummy_xsec)

    # # temporary xsecs from XSDB
    # config.get_process("dy").xsecs[13.6] = Number(67710.0)  # https://xsdb-temp.app.cern.ch/xsdb/?columns=37814272&currentPage=0&pageSize=10&searchQuery=DAS%3DWtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8  # noqa
    # config.get_process("w_lnu").xsecs[13.6] = Number(5558.0)  # https://xsdb-temp.app.cern.ch/xsdb/?columns=37814272&currentPage=0&ordDirection=1&ordFieldName=process_name&pageSize=10&searchQuery=DAS%3DWtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8  # noqa

    # temporary xsecs that were missing in xsdb
    # for proc in ("qcd_mu_pt170to300", "qcd_mu_pt470to600", "qcd_mu_pt1000"):
    #     proc_inst = config.get_process(proc)
    #     proc_inst.set_xsec(13.6, proc_inst.get_xsec(13))


def configure_hbw_processes(config: od.Config):
    """
    Function to modify the processes present in the config instance.
    NOTE: we should not rely on modifying process instances themselves as part of the config initialization.
    """
    # add main HH process
    # config.add_process(config.x.procs.n.hh_ggf)

    config.add_process(config.x.procs.n.t_bkg)
    config.add_process(config.x.procs.n.v_lep)
    config.add_process(config.x.procs.n.background)
    config.add_process(config.x.procs.n.other)
    minor = config.add_process(config.x.procs.n.minor)
    minor.label = "minor"

    color, sub_id = {
        "2022preEE": (color_palette["blue"], 1),
        "2022postEE": (color_palette["yellow"], 2),
        "2023preBPix": (color_palette["red"], 3),
        "2023postBPix": (color_palette["grey"], 4),
        "2024": (color_palette["black"], 5),
    }[config.x.cpn_tag]

    config.x.procs.n.background.add_process(
        name=f"bkg_{config.x.cpn_tag}",
        id=900000 + sub_id,
        label=f"Background {config.x.cpn_tag}",
        color=color,
        processes=config.x.procs.n.background.processes,
        aux={"unstack": True},
    )
    config.x.procs.n.data.add_process(
        name=f"data_{config.x.cpn_tag}",
        id=900010 + sub_id,
        label=f"Data {config.x.cpn_tag}",
        color=color,
        processes=config.x.procs.n.data.processes,
        aux={"unstack": True},
    )

    # Set dummy xsec for all processes if missing
    add_dummy_xsecs(config)

    # check that all process ids are unique
    unique_process_sanity_check(config)


def unique_process_sanity_check(config: od.Config):
    """
    Helper function to check that all process ids are unique in the config.
    Raises a ValueError if multiple processes with the same id are found.

    Note: this sanity check does not check for the uniqueness of process instances
    (i.e. multiple instances of the same process with different settings).
    """
    from collections import defaultdict
    proc_map = defaultdict(set)
    for proc, _, _ in config.walk_processes():
        proc_map[proc.id].add(proc.name)

    for proc_id, proc_list in proc_map.items():
        if len(proc_list) > 1:
            raise ValueError(
                f"Multiple processes with the same id {proc_id} found: "
                f"{', '.join([proc_name for proc_name in proc_list])}. "
                "Please ensure that process ids are unique. Note: you might have to re-run the "
                "Campaign creation after modifying the processes: \n"
                f"law run hbw.BuildCampaignSummary --config {config.name} --remove-output 0,a,y",
            )


def set_proc_attr(proc_inst, attr, value):
    if attr in ("id", "name"):
        raise ValueError(f"Setting {attr} via `set_proc_attr` helper is not allowed")
    elif attr == "label":
        setattr(proc_inst, attr, f"${value}$")
    elif attr == "color":
        setattr(proc_inst, attr, value)
    else:
        proc_inst.set_aux(attr, value)


def apply_proc_settings(proc_inst, process_settings):
    for attr, value in process_settings.items():
        set_proc_attr(proc_inst, attr, value)


def prepare_ml_processes(config_inst: od.Config, train_nodes, sub_process_class_factors):
    # fallbacks
    default_process_settings = {
        "weighting": None,
        "ml_id": -1,
    }

    for proc_name, process_settings in train_nodes.items():
        process_settings = law.util.merge_dicts(default_process_settings, process_settings)

        if process_settings["ml_id"] == -1:
            logger.warning("ml_id for process {proc_name} set to '-1'; will not be used in training")

        sub_processes = process_settings.pop("sub_processes", None)

        if config_inst.has_process(proc_name):
            logger.debug(f"update process {proc_name}")
            if sub_processes:
                raise NotImplementedError("Cannot re-assign sub-processes to already existing Process {proc_name}")
            proc_inst = config_inst.get_process(proc_name)
            apply_proc_settings(proc_inst, process_settings)
            set_proc_attr(proc_inst, "sub_process_class_factor", sub_process_class_factors.get(proc_name, 1))
        else:
            logger.debug(f"create new process {proc_name}")
            proc_id = process_settings.pop("id", int(1e7) + law.util.create_hash(proc_name, l=6, to_int=True))
            sub_process_insts = []
            for proc in sub_processes:
                if not config_inst.has_process(proc):
                    raise ValueError(
                        f"Trying to create parent process {proc_name}, but requested child {proc} "
                        f"is not included in config {config_inst.name}",
                    )
                sub_process_inst = config_inst.get_process(proc, default=None)

                # assign attributes to sub process insts
                set_proc_attr(sub_process_inst, "ml_id", process_settings["ml_id"])
                set_proc_attr(sub_process_inst, "sub_process_class_factor", sub_process_class_factors.get(proc, 1))
                sub_process_insts.append(sub_process_inst)

            # create parent process and assign attributes of relevance
            proc_inst = add_parent_process(
                config_inst,
                sub_process_insts,
                name=proc_name,
                id=proc_id,
            )
            apply_proc_settings(proc_inst, process_settings)
            set_proc_attr(sub_process_inst, "sub_process_class_factor", sub_process_class_factors.get(proc, 1))
