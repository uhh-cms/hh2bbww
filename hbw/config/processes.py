# coding: utf-8

"""
Configuration of the Run 2 HH -> bbWW processes.
"""

import order as od

from scinum import Number

from columnflow.util import DotDict


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
    config.add_process(config.x.procs.n.hh_ggf)

    config.add_process(config.x.procs.n.t_bkg)
    config.add_process(config.x.procs.n.v_lep)
    config.add_process(config.x.procs.n.background)

    # Set dummy xsec for all processes if missing
    add_dummy_xsecs(config)


from random import randint


def create_combined_proc_forML(config: od.Config, proc_name: str, proc_dict: dict, color=None):

    combining_proc = []
    for proc in proc_dict.sub_processes:
        combining_proc.append(config.get_process(proc, default=None))
    proc_name = add_parent_process(config,
        combining_proc,
        name=proc_name,
        id=randint(10000000, 99999999),
        # TODO: random number (could by chance be a already used number --> should be checked)
        label=proc_dict.get("label", "combined custom process"),
        color=proc_dict.get("color", None),
    )
    ml_config = DotDict({"weighting": proc_dict.get("weighting", None), "sub_processes": proc_dict.sub_processes})
    proc_name.x.ml_config = ml_config
