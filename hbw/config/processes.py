# coding: utf-8

"""
Configuration of the Run 2 HH -> bbWW processes.
"""

import cmsdb
import order as od

from scinum import Number

from cmsdb.util import add_decay_process
from columnflow.util import DotDict

from hbw.config.styling import color_palette


def add_parent_process(config: od.Config, child_procs: list[od.Process], **kwargs):
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
        valid_ecms = set.intersection(*[set(proc.xsecs.keys()) for proc in child_procs])
        proc_kwargs["xsecs"] = {ecm: sum([proc.get_xsec(ecm) for proc in child_procs]) for ecm in valid_ecms}

    parent_process = config.add_process(**proc_kwargs)

    # add child processes to parent
    for child_proc in child_procs:
        parent_process.add_process(child_proc)

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
    # add main HH process
    config.add_process(cmsdb.processes.hh_ggf.copy())

    # Set dummy xsec for all processes if missing
    add_dummy_xsecs(config)

    # QCD process customization
    qcd_mu = config.get_process("qcd_mu", default=None)
    if qcd_mu:
        qcd_mu = "QCD Muon enriched"

    # add custom qcd_ele process
    qcd_em = config.get_process("qcd_em", default=None)
    qcd_bctoe = config.get_process("qcd_bctoe", default=None)
    if qcd_em and qcd_bctoe:
        qcd_ele = add_parent_process(  # noqa
            config,
            [qcd_em, qcd_bctoe],
            name="qcd_ele",
            id=31199,
            label="QCD Electron enriched",
        )
    elif qcd_em:
        qcd_ele = add_parent_process(  # noqa
            config,
            [qcd_em],
            name="qcd_ele",
            id=31199,
            label="QCD Electron enriched",
        )

    # custom v_lep process for ML Training, combining W+DY
    w_lnu = config.get_process("w_lnu", default=None)
    dy = config.get_process("dy", default=None)
    if w_lnu and dy:
        v_lep = add_parent_process(  # noqa
            config,
            [w_lnu, dy],
            name="v_lep",
            id=64575573,  # random number
            label="W and DY",
        )

    # Custom t_bkg process for ML Training, combining tt+st
    st = config.get_process("st", default=None)
    tt = config.get_process("tt", default=None)
    if st and tt:
        t_bkg = add_parent_process(  # noqa
            config,
            [st, tt],
            name="t_bkg",
            id=97842611,  # random number
            label="tt + st",
        )

    if config.has_tag("is_dl") and config.has_tag("is_nonresonant") and config.x.run == 2:
        # Custom signal  process for ML Training, combining multiple kl signal samples
        # NOTE: only built for run 2 because kl variations are missing in run 3
        signal_processes = [
            config.get_process(f"hh_ggf_hbb_hvv2l2nu_kl{kl}_kt1", deep=True)
            for kl in [0, 1, "2p45"]
        ]
        sig = config.add_process(
            name="sig",
            id=75835213,  # random number
            xsecs={
                13: sum([proc.get_xsec(13) for proc in signal_processes]),
            },
            label="signal",
        )
        for proc in signal_processes:
            try:
                sig.add_process(proc)
            except Exception:
                # this also adds 'sig' as parent to 'proc', but sometimes this is happening
                # multiple times, since we create multiple configs
                pass

    # add auxiliary information if process is signal
    for proc_inst, _, _ in config.walk_processes():
        is_signal = any([
            signal_tag in proc_inst.name
            for signal_tag in ("hh_vbf", "hh_ggf", "radion", "gravition")
        ])
        if is_signal:
            proc_inst.add_tag("is_signal")

    decay_map = {
        "lf": {
            "name": "lf",
            "id": 50,
            "label": "(lf)",
            "br": -1,
        },
        "hf": {
            "name": "hf",
            "id": 60,
            "label": "(hf)",
            "br": -1,
        },
    }

    # add heavy flavour and light flavour dy processes
    for proc in (
        "dy",
        "dy_m4to10", "dy_m10to50",
        "dy_m50toinf",
        "dy_m50toinf_0j", "dy_m50toinf_1j", "dy_m50toinf_2j",
    ):
        dy_proc_inst = config.get_process(proc, default=None)
        if dy_proc_inst:
            add_production_mode_parent = proc != "dy"
            for flavour in ("hf", "lf"):
                # the 'add_decay_process' function helps us to create all parent-daughter relationships
                add_decay_process(
                    dy_proc_inst,
                    decay_map[flavour],
                    add_production_mode_parent=add_production_mode_parent,
                    name_func=lambda parent_name, decay_name: f"{parent_name}_{decay_name}",
                    label_func=lambda parent_label, decay_label: f"{parent_label} {decay_label}",
                    xsecs=None,
                    aux={"flavour": flavour},
                )

    # create main background process
    background = config.add_process(
        name="background",
        id=99999,
        label="background",
        color=color_palette["blue"],
    )
    for bg in ["tt", "dy", "st", "vv", "w_lnu", "h"]:
        if config.has_process(bg):
            bg = config.get_process(bg)
            background.add_process(bg)


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
