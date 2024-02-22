# coding: utf-8

"""
Configuration of the Run 2 HH -> bbWW processes.
"""

import order as od

from scinum import Number


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
    # config.get_process("dy_lep").xsecs[13.6] = Number(67710.0)  # https://xsdb-temp.app.cern.ch/xsdb/?columns=37814272&currentPage=0&pageSize=10&searchQuery=DAS%3DWtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8  # noqa
    # config.get_process("w_lnu").xsecs[13.6] = Number(5558.0)  # https://xsdb-temp.app.cern.ch/xsdb/?columns=37814272&currentPage=0&ordDirection=1&ordFieldName=process_name&pageSize=10&searchQuery=DAS%3DWtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8  # noqa

    # temporary xsecs that were missing in xsdb
    # for proc in ("qcd_mu_pt170to300", "qcd_mu_pt470to600", "qcd_mu_pt1000"):
    #     proc_inst = config.get_process(proc)
    #     proc_inst.set_xsec(13.6, proc_inst.get_xsec(13))


def configure_hbw_processes(config: od.Config):
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
    w_lnu = config.get_process("w_lnu")
    dy_lep = config.get_process("dy_lep")
    if w_lnu and dy_lep:
        v_lep = add_parent_process(  # noqa
            config,
            [w_lnu, dy_lep],
            name="v_lep",
            id=64575573,  # random number
            label="W and DY",
        )

    # Custom t_bkg process for ML Training, combining tt+st
    st = config.get_process("st")
    tt = config.get_process("tt")
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
            config.get_process(f"ggHH_kl_{kl}_kt_1_dl_hbbhww")
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
            for signal_tag in ("qqHH", "ggHH", "radion", "gravition")
        ])
        if is_signal:
            proc_inst.add_tag("is_signal")
