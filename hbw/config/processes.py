# coding: utf-8

"""
Configuration of the Run 2 HH -> bbWW processes.
"""

import order as od


def get_process_names_for_config(config: od.Config):
    process_names = [
        "data",
        "data_mu",
        "data_e",
        "tt",
        "st",
        "w_lnu",
        "dy_lep",
        "qcd",
        "qcd_mu",
        "qcd_em",
        "qcd_bctoe",
        "ttv",
        "vv",
        "h",
    ]

    if config.has_tag("is_resonant"):
        # add resonant signal processes/datasets
        process_names.append("graviton_hh_ggf_bbww")
        process_names.append("radion_hh_ggf_bbww")

    if config.has_tag("is_nonresonant"):
        # add nonresonant signal processes/datasets
        for hh_proc in ("ggHH_sl_hbbhww", "ggHH_dl_hbbhww", "qHH_sl_hbbhww", "qqHH_dl_hbbhww"):
            process_names.append(hh_proc)

    return process_names


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


def configure_hbw_processes(config: od.Config, campaign: od.Campaign):
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

    if config.has_tag("is_dl") and config.has_tag("is_nonresonant"):
        # Custom signal  process for ML Training, combining multiple kl signal samples
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
            sig.add_process(proc)

    # add auxiliary information if process is signal
    for proc_inst in config.processes:
        is_signal = any([
            signal_tag in proc_inst.name
            for signal_tag in ("qqHH", "ggHH", "radion", "gravition")
        ])
        if is_signal:
            proc_inst.add_tag("is_signal")
