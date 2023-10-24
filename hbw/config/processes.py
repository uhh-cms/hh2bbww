# coding: utf-8

"""
Configuration of the Run 2 HH -> bbWW processes.
"""

import order as od

from columnflow.config_util import get_root_processes_from_campaign


def add_hbw_processes(config: od.Config, campaign: od.Campaign):
    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # add processes we are interested in
    config.add_process(procs.n.data)
    config.add_process(procs.n.tt)
    config.add_process(procs.n.st)
    config.add_process(procs.n.w_lnu)
    config.add_process(procs.n.dy_lep)
    config.add_process(procs.n.qcd)
    config.add_process(procs.n.qcd_mu)
    config.add_process(procs.n.qcd_em)
    config.add_process(procs.n.qcd_bctoe)
    # config.add_process(procs.n.ttv)
    # config.add_process(procs.n.vv)
    # config.add_process(procs.n.vv)
    # config.add_process(procs.n.hh_ggf_bbtautau)

    if config.has_tag("is_sl") and config.has_tag("is_nonresonant"):
        config.add_process(procs.n.ggHH_kl_0_kt_1_sl_hbbhww)
        config.add_process(procs.n.ggHH_kl_1_kt_1_sl_hbbhww)
        config.add_process(procs.n.ggHH_kl_2p45_kt_1_sl_hbbhww)
        config.add_process(procs.n.ggHH_kl_5_kt_1_sl_hbbhww)
        config.add_process(procs.n.qqHH_CV_1_C2V_1_kl_1_sl_hbbhww)
        config.add_process(procs.n.qqHH_CV_1_C2V_1_kl_0_sl_hbbhww)
        config.add_process(procs.n.qqHH_CV_1_C2V_1_kl_2_sl_hbbhww)
        config.add_process(procs.n.qqHH_CV_1_C2V_0_kl_1_sl_hbbhww)
        config.add_process(procs.n.qqHH_CV_1_C2V_2_kl_1_sl_hbbhww)
        config.add_process(procs.n.qqHH_CV_0p5_C2V_1_kl_1_sl_hbbhww)
        config.add_process(procs.n.qqHH_CV_1p5_C2V_1_kl_1_sl_hbbhww)

    if config.has_tag("is_dl") and config.has_tag("is_nonresonant"):
        config.add_process(procs.n.ggHH_kl_0_kt_1_dl_hbbhww)
        config.add_process(procs.n.ggHH_kl_1_kt_1_dl_hbbhww)
        config.add_process(procs.n.ggHH_kl_2p45_kt_1_dl_hbbhww)
        config.add_process(procs.n.ggHH_kl_5_kt_1_dl_hbbhww)
        config.add_process(procs.n.qqHH_CV_1_C2V_1_kl_1_dl_hbbhww)
        config.add_process(procs.n.qqHH_CV_1_C2V_1_kl_0_dl_hbbhww)
        config.add_process(procs.n.qqHH_CV_1_C2V_1_kl_2_dl_hbbhww)
        config.add_process(procs.n.qqHH_CV_1_C2V_0_kl_1_dl_hbbhww)
        config.add_process(procs.n.qqHH_CV_1_C2V_2_kl_1_dl_hbbhww)
        config.add_process(procs.n.qqHH_CV_0p5_C2V_1_kl_1_dl_hbbhww)
        config.add_process(procs.n.qqHH_CV_1p5_C2V_1_kl_1_dl_hbbhww)

    if config.has_tag("is_sl") and config.has_tag("is_resonant"):
        for mass in config.x.graviton_masspoints:
            config.add_process(procs.n(f"graviton_hh_ggf_bbww_m{mass}"))
        for mass in config.x.radion_masspoints:
            config.add_process(procs.n(f"radion_hh_ggf_bbww_m{mass}"))

    #
    # add some custom processes
    #

    # QCD process customization
    config.get_process("qcd_mu").label = "QCD Muon enriched"
    qcd_ele = config.add_process(
        name="qcd_ele",
        id=31199,
        xsecs={13: config.get_process("qcd_em").get_xsec(13) + config.get_process("qcd_bctoe").get_xsec(13)},
        label="QCD Electron enriched",
    )
    qcd_ele.add_process(config.get_process("qcd_em"))
    qcd_ele.add_process(config.get_process("qcd_bctoe"))

    # Custom v_lep process for ML Training, combining W+DY
    v_lep = config.add_process(
        name="v_lep",
        id=64575573,  # random number
        xsecs={13: config.get_process("w_lnu").get_xsec(13) + config.get_process("dy_lep").get_xsec(13)},
        label="W and DY",
    )
    v_lep.add_process(config.get_process("w_lnu"))
    v_lep.add_process(config.get_process("dy_lep"))
    
    # Custom t_bkg process for ML Training, combining tt+st
    t_bkg = config.add_process(
        name="t_bkg",
        id=97842611,  # random number
        xsecs={13: config.get_process("tt").get_xsec(13) + config.get_process("st").get_xsec(13)},
        label="tt + st",
    )
    t_bkg.add_process(config.get_process("tt"))
    t_bkg.add_process(config.get_process("st"))
    ''' 
    # Custom signal  process for ML Training, combining all kl signal samples 
    sig = config.add_process(
        name="sig",
        id=97842611,  # random number
        xsecs={13: config.get_process("ggHH_kl_0_kt_1_dl_hbbhww").get_xsec(13) + config.get_process("ggHH_kl_1_kt_1_dl_hbbhww").get_xsec(13)},
        label="signal",
    )
    sig.add_process(config.get_process("ggHH_kl_0_kt_1_dl_hbbhww"))
    sig.add_process(config.get_process("ggHH_kl_1_kt_1_dl_hbbhww"))
    '''
