# coding: utf-8

"""
Creation and modification of processes in the HH -> bbWW analysis.
NOTE: it is crucial to modify processes before the campaign is created. Otherwise,
the changes will not be reflected in the campaign and there will be inconsistencies.
"""

# import order as od


from hbw.config.processes import create_parent_process
from hbw.config.styling import color_palette
from cmsdb.util import add_decay_process


def modify_cmsdb_processes():
    from cmsdb.processes import (
        data, data_met,
        qcd_mu, qcd_em, qcd_bctoe,
        tt, ttv, st, w_lnu, vv, h,
        dy, dy_m4to10, dy_m10to50, dy_m50toinf, dy_m50toinf_0j, dy_m50toinf_1j, dy_m50toinf_2j,
        ttvv, tttt, vvv,
        h_ggf, h_vbf, vh,
        tth, thq, thw,
        st_twchannel_t_dl, st_twchannel_tbar_dl,
        tt_dl,
        hh_ggf_hbb_hzz_kl1_kt1, hh_vbf_hbb_hzz_kv1_k2v1_kl1,
        hh_ggf_hbb_htt_kl1_kt1, hh_vbf_hbb_htt_kv1_k2v1_kl1,
    )

    data.remove_process(data_met)

    decay_map = {
        "lf": {
            "name": "lf",
            "id": 50,
            "label": "(lf)",
            "br": -1,
        },
        "hf": {
            "name": "hf",
            "id": 70,
            "label": "(hf)",
            "br": -1,
        },
    }

    for dy_proc_inst in (
        dy, dy_m4to10, dy_m10to50, dy_m50toinf, dy_m50toinf_0j, dy_m50toinf_1j, dy_m50toinf_2j,
    ):
        add_production_mode_parent = dy_proc_inst.name != "dy"
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

    qcd_mu.label = "QCD Muon enriched"
    qcd_ele = create_parent_process(
        [qcd_em, qcd_bctoe],
        name="qcd_ele",
        id=31199,
        label="QCD Electron enriched",
    )

    v_lep = create_parent_process(
        [w_lnu, dy],
        name="v_lep",
        id=64575573,  # random number
        label="W and DY",
    )

    t_bkg = create_parent_process(
        [st, tt, ttv],
        name="t_bkg",
        id=97842611,  # random number
        label="tt + st",
    )

    background = create_parent_process(  # noqa: F841
        [t_bkg, v_lep, vv, w_lnu, h, qcd_ele, qcd_mu],
        name="background",
        id=99999,
        label="background",
        color=color_palette["blue"],
    )

    other = create_parent_process(  # noqa: F841
        [ttvv, tttt, vvv],
        name="other",
        id=99998,
        label="other",
        color=color_palette["grey"],
    )

    minor = create_parent_process(  # noqa: F841
        [w_lnu, vv, vvv, ttv, ttvv, tttt, tth, thq, thw, h_ggf, h_vbf, vh],
        name="minor",
        id=99997,
        label="minor processes",
        color=color_palette["purple"],
    )

    ttboson = create_parent_process(  # noqa: F841
        [tttt, ttv, ttvv, tth, thq, thw],
        name="ttboson",
        id=99996,
        label="tt + boson",
        color=color_palette["orange"],
    )

    minor_bosons = create_parent_process(  # noqa: F841
        [w_lnu, vv, vvv, h_ggf, h_vbf, vh],
        name="minor_bosons",
        id=99995,
        label="",
        color=color_palette["purple"],
    )

    multiboson = create_parent_process(  # noqa: F841
        [vv, vvv],
        name="multiboson",
        id=99994,
        label="multiboson",
        color=color_palette["green"],
    )
    hh_other = create_parent_process(  # noqa: F841
        [hh_ggf_hbb_hzz_kl1_kt1, hh_vbf_hbb_hzz_kv1_k2v1_kl1,
         hh_ggf_hbb_htt_kl1_kt1, hh_vbf_hbb_htt_kv1_k2v1_kl1],
        name="hh_other",
        id=99993,
        label="HH (other)",
        color=color_palette["darkgrey"],
    )

    tt_dy = create_parent_process(  # noqa: F841
        [tt, dy],
        name="tt_dy",
        id=99890206,
        label="tt + DY",
        color=color_palette["red"],
    )

    sf_bkg = create_parent_process(  # noqa: F841
        [h, ttv, vv, dy, st, tt],
        name="sf_bkg",
        id=99890207,
        label="MC background",
        color=color_palette["green"],
    )

    sf_bkg_reduced = create_parent_process(  # noqa: F841
        [dy_m50toinf, st_twchannel_t_dl, st_twchannel_tbar_dl, tt_dl],
        name="sf_bkg_reduced",
        id=99890208,
        label="MC background",
        color=color_palette["green"],
    )
