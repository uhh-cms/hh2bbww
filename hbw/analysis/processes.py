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
        qcd_mu, qcd_em, qcd_bctoe, data,
        tt, ttv, st, w_lnu, vv, h,
        dy, dy_m4to10, dy_m10to50, dy_m50toinf, dy_m50toinf_0j, dy_m50toinf_1j, dy_m50toinf_2j,
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

    data.remove_process("data_jethtmet")

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
