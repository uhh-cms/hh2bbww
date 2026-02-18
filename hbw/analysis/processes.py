# coding: utf-8

"""
Creation and modification of processes in the HH -> bbWW analysis.
NOTE: it is crucial to modify processes before the campaign is created. Otherwise,
the changes will not be reflected in the campaign and there will be inconsistencies.
"""

# import order as od

import law
logger = law.logger.get_logger(__name__)


from hbw.config.processes import create_parent_process
from hbw.config.styling import color_palette
from cmsdb.util import add_decay_process


def modify_cmsdb_processes():
    from cmsdb.processes import (
        data, data_met,
        qcd_mu, qcd_em, qcd_bctoe,
        tt, ttv, st, w_lnu, vv, h,
        dy, dy_m4to10, dy_m10to50, dy_m50toinf, dy_m50toinf_0j, dy_m50toinf_1j, dy_m50toinf_2j,
        dy_ee, dy_ee_m10to50, dy_ee_m50toinf, dy_ee_m50toinf_0j, dy_ee_m50toinf_1j, dy_ee_m50toinf_2j,
        dy_mumu, dy_mumu_m10to50, dy_mumu_m50toinf, dy_mumu_m50toinf_0j, dy_mumu_m50toinf_1j, dy_mumu_m50toinf_2j,
        dy_tautau, dy_tautau_m10to50, dy_tautau_m50toinf, dy_tautau_m50toinf_0j, dy_tautau_m50toinf_1j, dy_tautau_m50toinf_2j,  # noqa E501
        ttvv, tttt, vvv,
        h_ggf, h_vbf, vh,
        tth, thq, thw,
        st_twchannel_t_dl, st_twchannel_tbar_dl,
        tt_dl,
        tt_dl_nonb, tt_sl_nonb, tt_fh_nonb,
        ttbb, ttbb_dl_1b, ttbb_sl_1b, ttbb_fh_1b,
        hh_ggf_hbb_hzz_kl1_kt1, hh_vbf_hbb_hzz_kv1_k2v1_kl1,
        hh_ggf_hbb_htt_kl1_kt1, hh_vbf_hbb_htt_kv1_k2v1_kl1,
    )

    data.remove_process(data_met)

    # NOTE: This can be commented out in order to plot data split in era
    # !!! The process_ids have to be touched though in the hist producer base
    # configure_data_split_in_eras()

    decay_map = {
        "lf": {
            "name": "lf",
            "id": 50050,
            "label": "(lf)",
            "br": -1,
        },
        "hf": {
            "name": "hf",
            "id": 70070,
            "label": "(hf)",
            "br": -1,
        },
    }

    # top-level disambiguation
    # NOTE: this is needed since 2024 we have the additional lep verbosity
    dy.aux = {}
    dy_m4to10.aux = {}
    dy_m10to50.aux = {}
    dy_m50toinf.aux = {}
    dy_ee.aux = {"production_mode_parent": ["dy"]}
    dy_mumu.aux = {"production_mode_parent": ["dy"]}
    dy_tautau.aux = {"production_mode_parent": ["dy"]}
    dy_ee_m10to50.aux = {"production_mode_parent": ["dy_ee"]}
    dy_ee_m50toinf.aux = {"production_mode_parent": ["dy_ee"]}
    dy_ee_m50toinf_0j.aux = {"production_mode_parent": ["dy_ee_m50toinf"]}
    dy_ee_m50toinf_1j.aux = {"production_mode_parent": ["dy_ee_m50toinf"]}
    dy_ee_m50toinf_2j.aux = {"production_mode_parent": ["dy_ee_m50toinf"]}

    for dy_proc_inst in (
        dy, dy_m4to10, dy_m10to50, dy_m50toinf, dy_m50toinf_0j, dy_m50toinf_1j, dy_m50toinf_2j,
        dy_ee, dy_ee_m10to50, dy_ee_m50toinf, dy_ee_m50toinf_0j, dy_ee_m50toinf_1j, dy_ee_m50toinf_2j,
        dy_mumu, dy_mumu_m10to50, dy_mumu_m50toinf, dy_mumu_m50toinf_0j, dy_mumu_m50toinf_1j, dy_mumu_m50toinf_2j,
        dy_tautau, dy_tautau_m10to50, dy_tautau_m50toinf, dy_tautau_m50toinf_0j, dy_tautau_m50toinf_1j, dy_tautau_m50toinf_2j,  # noqa E501
    ):
        for flavour in ("hf", "lf"):
            aux = {"flavour": flavour}
            if hasattr(dy_proc_inst, "aux") and dy_proc_inst.aux:
                aux.update(dy_proc_inst.aux)  # merge pre-set production_mode_parent
            add_production_mode_parent = "production_mode_parent" in aux
            add_decay_process(
                dy_proc_inst,
                decay_map[flavour],
                add_production_mode_parent=add_production_mode_parent,
                name_func=lambda parent_name, decay_name: f"{parent_name}_{decay_name}",
                label_func=lambda parent_label, decay_label: f"{parent_label} {decay_label}",
                xsecs=None,
                aux=aux,
            )

    tt_custom = create_parent_process(
        [tt_dl_nonb, tt_sl_nonb, tt_fh_nonb],
        name="tt_custom",
        id=21199,
        label="TT Custom",
    )
    tt_custom.add_parent_process(tt)

    ttbb_custom = create_parent_process(
        [ttbb_dl_1b, ttbb_sl_1b, ttbb_fh_1b],
        name="ttbb_custom",
        id=68899,
        label="TTBB Custom",
    )
    ttbb_custom.add_parent_process(ttbb)

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


def configure_data_split_in_eras():
    from cmsdb.processes import data

    data_era_map = {
        "data_2024_c": {"color": "#1f77b4", "label": "Data c 2024", "id": 893685478},
        "data_2024_d": {"color": "#ff7f0e", "label": "Data d 2024", "id": 893685479},
        "data_2024_e": {"color": "#2ca02c", "label": "Data e 2024", "id": 893685480},
        "data_2024_f": {"color": "#d62728", "label": "Data f 2024", "id": 893685481},
        "data_2024_g": {"color": "#9467bd", "label": "Data g 2024", "id": 893685482},
        "data_2024_h": {"color": "#8c564b", "label": "Data h 2024", "id": 893685483},
        "data_2024_i": {"color": "#e377c2", "label": "Data i 2024", "id": 893685484},
    }

    for name, info in data_era_map.items():
        data.add_process(
            name=name,
            id=info["id"],
            is_data=False,
            label=info["label"],
            color=info["color"],
            aux={
                "scale": 1.0,
                "stack": False,
            },
        )
