# coding: utf-8

"""
SL-specific configuration of the resonant HH -> bbWW analysis.
"""

from __future__ import annotations

import order as od

from hbw.util import four_vec
# from hbw.config.sl.defaults_and_groups import set_config_defaults_and_groups


def configure_sl_res(config: od.Config):
    config.x.keep_columns["cf.ReduceEvents"] = (
        config.x.keep_columns["cf.ReduceEvents"] | four_vec("Lightjet", {"btagDeepFlavB", "hadronFlavour"})
    )
    sig_true_mass = [250, 300, 350, 400, 450, 600, 750, 1000]
    for mass in sig_true_mass: 
        dataset_signal = config.get_dataset(f"graviton_hh_ggf_bbww_m{mass}_madgraph")
        dataset_signal.add_tag(f"is_graviton{mass}")
    for dataset in config.datasets:
        if dataset.name.startswith(("st", "tt", "dy", "w_lnu")):
            dataset.add_tag("is_bkg_pnn")