# coding: utf-8

"""
SL-specific configuration of the HH -> bbWW analysis.
"""

from __future__ import annotations

import order as od

from hbw.util import four_vec
# from hbw.config.sl.defaults_and_groups import set_config_defaults_and_groups


def configure_sl_res(config: od.Config):
    config.x.keep_columns["cf.ReduceEvents"] = (
        config.x.keep_columns["cf.ReduceEvents"] | four_vec("Lightjet", {"btagDeepFlavB", "hadronFlavour"})
    )

    # set some config defaults and groups
    # set_config_defaults_and_groups(config)
