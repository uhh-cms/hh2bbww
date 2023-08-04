# coding: utf-8

"""
DL-specific configuration of the HH -> bbWW analysis.
"""

from __future__ import annotations

import order as od

# from hbw.config.dl.defaults_and_groups import set_config_defaults_and_groups


def configure_dl(config: od.Config):
    # add columns produced during dl selection
    config.x.keep_columns["cf.ReduceEvents"] = (
        config.x.keep_columns["cf.ReduceEvents"] | {"m_ll", "channel_id"}
    )
    config.x.keep_columns["cf.MergeSelectionMasks"] = (
        config.x.keep_columns["cf.MergeSelectionMasks"] | {"m_ll", "channel_id"}
    )

    # set some config defaults and groups
    # set_config_defaults_and_groups(config)
