# coding: utf-8

"""
SL-specific configuration of the HH -> bbWW analysis.
"""

from __future__ import annotations

import order as od

# from hbw.config.sl.defaults_and_groups import set_config_defaults_and_groups


def configure_sl(config: od.Config):
    """
    Configure the SL-specific settings of the HH -> bbWW analysis.
    """

    # add qcd as process
    config.add_process(config.x.procs.n.qcd)
    return
