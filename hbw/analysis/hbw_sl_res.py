# coding: utf-8

"""
Main analysis object for the resonant HH -> bbWW(SL) analysis
"""

from hbw.analysis.create_analysis import create_hbw_analysis

hbw_sl_res = create_hbw_analysis(
    "hbw_sl_res", 5,
    tags={
        "is_resonant",
        "is_sl",
        # "custom_signals",
    },
)
