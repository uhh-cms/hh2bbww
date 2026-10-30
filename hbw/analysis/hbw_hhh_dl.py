# coding: utf-8

"""
Main analysis object for the nonresonant HHH -> 4b2W(DL) analysis
"""

from hbw.analysis.create_analysis import create_hbw_analysis

hbw_hhh_dl = create_hbw_analysis("hbw_hhh_dl", 4, tags={"is_dl", "is_hhh", "is_nonresonant"})
