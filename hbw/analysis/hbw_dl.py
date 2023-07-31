# coding: utf-8

"""
Main analysis object for the nonresonant HH -> bbWW(DL) analysis
"""

from hbw.analysis.create_analysis import create_hbw_analysis

hbw_dl = create_hbw_analysis("hbw_dl", 3, tags={"is_dl", "is_nonresonant"})
