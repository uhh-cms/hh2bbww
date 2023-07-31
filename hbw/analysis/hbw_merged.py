# coding: utf-8

"""
Main analysis object for the HH -> bbWW analysis, combining all sub-analyses.
This analysis inst is mainly used to produce commonly used outputs, e.g. from these tasks
- cf.GetDatasetLFNs
- cf.CalibrateEvents
"""

from hbw.analysis.create_analysis import create_hbw_analysis

hbw_merged = create_hbw_analysis("hbw_merged", 1, tags={"is_sl", "is_dl", "is_resonant", "is_nonresonant"})
