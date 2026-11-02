# coding: utf-8

"""
Separate instances of the hbw analyses for tests.

Configs are shared by all tests of a session, but some setups cannot be combined on the same config:

- The categories can either be set up for production (``pre_ml_cats``) or for an ML evaluation
  (``cats_ml_*``), but not both (see ``hbw.config.categories.add_categories_ml``).
- ML models with the same train node names cannot be set up together, since their training
  processes clash (see ``hbw.config.processes.prepare_ml_processes``).

Tests that need such setups use copies of the analysis (e.g. ``get_config("tests.analyses.hbw_dl_ml",
"c24v15")`` or ``--analysis tests.analyses.hbw_dl_ml``), or an isolated one via
:py:func:`isolated_analysis`.
"""

from __future__ import annotations

import law

from hbw.analysis.create_analysis import create_hbw_analysis
from hbw.analysis.hbw_dl import hbw_dl


def copy_analysis(analysis_inst):
    return create_hbw_analysis(analysis_inst.name, analysis_inst.id, tags=set(analysis_inst.tags))


# analysis for tasks and models using the ML categorization
hbw_dl_ml = copy_analysis(hbw_dl)


def isolated_analysis(analysis_name: str, key: str) -> str:
    """
    Creates (once per *key*) a copy of the hbw analysis *analysis_name* (e.g. "hbw_dl") with its own
    configs, and returns its path to be used with the ``get_config`` fixture.
    """
    from tests.helpers import load_analysis

    attr = f"{analysis_name}_{law.util.create_hash(key)}"
    if attr not in globals():
        globals()[attr] = copy_analysis(load_analysis(analysis_name))
    return f"{__name__}.{attr}"
