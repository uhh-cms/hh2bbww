# coding: utf-8

"""
Shared pytest configuration and fixtures of the hbw test suite.

Environment variables:

- ``HBW_TEST_STORE``: when set, it replaces ``CF_STORE_LOCAL`` for the test session, e.g. to check
  that the campaign summaries can be built from scratch in an empty store.
"""

from __future__ import annotations

import os
import sys

import pytest

base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if base not in sys.path:
    sys.path.insert(0, base)

if os.getenv("HBW_TEST_STORE"):
    os.environ["CF_STORE_LOCAL"] = os.environ["HBW_TEST_STORE"]

# superseded by tests/tasks/test_task_resolution.py
collect_ignore = ["test_workflow.py"]


@pytest.fixture(scope="session")
def get_config():
    """
    Factory fixture returning (and caching) configs per (analysis, config) name pair, see
    :py:func:`tests.helpers.load_config`. Configs are expensive to build, so they are shared by all
    tests of a session.
    """
    from tests.helpers import load_config

    cache = {}

    def get(analysis_name: str, config_name: str):
        key = (analysis_name, config_name)
        if key not in cache:
            cache[key] = load_config(analysis_name, config_name)
        return cache[key]

    return get
