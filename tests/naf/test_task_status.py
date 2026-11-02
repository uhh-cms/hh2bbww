# coding: utf-8

"""
Smoke test of the law command line: resolve the dependency tree of the datacard task without
running anything.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from tests.known_issues import BROKEN_TASK_DEFAULTS

pytestmark = pytest.mark.naf

base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

DEFAULT_DATACARDS_XFAIL = pytest.mark.xfail(reason=BROKEN_TASK_DEFAULTS["CreateDatacards-default"], strict=True)


@pytest.mark.skipif(not shutil.which("law"), reason="law executable not found")
@pytest.mark.parametrize("inference_model", [
    pytest.param("default", marks=DEFAULT_DATACARDS_XFAIL),
    "rate_boostedbkg32",
])
def test_create_datacards_print_deps(inference_model):
    cmd = [
        "law", "run", "cf.cms.CreateDatacards",
        "--analysis", "hbw.analysis.hbw_dl.hbw_dl",
        "--configs", "c24v15",
        "--version", "test_task_status",
        "--print-deps", "3",
    ]
    if inference_model != "default":
        cmd += ["--inference-model", inference_model]

    proc = subprocess.run(cmd, cwd=base, capture_output=True, text=True, timeout=1800)
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output[-5000:]
    for family in ("cf.MergeShiftedHistograms", "cf.MergeHistograms", "cf.CreateHistograms"):
        assert family in output, f"{family} missing in dependency tree:\n{output[-5000:]}"
