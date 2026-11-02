# coding: utf-8

"""
Checks of the module lists in all tracked law config files.

columnflow loads these modules via ``maybe_import``, which silently replaces a missing module by a
mock, so a typo or a removed module would otherwise only surface much later as an unknown
selector, producer, model or task.
"""

from __future__ import annotations

import os
import glob
import subprocess
import importlib.util
import configparser

import law
import pytest

from tests.known_issues import MISPLACED_LAW_CFG_MODULES, maybe_xfail

base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def tracked_law_configs() -> list[str]:
    try:
        out = subprocess.check_output(["git", "ls-files", "law*.cfg"], cwd=base, text=True)
        files = out.split()
    except (OSError, subprocess.CalledProcessError):
        files = []
    return sorted(files or map(os.path.basename, glob.glob(os.path.join(base, "law*.cfg"))))


def read_config(name: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(allow_no_value=True, interpolation=None, strict=False)
    # keep case of options (task families, module names)
    parser.optionxform = str
    parser.read(os.path.join(base, name))
    return parser


def split_modules(value: str) -> list[str]:
    """
    Splits a comma-separated module list at commas outside of braces and expands the braces, e.g.
    "hbw.production.{weights,met}, hbw.ml.stats" -> ["hbw.production.weights", "hbw.production.met",
    "hbw.ml.stats"].
    """
    parts, depth, current = [], 0, ""
    for char in value:
        depth += {"{": 1, "}": -1}.get(char, 0)
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)

    modules = []
    for part in map(str.strip, parts):
        if part:
            modules.extend(m.strip() for m in law.util.brace_expand(part))
    return modules


def module_entries(parser: configparser.ConfigParser) -> list[tuple[str, str]]:
    """
    Returns (option, module) pairs of all modules that columnflow would load from *parser*.
    """
    entries = []
    if parser.has_section("analysis"):
        for option, value in parser.items("analysis"):
            if option.endswith("_modules") and value:
                entries.extend((option, m) for m in split_modules(value))
    if parser.has_section("modules"):
        for option in parser.options("modules"):
            entries.extend(("modules", m) for m in split_modules(option))
    return entries


@pytest.mark.parametrize("cfg_name", tracked_law_configs())
def test_modules_exist(cfg_name):
    missing = [
        f"[{option}] {module}"
        for option, module in module_entries(read_config(cfg_name))
        if importlib.util.find_spec(module) is None
    ]
    assert not missing, f"{cfg_name} lists modules that do not exist: {missing}"


@pytest.mark.parametrize("cfg_name", [
    maybe_xfail(name, MISPLACED_LAW_CFG_MODULES)
    for name in tracked_law_configs()
])
def test_module_lists_in_analysis_section(cfg_name):
    # module lists outside of [analysis] are ignored by columnflow
    parser = read_config(cfg_name)
    misplaced = [
        f"[{section}] {option}"
        for section in parser.sections()
        if section != "analysis"
        for option in parser.options(section)
        if option.endswith("_modules")
    ]
    assert not misplaced, f"{cfg_name} has module lists outside of [analysis]: {misplaced}"


def test_split_modules():
    assert split_modules("hbw.production.{weights,met}, hbw.ml.stats") == [
        "hbw.production.weights", "hbw.production.met", "hbw.ml.stats",
    ]
    assert split_modules("hbw.tasks.{a,b}") == ["hbw.tasks.a", "hbw.tasks.b"]
    assert split_modules("") == []
