# coding: utf-8

"""
Checks that all local external files of the maintained configs exist (requires cvmfs and the
DESY/CERN AFS and dust areas, so only run on the NAF).
"""

from __future__ import annotations

import os

import pytest

from columnflow.tasks.external import ExternalFile

pytestmark = [pytest.mark.naf, pytest.mark.config]

CONFIGS = [("hbw_dl", "c24v15"), ("hbw_dl", "c25v15")]


def iter_locations(value, key: str = ""):
    """
    Yields (key, location) pairs of all files in the (nested) external files *value*.
    """
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            yield from iter_locations(sub_value, f"{key}.{sub_key}" if key else sub_key)
    elif isinstance(value, ExternalFile):
        yield key, value.location
    elif isinstance(value, (tuple, list)):
        yield key, value[0]
    elif isinstance(value, str):
        yield key, value


@pytest.mark.parametrize("case", CONFIGS, ids="-".join)
def test_external_files_exist(get_config, case):
    config_inst = get_config(*case)
    locations = dict(iter_locations(config_inst.x.external_files))
    assert locations

    missing = {
        key: location
        for key, location in locations.items()
        # remote files are fetched by BundleExternalFiles
        if "://" not in location and not os.path.exists(os.path.expandvars(location))
    }
    assert not missing, f"missing external files in {config_inst.name}: {missing}"
