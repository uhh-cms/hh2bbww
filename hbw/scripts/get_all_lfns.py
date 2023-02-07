# coding: utf-8

"""
Minimal script to run GetDatasetLFNs for all datasets of the 2017 config
"""

from columnflow.tasks.external import GetDatasetLFNs
from hbw import config_2017

config = "config_2017_limited"

for dataset in config_2017.datasets:
    task = GetDatasetLFNs(
        dataset=dataset.name,
        config=config,
    )
    task.law_run()
