# coding: utf-8

"""
Configuration of the HH -> bbWW analysis.
"""

import os

import order as od


thisdir = os.path.dirname(os.path.abspath(__file__))

#
# the main analysis object
#

analysis_hbw = od.Analysis(
    name="analysis_hbw",
    id=1,
)

# analysis-global versions
analysis_hbw.set_aux("versions", {
})

# sandboxes that might be required by remote tasks
# (used in PrepareJobSandboxes)
analysis_hbw.x.job_sandboxes = [
    "bash::$CF_BASE/sandboxes/venv_columnar.sh",
]

# cmssw sandboxes that should be bundled for remote jobs in case they are needed
analysis_hbw.set_aux("cmssw_sandboxes", [
    # "cmssw_default.sh",
])

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
analysis_hbw.set_aux("config_groups", {})

# trailing imports for different configs
import hbw.config.config_2017  # noqa
# import hbw.config.config_2018  # noqa
