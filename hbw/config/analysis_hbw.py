# coding: utf-8

"""
Configuration of the HH -> bbWW analysis.
"""

import os

import law
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

# files of sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
analysis_hbw.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf_prod.sh",
    "$CF_BASE/sandboxes/venv_columnar.sh",
    "$HBW_BASE/sandboxes/venv_columnar_tf.sh",
]

# cmssw sandboxes that should be bundled for remote jobs in case they are needed
analysis_hbw.set_aux("cmssw_sandboxes", [
    # "$CF_BASE/sandboxes/cmssw_default.sh",
])


# clear the list when cmssw bundling is disabled
if not law.util.flag_to_bool(os.getenv("HBW_BUNDLE_CMSSW", "1")):
    del analysis_hbw.x.cmssw_sandboxes[:]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
analysis_hbw.set_aux("config_groups", {})

# trailing imports for different configs
import hbw.config.config_2017  # noqa
# import hbw.config.config_2018  # noqa
