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
    # "$HBW_BASE/sandboxes/venv_columnar_tf.sh",
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

#
# import campaigns and load configs
#

from hbw.config.config_run2 import add_config
import cmsdb.campaigns.run2_2017_nano_v9

campaign_run2_2017_nano_v9 = cmsdb.campaigns.run2_2017_nano_v9.campaign_run2_2017_nano_v9

# default config
add_config(
    analysis_hbw,
    campaign_run2_2017_nano_v9.copy(),
    config_name="config_2017",
    config_id=2,
)

# config with limited number of files
add_config(
    analysis_hbw,
    campaign_run2_2017_nano_v9.copy(),
    config_name="config_2017_limited",
    config_id=12,
    limit_dataset_files=2,
)
