# coding: utf-8

"""
Configuration of the HH -> bbWW analysis.
"""

import os

import law
import order as od


thisdir = os.path.dirname(os.path.abspath(__file__))


def create_hbw_analysis(
    name,
    id,
    **kwargs,
) -> od.Analysis:

    #
    # the main analysis object
    #

    analysis_inst = od.Analysis(
        name=name,
        id=id,
        **kwargs,
    )

    # analysis-global versions
    analysis_inst.set_aux("versions", {
    })

    # files of sandboxes that might be required by remote tasks
    # (used in cf.HTCondorWorkflow)
    analysis_inst.x.bash_sandboxes = [
        "$CF_BASE/sandboxes/cf.sh",
        "$CF_BASE/sandboxes/venv_columnar.sh",
        # "$CF_BASE/sandboxes/venv_ml_tf.sh",
        "$HBW_BASE/sandboxes/venv_ml_plotting.sh",
    ]

    # cmssw sandboxes that should be bundled for remote jobs in case they are needed
    analysis_inst.set_aux("cmssw_sandboxes", [
        "$CF_BASE/sandboxes/cmssw_default.sh",
    ])

    # clear the list when cmssw bundling is disabled
    if not law.util.flag_to_bool(os.getenv("HBW_BUNDLE_CMSSW", "1")):
        del analysis_inst.x.cmssw_sandboxes[:]

    # config groups for conveniently looping over certain configs
    # (used in wrapper_factory)
    analysis_inst.set_aux("config_groups", {})

    #
    # import campaigns and load configs
    #

    from hbw.config.config_run2 import add_config

    import cmsdb.campaigns.run2_2017_nano_v9
    import cmsdb.campaigns.run3_2022_preEE_nano_v12
    import cmsdb.campaigns.run3_2022_postEE_nano_v12

    campaign_run2_2017_nano_v9 = cmsdb.campaigns.run2_2017_nano_v9.campaign_run2_2017_nano_v9
    campaign_run3_2022_preEE_nano_v12 = cmsdb.campaigns.run3_2022_preEE_nano_v12.campaign_run3_2022_preEE_nano_v12
    campaign_run3_2022_preEE_nano_v12.x.EE = "pre"

    campaign_run3_2022_postEE_nano_v12 = cmsdb.campaigns.run3_2022_postEE_nano_v12.campaign_run3_2022_postEE_nano_v12
    campaign_run3_2022_postEE_nano_v12.x.EE = "post"

    # 2017
    c17 = add_config(  # noqa
        analysis_inst,
        campaign_run2_2017_nano_v9.copy(),
        config_name="c17",
        config_id=1700,
        add_dataset_extensions=False,
    )
    l17 = add_config(  # noqa
        analysis_inst,
        campaign_run2_2017_nano_v9.copy(),
        config_name="l17",
        config_id=1701,
        limit_dataset_files=2,
        add_dataset_extensions=False,
    )

    # 2022 preEE
    c22pre = add_config(  # noqa
        analysis_inst,
        campaign_run3_2022_preEE_nano_v12.copy(),
        config_name="c22pre",
        config_id=2200,
        add_dataset_extensions=False,
    )
    l22pre = add_config(  # noqa
        analysis_inst,
        campaign_run3_2022_preEE_nano_v12.copy(),
        config_name="l22pre",
        config_id=2201,
        limit_dataset_files=2,
        add_dataset_extensions=False,
    )

    # 2022 postEE
    c22post = add_config(  # noqa
        analysis_inst,
        campaign_run3_2022_postEE_nano_v12.copy(),
        config_name="c22post",
        config_id=2210,
        add_dataset_extensions=False,
    )
    l22post = add_config(  # noqa
        analysis_inst,
        campaign_run3_2022_postEE_nano_v12.copy(),
        config_name="l22post",
        config_id=2211,
        limit_dataset_files=2,
        add_dataset_extensions=False,
    )

    return analysis_inst
