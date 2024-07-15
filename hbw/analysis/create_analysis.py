# coding: utf-8

"""
Configuration of the HH -> bbWW analysis.
"""

import os

import law
import order as od


thisdir = os.path.dirname(os.path.abspath(__file__))

logger = law.logger.get_logger(__name__)


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
    campaign_run2_2017_nano_v9.x.run = 2
    campaign_run2_2017_nano_v9.x.postfix = ""

    campaign_run3_2022_preEE_nano_v12 = cmsdb.campaigns.run3_2022_preEE_nano_v12.campaign_run3_2022_preEE_nano_v12
    campaign_run3_2022_preEE_nano_v12.x.EE = "pre"
    campaign_run3_2022_preEE_nano_v12.x.run = 3
    campaign_run3_2022_preEE_nano_v12.x.postfix = ""

    campaign_run3_2022_postEE_nano_v12 = cmsdb.campaigns.run3_2022_postEE_nano_v12.campaign_run3_2022_postEE_nano_v12
    campaign_run3_2022_postEE_nano_v12.x.EE = "post"
    campaign_run3_2022_postEE_nano_v12.x.run = 3
    campaign_run3_2022_postEE_nano_v12.x.postfix = "EE"

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

    #
    # modify store_parts
    #

    def merged_analysis_parts(task, store_parts):
        software_tasks = ("cf.BundleBashSandbox", "cf.BundleCMSSWSandbox", "cf.BundleRepo", "cf.BundleSoftware")
        if task.task_family in software_tasks:
            store_parts["analysis"] = "software_bundles"
            return store_parts

        shareable_tasks = ("cf.CalibrateEvents", "cf.GetDatasetLFNs")
        if task.task_family not in shareable_tasks:
            logger.warning(f"task {task.task_family} is not shareable")
            return store_parts

        if "analysis" in store_parts:
            # always use the same analysis
            store_parts["analysis"] = "hbw_merged"

        if "config" in store_parts:
            # share outputs between limited and non-limited config
            store_parts["config"] = store_parts["config"].replace("l", "c")

        return store_parts

    def limited_config_shared_parts(task, store_parts):
        shareable_tasks = ("cf.CalibrateEvents", "cf.SelectEvents", "cf.ReduceEvents")

        if task.task_family not in shareable_tasks:
            logger.warning(f"task {task.task_family} should not be shared between limited and non-limited config")
            return store_parts

        if "config" in store_parts:
            # share outputs between limited and non-limited config
            store_parts["config"] = store_parts["config"].replace("l", "c")

        return store_parts

    analysis_inst.x.store_parts_modifiers = {
        "merged_analysis": merged_analysis_parts,
        "limited_config_shared": limited_config_shared_parts,
    }

    return analysis_inst
