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
        # "$CF_BASE/sandboxes/cmssw_default.sh",
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

    software_tasks = ("cf.BundleBashSandbox", "cf.BundleCMSSWSandbox", "cf.BundleSoftware")
    shareable_analysis_tasks = ("cf.CalibrateEvents", "cf.GetDatasetLFNs")
    limited_config_shared_tasks = ("cf.CalibrateEvents", "cf.GetDatasetLFNs", "cf.SelectEvents", "cf.ReduceEvents")
    skip_new_version_schema = ("cf.CalibrateEvents", "cf.GetDatasetLFNs")
    known_parts = (
        # from cf
        "analysis", "task_family", "config", "configs", "dataset", "shift", "version",
        "calibrator", "calibrators", "selector", "producer", "producers",
        "ml_model", "ml_data", "ml_models",
        "weightprod", "inf_model",
        "plot", "shift_sources", "shifts", "datasets"
        # from hbw
        "processes",
    )

    def software_tasks_parts(task, store_parts):
        if task.task_family not in software_tasks:
            logger.warning(f"task {task.task_family} is not a software task")
            return store_parts

        if "analysis" in store_parts:
            store_parts["analysis"] = "software_bundles"
        return store_parts

    def merged_analysis_parts(task, store_parts):
        if task.task_family not in shareable_analysis_tasks:
            logger.warning(f"task {task.task_family} is not shareable over multiple analyses")
            return store_parts

        if "analysis" in store_parts:
            # always use the same analysis
            store_parts["analysis"] = "hbw_merged"

        return store_parts

    def limited_config_shared_parts(task, store_parts):
        if task.task_family not in limited_config_shared_tasks:
            logger.warning(f"task {task.task_family} should not be shared between limited and non-limited config")
            return store_parts

        if "config" in store_parts:
            # share outputs between limited and non-limited config
            store_parts["config"] = store_parts["config"].replace("l", "c")

        return store_parts

    def reorganize_parts(task, store_parts):
        # check for unknown parts
        unknown_parts = set(store_parts.keys()) - set(known_parts)
        if unknown_parts:
            logger.warning(f"task {task.task_family} has unknown parts: {unknown_parts}")

        # add placeholder part
        store_parts.insert_before("analysis", "PLACEHOLDER", "DUMMY")

        # define order of parts as we want them
        parts_order_start = [
            "analysis",
            "calibrator", "calibrators",
            "selector",
            "producer", "producers",
            "ml_data", "ml_model", "ml_models",
            "weightprod", "inf_model",
            "task_family",
            "config", "dataset", "shift",
        ]
        parts_order_end = ["version"]

        # move parts to the desired position
        for part in parts_order_start:
            if part in store_parts:
                store_parts.insert_before("PLACEHOLDER", part, store_parts.pop(part))
        store_parts.pop("PLACEHOLDER")
        for part in parts_order_end:
            if part in store_parts:
                store_parts.move_to_end(part)

        task_version_from_cspmw = (
            "cf.CalibrateEvents",
            "cf.SelectEvents", "cf.MergeSelectionStats", "cf.MergeSelectionMasks",
            "cf.ReduceEvents", "cf.MergeReducedEvents", "cf.MergeReductionStats",
            "cf.ProduceColumns",
            "cf.PrepareMLEvents", "cf.MergeMLEvents", "cf.MergeMLStats", "cf.MLTraining", "cf.MLEvaluation",
            "hbw.MLPreTraining", "hbw.MLEvaluationSingleFold", "hbw.PlotMLResultsSingleFold",
        )
        if task.task_family in task_version_from_cspmw:
            # skip version when it is already encoded from CSPMW TaskArrayFunction
            store_parts.pop("version")

        return store_parts

    def hbw_parts(task, store_parts):
        name = task.task_family
        if name in software_tasks:
            return software_tasks_parts(task, store_parts)
        if name in shareable_analysis_tasks:
            store_parts = merged_analysis_parts(task, store_parts)
        if name in limited_config_shared_tasks:
            store_parts = limited_config_shared_parts(task, store_parts)
        if name not in skip_new_version_schema:
            store_parts = reorganize_parts(task, store_parts)
        return store_parts

    analysis_inst.x.store_parts_modifiers = {
        "hbw_parts": hbw_parts,
    }

    return analysis_inst
