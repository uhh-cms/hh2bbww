# coding: utf-8

"""
Configuration of the HH -> bbWW analysis.
"""

from __future__ import annotations

import os
import importlib

import law
import order as od

from hbw.util import timeit_multiple

thisdir = os.path.dirname(os.path.abspath(__file__))

logger = law.logger.get_logger(__name__)


@timeit_multiple
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
        "$HBW_BASE/sandboxes/venv_ml_keras.sh",
    ]

    # cmssw sandboxes that should be bundled for remote jobs in case they are needed
    analysis_inst.set_aux("cmssw_sandboxes", [
    #    "$CF_BASE/sandboxes/cmssw_default.sh",
    ])

    # clear the list when cmssw bundling is disabled
    if not law.util.flag_to_bool(os.getenv("HBW_BUNDLE_CMSSW", "1")):
        del analysis_inst.x.cmssw_sandboxes[:]

    # config groups for conveniently looping over certain configs
    # (used in wrapper_factory)
    analysis_inst.set_aux("config_groups", {})

    #
    # define configs
    #

    from hbw.config.config_run2 import add_config

    def add_lazy_config(
        campaigns: dict[str, str],
        config_name: str,
        config_id: int,
        **kwargs,
    ):
        """
        Helper to add configs to the analysis lazily.
        Calling *add_lazy_config* will add two configs to the analysis:
        - one with the given *config_name* and *config_id*,
        - one with the *config_id* +1 and *config_name* with the character "c" replaced by "l",
            which is used to limit the number of dataset files to 2 per dataset.
        """
        def create_factory(
            config_id: int,
            final_config_name: str,
            limit_dataset_files: int | None = None,
        ):
            @timeit_multiple
            def analysis_factory(configs: od.UniqueObjectIndex):
                hbw_campaign_inst = None

                for mod, campaign in campaigns.items():
                    # import the campaign
                    mod = importlib.import_module(mod)
                    if not hbw_campaign_inst:
                        # copy the main campaign
                        hbw_campaign_inst = getattr(mod, campaign).copy()
                    else:
                        # add datasets to the main campaign
                        campaign_inst = getattr(mod, campaign).copy()
                        for dataset in list(campaign_inst.datasets):
                            dataset.x.campaign = campaign
                            if not hbw_campaign_inst.has_dataset(dataset.name):
                                hbw_campaign_inst.add_dataset(dataset)

                return add_config(
                    analysis_inst,
                    hbw_campaign_inst,
                    config_name=final_config_name,
                    config_id=config_id,
                    limit_dataset_files=limit_dataset_files,
                    **kwargs,
                )
            return analysis_factory

        analysis_inst.configs.add_lazy_factory(
            config_name,
            create_factory(config_id, config_name),
        )
        limited_config_name = config_name.replace("c", "l")
        analysis_inst.configs.add_lazy_factory(
            limited_config_name,
            create_factory(config_id + 1, limited_config_name, 2),
        )

    # 2017
    add_lazy_config(
        {
            "cmsdb.campaigns.run2_2017_nano_v9": "campaign_run2_2017_nano_v9",
        },
        "c17",
        1700,
    )

    # 2022 preEE
    add_lazy_config(
        {
            "cmsdb.campaigns.run3_2022_preEE_nano_v12": "campaign_run3_2022_preEE_nano_v12",
            "cmsdb.campaigns.run3_2022_preEE_nano_v13": "campaign_run3_2022_preEE_nano_v13",
        },
        "c22pre",
        2200,
    )

    # 2022 postEE
    add_lazy_config(
        {
            "cmsdb.campaigns.run3_2022_postEE_nano_v12": "campaign_run3_2022_postEE_nano_v12",
            "cmsdb.campaigns.run3_2022_postEE_nano_v13": "campaign_run3_2022_postEE_nano_v13",
        },
        "c22post",
        2210,
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
        "plot", "shift_sources", "shifts", "datasets",
        # MLTraining
        "calib", "sel", "prod",
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
        # TODO: move "producer" after "task_family" since it is used in ProduceColumn
        # to be decided for "ml_data", "ml_model", "inf_model" (used for multiple tasks)
        parts_order_start = [
            "analysis",
            "calibrator", "calibrators", "calib",
            "selector", "sel",
            "producer", "producers", "prod",
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
