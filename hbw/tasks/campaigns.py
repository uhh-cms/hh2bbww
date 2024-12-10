# coding: utf-8

"""
Custom tasks for creating and managing campaigns.
"""

from collections import defaultdict
from functools import cached_property
import importlib

import law
import luigi

from columnflow.tasks.framework.base import AnalysisTask
from hbw.tasks.base import HBWTask


logger = law.logger.get_logger(__name__)


campaign_map = {
    "c17": {
        "cmsdb.campaigns.run2_2017_nano_v9": "campaign_run2_2017_nano_v9",
    },
    "c22pre": {
        "cmsdb.campaigns.run3_2022_preEE_nano_v12": "campaign_run3_2022_preEE_nano_v12",
        "cmsdb.campaigns.run3_2022_preEE_nano_v13": "campaign_run3_2022_preEE_nano_v13",
    },
    "c22post": {
        "cmsdb.campaigns.run3_2022_postEE_nano_v12": "campaign_run3_2022_postEE_nano_v12",
        "cmsdb.campaigns.run3_2022_postEE_nano_v13": "campaign_run3_2022_postEE_nano_v13",
        "cmsdb.campaigns.run3_2022_postEE_nano_uhh_v12": "campaign_run3_2022_postEE_nano_uhh_v12",
    },
}


class BuildCampaignSummary(
    HBWTask,
    AnalysisTask,
):

    config = luigi.Parameter()
    # TODO: set campaigns as part of this function instead of configuring in the config?

    recreate_backup_summary = luigi.BoolParameter(default=False)

    def requires(self):
        return {}

    def store_parts(self):
        parts = super().store_parts()

        # add the config name
        parts.insert_after("task_family", "config", self.config)

        return parts

    @cached_property
    def campaigns(self):
        if self.config not in campaign_map:
            raise ValueError(f"Unknown config {self.config}")
        return campaign_map[self.config]

    @cached_property
    def campaign_insts(self):
        return [
            getattr(importlib.import_module(mod), campaign).copy()
            for mod, campaign in self.campaigns.items()
        ]

    dataset_from_uhh_identifier = {
        # TODO: use DY from uhh campaign
        # "dy_m10to50_amcatnlo",
        # "dy_m4to10_amcatnlo",
        "ttw_",
        "ttz_",
    }

    def get_dataset_prio(self, dataset_name, campaign):
        """
        If dataset should be overwritten from this campaign, return True.
        Otherwise, return False.
        """
        if "uhh" in campaign.name and any(
            dataset_identifier in dataset_name
            for dataset_identifier in self.dataset_from_uhh_identifier
        ):
            return True

        return False

    def output(self):
        output = {
            "dataset_summary": self.target("dataset_summary.yaml"),
            "campaign_summary": self.target("campaign_summary.yaml"),
            "hbw_campaign_inst": self.target("hbw_campaign_inst.pickle"),
        }
        return output

    @cached_property
    def dataset_summary(self):
        dataset_summary = defaultdict(dict)
        used_datasets = set()
        # create campaign summary with one key per dataset (to fulfill dataset uniqueness)
        for campaign in self.campaign_insts:
            for dataset in campaign.datasets:
                if dataset.name not in used_datasets or self.get_dataset_prio(dataset.name, campaign):
                    dataset_summary[dataset.name] = {
                        "campaign": campaign.name,
                        "n_events": dataset.n_events,
                        "n_files": dataset.n_files,
                    }
                    used_datasets.add(dataset.name)

        return dict(dataset_summary)

    @cached_property
    def campaign_summary(self,):
        campaign_summary = {
            campaign.name: {} for campaign in self.campaign_insts
        }

        for dataset, dataset_info in self.dataset_summary.items():
            campaign_summary[dataset_info["campaign"]][dataset] = {
                "n_events": dataset_info["n_events"],
                "n_files": dataset_info["n_files"],
            }
        return campaign_summary

    def get_custom_campaign(self):
        hbw_campaign_inst = self.campaign_insts[0].copy()
        hbw_campaign_inst.clear_datasets()
        for campaign_inst in self.campaign_insts:
            campaign_info = self.campaign_summary[campaign_inst.name]
            for dataset in campaign_info.keys():
                dataset_inst = campaign_inst.get_dataset(dataset)
                dataset_inst.x.campaign = campaign_inst.name
                hbw_campaign_inst.add_dataset(dataset_inst)

        hbw_campaign_inst.x.campaigns = list(self.campaigns)

        return hbw_campaign_inst

    from hbw.util import timeit_multiple

    @timeit_multiple
    def run(self):
        output = self.output()

        # cross check if the dataset summary did change
        backup_target = self.target("backup_dataset_summary.yaml")
        if backup_target.exists():
            backup_dataset_summary = backup_target.load(formatter="yaml")
            if backup_dataset_summary != self.dataset_summary:
                from hbw.util import gather_dict_diff
                logger.warning(
                    "Backup dataset summary does not match the current one \n"
                    f"{gather_dict_diff(backup_dataset_summary, self.dataset_summary)}",
                )
                if self.recreate_backup_summary:
                    logger.warning("Recreating backup dataset summary")
                    backup_target.dump(self.dataset_summary, formatter="yaml")
                else:
                    logger.warning(
                        "Run the following command to recreate the backup dataset summary:\n"
                        f"law run {self.task_family} --recreate_backup_summary --config {self.config} --remove-output 0,a,y",  # noqa
                    )
        else:
            logger.warning("No backup dataset summary found, creating one now")
            backup_target.dump(self.dataset_summary, formatter="yaml")

        output["dataset_summary"].dump(self.dataset_summary, formatter="yaml")
        output["campaign_summary"].dump(self.campaign_summary, formatter="yaml")

        import sys
        orig_rec_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(orig_rec_limit, 100000))
        output["hbw_campaign_inst"].dump(self.get_custom_campaign(), formatter="pickle")
        sys.setrecursionlimit(orig_rec_limit)
