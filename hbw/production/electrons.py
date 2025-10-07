# coding: utf-8

"""
Production modules related to electrons.
"""

from __future__ import annotations

from functools import partial


import law
from columnflow.util import maybe_import

from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")


set_ak_bool = partial(set_ak_column, value_type=np.bool_)


logger = law.logger.get_logger(__name__)


@producer(
    uses={"run", "Electron.{pt,eta,seediEtaOriX,seediPhiOriY}"},
    produces={"Electron.ee_leak_veto_mask"},
)
def electron_ee_veto(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer that identifies electrons that are affected by the EE leak veto.
    Should only be used for the 2022 post-EE campaign.

    Resources:
    - https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis?rev=169#From_ECAL_and_EGM
    """
    n_electrons = ak.count(events.Electron.pt)
    logger.debug("Running electron EE veto")
    ee_leak_veto_mask = ~(
        (events.Electron["eta"] > 1.556) &
        (events.Electron.seediPhiOriY > 72) &
        (events.Electron.seediEtaOriX < 45)
    )
    logger.debug(f"{ak.sum(~ee_leak_veto_mask)} / {n_electrons} electrons affected by EE leak veto")

    if self.dataset_inst.is_data:
        # issue with noisy EB channel in one fill (8456).
        # Details in this talk: https://indico.cern.ch/event/1397512/#3-run-3ecal-dpg-on-the-hot-tow
        affected_runs = {362430, 362433, 362434, 362435, 362436, 362437, 362438, 362439}
        runs = set(events.run)
        if relevant_runs := runs.intersection(affected_runs):
            logger.debug(f"Fill 8456 veto applied for runs {relevant_runs}.")
            from hbw.util import ak_any
            fill_8456_veto_mask = ak.where(
                ak_any([events.run == r for r in affected_runs]),
                ~(
                    (abs(events.Electron["eta"]) < 1.556) &
                    (events.Electron["pt"] > 700) &
                    (events.Electron["pt"] < 900) &
                    (events.Electron.seediEtaOriX == -21) &
                    (events.Electron.seediPhiOriY == 260)
                ),
                events.Electron["pt"] > 0,  # default to True for all other runs (no veto applied
            )
            logger.debug(f"{ak.sum(~fill_8456_veto_mask)} / {n_electrons} electrons affected by fill 8456 veto")
            ee_leak_veto_mask = ee_leak_veto_mask & fill_8456_veto_mask
            logger.debug(
                f"{ak.sum(~ee_leak_veto_mask)} / {n_electrons} electrons affected by "
                "EE leak veto after applying fill 8456 veto",
            )

    events = set_ak_bool(events, "Electron.ee_leak_veto_mask", ee_leak_veto_mask)

    return events


@electron_ee_veto.skip
def electron_ee_veto_skip_func(self: Producer):
    if "UL" in self.dataset_inst.x.campaign:
        return True
    # skip for all exept for the 2022 post-EE campaign
    if self.config_inst.campaign.has_tag("postEE"):
        return False
    return True
