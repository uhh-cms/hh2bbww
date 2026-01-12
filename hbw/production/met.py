# coding: utf-8

"""
MET corrections Producer.
"""

from __future__ import annotations

from hbw.util import IF_DY
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.calibration.cms.met import met_phi
from columnflow.production.cms.dy import recoil_corrected_met
from columnflow.production import Producer, producer

ak = maybe_import("awkward")

recoil_corrected_met.njet_column = "njet_for_recoil"


@producer(
    uses={
        met_phi,
        IF_DY(recoil_corrected_met),
    },
    produces={
        met_phi,
        IF_DY(recoil_corrected_met),
        "RecoilCorrMET.{pt,phi}",
    },
    version=0,
)
def met_phi_prod(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer that applies the MET phi correction to the events.

    :param events: Awkward array containing events to process
    """
    events = self[met_phi](events, **kwargs)
    if self.has_dep(recoil_corrected_met):
        events = self[recoil_corrected_met](events, **kwargs)
    else:
        # copy the uncorrected values to new fields for convenience
        events = set_ak_column(events, "RecoilCorrMET.pt", events.PuppiMET.pt)
        events = set_ak_column(events, "RecoilCorrMET.phi", events.PuppiMET.phi)

    return events


from hbw.util import call_once_on_config
import order as od


@call_once_on_config
def add_met_phi_variable(config: od.Config):
    """
    Add the MET phi variable to the config.

    This function adds the MET phi variable to the config, which is used in the
    :py:class:`~columnflow.calibration.met.MetPhiConfig` class.
    """
    config.add_variable(
        name="met_phi_corrected",
        expression="PuppiMET.phi",
        binning=(40, -3.2, 3.2),
        x_title=r"$\phi(\mathrm{MET})$ (after MET phi correction)",
        aux={"overflow": True},
    )
    config.add_variable(
        name="met_pt_corrected",
        expression="PuppiMET.pt",
        binning=(40, 0, 400),
        x_title=r"$p_T(\mathrm{MET})$ (after MET phi correction)",
        aux={"overflow": True},
    )
    config.add_variable(
        name="met_phi_uncorrected",
        expression="PuppiMET.phi_metphi_uncorrected",
        binning=(40, -3.2, 3.2),
        x_title=r"$\phi(\mathrm{MET})$ (before MET phi correction)",
        aux={"overflow": True},
    )
    config.add_variable(
        name="met_pt_uncorrected",
        expression="PuppiMET.pt_metphi_uncorrected",
        binning=(40, 0, 400),
        x_title=r"$p_T(\mathrm{MET})$ (before MET phi correction)",
        aux={"overflow": True},
    )
    config.add_variable(
        name="met_phi_recoil_corrected",
        expression="RecoilCorrMET.phi",
        binning=(40, -3.2, 3.2),
        x_title=r"$\phi(\mathrm{MET})$ (after MET phi + recoil correction)",
        aux={"overflow": True},
    )
    config.add_variable(
        name="met_pt_recoil_corrected",
        expression="RecoilCorrMET.pt",
        binning=(40, 0, 400),
        x_title=r"$p_T(\mathrm{MET})$ (after MET phi + recoil correction)",
        aux={"overflow": True},
    )


@met_phi_prod.init
def met_phi_prod_init(self: Producer, **kwargs) -> None:
    add_met_phi_variable(self.config_inst)
