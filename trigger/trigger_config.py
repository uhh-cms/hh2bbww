# coding: utf-8

"""
Functions adding trigger related attributes to main analysis config:
triggers, reference triggers, trigger categories, variables for trigger studies
"""

import order as od
import law

from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT
from hbw.util import call_once_on_config

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


# add trigger columns and adjust binning for certain variables
@call_once_on_config()
def add_trigger_columns(config: od.Config) -> None:
    """
    Adds trigger columns to the config
    """
    config.add_variable(
        name="trig_bits_mu",
        aux={
            "axis_type": "strcat",
        },
        x_title="Trigger names",
    )
    config.add_variable(
        name="trig_bits_e",
        aux={
            "axis_type": "strcat",
        },
        x_title="Trigger names",
    )
    config.add_variable(
        name="trig_weights",
        binning=(100, -5, 5),
        x_title="event weights",
    )
    config.add_variable(
        name="trigger_sf_weights",
        binning=(50, 0, 2),
        x_title=("trigger scale factors")
    )

    # adjust binning for certain variables
    config.variables.remove("electron_pt")
    config.variables.remove("muon_pt")
    for obj in ["Electron", "Muon"]:
        config.add_variable(
            name=f"{obj.lower()}_pt",
            expression=f"{obj}.pt[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(600, 0., 350.),
            unit="GeV",
            x_title=obj + r" $p_{T}$",
        )

    config.variables.remove("ht")
    config.add_variable(
        name="ht",
        expression=lambda events: ak.sum(events.Jet.pt, axis=1),
        aux={"inputs": {"Jet.pt"}},
        binning=(600, 0, 1200),
        unit="GeV",
        x_title="HT",
    )


# add triggers and trigger related attributes to the main analysis config
@call_once_on_config()
def add_trigger_config(config: od.Config, **kwargs) -> None:
    """
    Adds trigger related attributes to the main analysis config
    """
    year = kwargs["year"]
    config.x.keep_columns["cf.ReduceEvents"] |= {"HLT.*"}

    # set triggers for trigger studies
    if year == 2017:
        config.x.trigger = {
            "e": ["Ele35_WPTight_Gsf"],
            "mu": ["IsoMu27"],
        }
        config.x.ref_trigger = {
            "e": "IsoMu27",
            "mu": "Ele35_WPTight_Gsf",
        }
    elif year == 2018:
        config.x.trigger = {
            "e": "Ele32_WPTight_Gsf",
            "mu": "IsoMu24",
        }
        config.x.ref_trigger = {
            "e": "IsoMu24",
            "mu": "Ele32_WPTight_Gsf",
        }
    elif year == 2022:
        config.x.trigger = {
            "e": [
                "Ele30_WPTight_Gsf",
                "Ele28_eta2p1_WPTight_Gsf_HT150",
                "Ele15_IsoVVVL_PFHT450",
                "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
            ],
            "mu": [
                "IsoMu24",
                "Mu50",
                "Mu15_IsoVVVL_PFHT450",
                "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65",
            ],
        }
        config.x.ref_trigger = {
            "e": "IsoMu24",
            "mu": "Ele30_WPTight_Gsf",
        }
    else:
        raise NotImplementedError(f"Trigger for year {year} is not defined.")

    # short labels for triggers with long names
    config.x.trigger_short = {
        "Ele30_WPTight_Gsf": "Ele30Tight",
        "Ele28_eta2p1_WPTight_Gsf_HT150": "Ele28eta2p1TightHT150",
        "Ele15_IsoVVVL_PFHT450": "Ele15IsoVVVLHT450",
        "QuadPFJet70_50_40_35_PFBTagParticleNet_2BTagSum0p65": "QuadPFJet2Btag",
        "IsoMu24": "IsoMu24",
        "Mu50": "Mu50",
        "Mu15_IsoVVVL_PFHT450": "Mu15IsoVVVLHT450",
    }

    # add trigger columns
    add_trigger_columns(config)
