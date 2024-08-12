# coding: utf-8

"""
Functions adding trigger related attributes to main analysis config:
triggers, reference triggers, trigger categories, variables for trigger studies
"""

import order as od
import law

from hbw.util import call_once_on_config

logger = law.logger.get_logger(__name__)


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
        name="trig_bits_orth_mu",
        aux={
            "axis_type": "strcat",
        },
        x_title="Trigger names (orthogonal)",
    )
    config.add_variable(
        name="trig_bits_e",
        aux={
            "axis_type": "strcat",
        },
        x_title="Trigger names",
    )
    config.add_variable(
        name="trig_bits_orth_e",
        aux={
            "axis_type": "strcat",
        },
        x_title="Trigger names (orthogonal)",
    )


@call_once_on_config()
def add_trigger_categories(config: od.Config) -> None:
    """
    Adds trigger categories to the config
    """
    # mc truth categories
    cat_trig_mu = config.add_category(  # noqa
        name="trig_mu",
        id=1000,
        selection="catid_trigger_mu",
        label="Muon\n(MC truth)",
    )
    cat_trig_ele = config.add_category(  # noqa
        name="trig_ele",
        id=2000,
        selection="catid_trigger_ele",
        label="Electron\n(MC truth)",
    )
    # orthogonal categories
    cat_trig_mu_orth = config.add_category(  # noqa
        name="trig_mu_orth",
        id=3000,
        selection="catid_trigger_orth_mu",
        label="Muon\northogonal\nmeasurement",
    )
    cat_trig_ele_orth = config.add_category(  # noqa
        name="trig_ele_orth",
        id=4000,
        selection="catid_trigger_orth_ele",
        label="Electron\northogonal\nmeasurement",
    )


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
