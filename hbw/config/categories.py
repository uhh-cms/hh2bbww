# coding: utf-8

"""
Definition of categories.

Categorizer modules (used to determine category masks) are defined in hbw.selection.categories

Ids for combinations of categories are built as the sum of category ids.
To avoid reusing category ids, each category block (e.g. leptons, jets, ...) uses ids of a different
power of 10.

power of 10 | category block

0: free (only used for inclusive category)
1: jet (resolved vs boosted)
2: bjet (1 vs geq 2)
3: lepton
4: dnn
5: gen-level leptons (not combined with other categories)
"""

from collections import OrderedDict

import law

from time import time

from columnflow.config_util import create_category_combinations
from columnflow.ml import MLModel
from hbw.util import call_once_on_config

import order as od

logger = law.logger.get_logger(__name__)


@call_once_on_config()
def add_gen_categories(config: od.Config) -> None:
    gen_0lep = config.add_category(  # noqa
        name="gen_0lep",
        id=100000,
        selection="catid_gen_0lep",  # this should not be called!
        label="No gen lepton",
    )
    gen_1lep = config.add_category(
        name="gen_1lep",
        id=200000,
        label="1 gen lepton",
    )
    gen_1lep.add_category(
        name="gen_1e",
        id=300000,
        selection="catid_gen_1e",
        label="1 gen electron",
    )
    gen_1lep.add_category(
        name="gen_1mu",
        id=400000,
        selection="catid_gen_1mu",
        label="1 gen muon",
    )
    gen_1lep.add_category(
        name="gen_1tau",
        id=500000,
        selection="catid_gen_1tau",
        label="1 gen tau",
    )
    gen_2lep = config.add_category(  # noqa
        name="gen_geq2lep",
        id=600000,
        selection="catid_geq_2_gen_leptons",
        label=r"$\geq 2$ gen leptons",
    )


@call_once_on_config()
def add_abcd_categories(config: od.Config) -> None:
    config.add_category(
        name="sr",
        id=1,
        selection="catid_sr",
    )
    config.add_category(
        name="fake",
        id=2,
        selection="catid_fake",
    )
    config.add_category(
        name="highmet",
        id=3,
        selection="catid_highmet",
        label=r"MET \geq 20",
    )
    config.add_category(
        name="lowmet",
        id=6,
        selection="catid_lowmet",
        label=r"MET < 20",
    )


@call_once_on_config()
def add_lepton_categories(config: od.Config) -> None:
    config.x.lepton_channels = {
        "sl": ("1e", "1mu"),
        "dl": ("2e", "2mu", "emu"),
    }[config.x.lepton_tag]

    config.add_category(
        name="incl",
        id=0,
        selection="catid_selection_incl",
        label="Inclusive",
    )

    cat_1e = config.add_category(  # noqa
        name="1e",
        id=10,
        selection="catid_selection_1e",
        label="1 Electron",
    )

    cat_1mu = config.add_category(  # noqa
        name="1mu",
        id=20,
        selection="catid_selection_1mu",
        label="1 Muon",
    )
    # dl categories
    cat_2e = config.add_category(  # noqa
        name="2e",
        id=30,
        selection="catid_selection_2e",
        label="2 Electron",
    )

    cat_2mu = config.add_category(  # noqa
        name="2mu",
        id=40,
        selection="catid_selection_2mu",
        label="2 Muon",
    )

    cat_emu = config.add_category(  # noqa
        name="emu",
        id=50,
        selection="catid_selection_emu",
        label="1 Electron 1 Muon",
    )


@call_once_on_config()
def add_jet_categories(config: od.Config) -> None:
    cat_resolved = config.add_category(  # noqa
        name="resolved",
        id=100,
        selection="catid_resolved",
        label="resolved",
    )
    cat_boosted = config.add_category(  # noqa
        name="boosted",
        id=200,
        selection="catid_boosted",
        label="boosted",
    )

    cat_1b = config.add_category(  # noqa
        name="1b",
        id=300,
        selection="catid_1b",
        label="1b",
    )
    cat_2b = config.add_category(  # noqa
        name="2b",
        id=600,
        selection="catid_2b",
        label="2b",
    )

@call_once_on_config()
def add_trigger_categories(config: od.Config) -> None:
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
        label="Muon\n(orthogonal measurement)",
    )
    cat_trig_ele_orth = config.add_category(  # noqa
        name="trig_ele_orth",
        id=4000,
        selection="catid_trigger_orth_ele",
        label="Electron\n(orthogonal measurement)",
    )

@call_once_on_config()
def add_categories_selection(config: od.Config) -> None:
    """
    Adds categories to a *config*, that are typically produced in `SelectEvents`.
    """

    # adds categories based on the existence of gen particles
    add_gen_categories(config)

    # adds categories for ABCD background estimation
    add_abcd_categories(config)

    # adds categories based on number of leptons
    add_lepton_categories(config)


def name_fn(root_cats):
    cat_name = "__".join(cat.name for cat in root_cats.values())
    return cat_name


def kwargs_fn(root_cats):
    kwargs = {
        "id": sum([c.id for c in root_cats.values()]),
        "label": ", ".join([c.name for c in root_cats.values()]),
        "aux": {
            "root_cats": {key: value.name for key, value in root_cats.items()},
        },
    }
    return kwargs


@call_once_on_config()
def add_categories_production(config: od.Config) -> None:
    """
    Adds categories to a *config*, that are typically produced in `ProduceColumns`.
    """
    if config.has_tag("add_categories_ml_called"):
        logger.warning("We should not call *add_categories_production* when also building ML categories")
        # when ML categories already exist, don't do anything
        return
    #
    # switch existing categories to different production module
    #

    cat_1e = config.get_category("1e")
    cat_1e.selection = "catid_1e"

    cat_1mu = config.get_category("1mu")
    cat_1mu.selection = "catid_1mu"

    cat_2e = config.get_category("2e")
    cat_2e.selection = "catid_2e"

    cat_2mu = config.get_category("2mu")
    cat_2mu.selection = "catid_2mu"

    cat_emu = config.get_category("emu")
    cat_emu.selection = "catid_emu"

    add_jet_categories(config)

    #
    # define all combinations of categories
    #

    category_blocks = OrderedDict({
        "lepid": [config.get_category("sr"), config.get_category("fake")],
        # "met": [config.get_category("highmet"), config.get_category("lowmet")],
        "lep": [config.get_category(lep_ch) for lep_ch in config.x.lepton_channels],
        "jet": [config.get_category("resolved"), config.get_category("boosted")],
        "b": [config.get_category("1b"), config.get_category("2b")],
    })
    t0 = time()
    n_cats = create_category_combinations(
        config,
        category_blocks,
        name_fn=name_fn,
        kwargs_fn=kwargs_fn,
        skip_existing=False,  # there should be no existing sub-categories
    )
    logger.info(f"Number of produced category insts: {n_cats} (took {(time() - t0):.3f}s)")


@call_once_on_config()
def add_categories_ml(config, ml_model_inst):
    if config.has_tag("add_categories_production_called"):
        raise Exception("We should not call *add_categories_production* when also building ML categories")
    #
    # prepare non-ml categories
    #

    cat_1e = config.get_category("1e")
    cat_1e.selection = "catid_1e"

    cat_1mu = config.get_category("1mu")
    cat_1mu.selection = "catid_1mu"

    cat_2e = config.get_category("2e")
    cat_2e.selection = "catid_2e"

    cat_2mu = config.get_category("2mu")
    cat_2mu.selection = "catid_2mu"

    cat_emu = config.get_category("emu")
    cat_emu.selection = "catid_emu"

    add_jet_categories(config)

    #
    # add parent ml model categories
    #

    # if not already done, get the ml_model instance
    if isinstance(ml_model_inst, str):
        ml_model_inst = MLModel.get_cls(ml_model_inst)(config)

    # add ml categories directly to the config
    # NOTE: this is a bit dangerous, because our ID depends on the MLModel, but
    #       we can reconfigure our MLModel after having created these categories
    ml_categories = []
    for i, proc in enumerate(ml_model_inst.processes):
        ml_categories.append(config.add_category(
            # NOTE: name and ID is unique as long as we don't use
            #       multiple ml_models simutaneously
            name=f"ml_{proc}",
            id=(i + 1) * 1000,
            selection=f"catid_ml_{proc}",
            label=f"ml_{proc}",
        ))

    #
    # create combination of categories
    #

    # NOTE: building this many categories takes forever: has to be improved...
    category_blocks = OrderedDict({
        "lepid": [config.get_category("sr"), config.get_category("fake")],
        # "met": [config.get_category("highmet"), config.get_category("lowmet")],
        "lep": [config.get_category(lep_ch) for lep_ch in config.x.lepton_channels],
        "jet": [config.get_category("resolved"), config.get_category("boosted")],
        "b": [config.get_category("1b"), config.get_category("2b")],
        "dnn": ml_categories,
    })

    # # NOTE: temporary solution: only build DNN leafs
    # combined_categories = [cat for cat in config.get_leaf_categories() if len(cat.parent_categories) != 0]
    # category_blocks = OrderedDict({
    #     "leafs": combined_categories,
    #     "dnn": ml_categories,
    # })

    t0 = time()
    # create combination of categories
    n_cats = create_category_combinations(
        config,
        category_blocks,
        name_fn=name_fn,
        kwargs_fn=kwargs_fn,
        skip_existing=True,
    )
    logger.info(f"Number of produced ml category insts: {n_cats} (took {(time() - t0):.3f}s)")
