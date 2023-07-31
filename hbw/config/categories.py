# coding: utf-8

"""
Definition of categories.
"""

from collections import OrderedDict

import law

from columnflow.config_util import create_category_combinations

import order as od

logger = law.logger.get_logger(__name__)


def add_categories_selection(config: od.Config) -> None:
    """
    Adds categories to a *config*, that are typically produced in `SelectEvents`.
    """
    config.add_category(
        name="incl",
        id=1,
        selection="catid_selection_incl",
        label="Inclusive",
    )

    cat_1e = config.add_category(  # noqa
        name="1e",
        id=1000,
        selection="catid_selection_1e",
        label="1 Electron",
    )

    cat_1mu = config.add_category(  # noqa
        name="1mu",
        id=2000,
        selection="catid_selection_1mu",
        label="1 Muon",
    )
    # dl categories
    cat_ee = config.add_category(  # noqa
        name="ee",
        id=3000,
        selection="catid_selection_ee",
        label="2 Electron",
    )

    cat_2mu = config.add_category(  # noqa
        name="2mu",
        id=4000,
        selection="catid_selection_2mu",
        label="2 Muon",
    )

    cat_emu = config.add_category(  # noqa
        name="emu",
        id=5000,
        selection="catid_selection_emu",
        label="1 Electron 1 Muon",
    )


def name_fn(root_cats):
    cat_name = "__".join(cat.name for cat in root_cats.values())
    return cat_name


def kwargs_fn(root_cats):
    kwargs = {
        "id": sum([c.id for c in root_cats.values()]),
        "label": ", ".join([c.name for c in root_cats.values()]),
    }
    return kwargs


def add_categories_production(config: od.Config) -> None:
    """
    Adds categories to a *config*, that are typically produced in `ProduceColumns`.
    """
    #
    # switch existing categories to different production module
    #

    cat_1e = config.get_category("1e")
    cat_1e.selection = "catid_1e"

    cat_1mu = config.get_category("1mu")
    cat_1mu.selection = "catid_1mu"

    cat_ee = config.get_category("ee")
    cat_ee.selection = "catid_ee"

    cat_2mu = config.get_category("2mu")
    cat_2mu.selection = "catid_2mu"

    cat_emu = config.get_category("emu")
    cat_emu.selection = "catid_emu"

    #
    # define additional 'main' categories
    #

    cat_resolved = config.add_category(
        name="resolved",
        id=10,
        selection="catid_resolved",
        label="resolved",
    )
    cat_boosted = config.add_category(
        name="boosted",
        id=20,
        selection="catid_boosted",
        label="boosted",
    )

    cat_1b = config.add_category(
        name="1b",
        id=100,
        selection="catid_1b",
        label="1b",
    )
    cat_2b = config.add_category(
        name="2b",
        id=200,
        selection="catid_2b",
        label="2b",
    )

    #
    # define all combinations of categories
    #

    category_blocks = OrderedDict({
        "lep": [cat_1e, cat_1mu],
        "jet": [cat_resolved, cat_boosted],
        "b": [cat_1b, cat_2b],
    })

    n_cats = create_category_combinations(
        config,
        category_blocks,
        name_fn=name_fn,
        kwargs_fn=kwargs_fn,
        skip_existing=False,  # there should be no existing sub-categories
    )
    logger.info(f"Number of produced category insts: {n_cats}")


def add_categories_ml(config, ml_model_inst):

    # add ml categories directly to the config
    ml_categories = []
    for i, proc in enumerate(ml_model_inst.processes):
        ml_categories.append(config.add_category(
            # NOTE: name and ID is unique as long as we don't use
            #       multiple ml_models simutaneously
            name=f"ml_{proc}",
            id=(i + 1) * 10000,
            selection=f"catid_ml_{proc}",
            label=f"ml_{proc}",
        ))

    category_blocks = OrderedDict({
        "lep": [config.get_category("1e"), config.get_category("1mu")],
        "jet": [config.get_category("resolved"), config.get_category("boosted")],
        "b": [config.get_category("1b"), config.get_category("2b")],
        "dnn": ml_categories,
    })

    # create combination of categories
    n_cats = create_category_combinations(
        config,
        category_blocks,
        name_fn=name_fn,
        kwargs_fn=kwargs_fn,
        skip_existing=True,
    )
    logger.info(f"Number of produced ml category insts: {n_cats}")

    # TODO unfinished
