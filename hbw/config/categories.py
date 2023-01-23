# coding: utf-8

"""
Definition of categories.
"""

import order as od


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

    #
    # define additional categories
    #

    # TODO: we might need a more systematic approach as soon as the number of
    #       categories grows by the DNN categorization

    # Electrons

    cat_1e_resolved = cat_1e.add_category(
        name="1e_resolved",
        id=1100,
        selection="catid_1e_resolved",
        label="1 Electron, resolved",
    )
    cat_1e_resolved_1b = cat_1e_resolved.add_category(
        name="1e_resolved_1b",
        id=1110,
        selection="catid_1e_resolved_1b",
        label="1 Electron, resolved, 1b",
    )
    cat_1e_resolved_2b = cat_1e_resolved.add_category(
        name="1e_resolved_2b",
        id=1120,
        selection="catid_1e_resolved_2b",
        label="1 Electron, resolved, 2b",
    )
    cat_1e_boosted = cat_1e.add_category(
        name="1e_boosted",
        id=1500,
        selection="catid_1e_boosted",
        label="1 Electron, boosted",
    )

    # Muons

    cat_1mu_resolved = cat_1mu.add_category(
        name="1mu_resolved",
        id=2100,
        selection="catid_1mu_resolved",
        label="1 Muon, resolved",
    )
    cat_1mu_resolved_1b = cat_1mu_resolved.add_category(
        name="1mu_resolved_1b",
        id=2110,
        selection="catid_1mu_resolved_1b",
        label="1 Muon, resolved, 1b",
    )
    cat_1mu_resolved_2b = cat_1mu_resolved.add_category(
        name="1mu_resolved_2b",
        id=2120,
        selection="catid_1mu_resolved_2b",
        label="1 Muon, resolved, 2b",
    )
    cat_1mu_boosted = cat_1mu.add_category(
        name="1mu_boosted",
        id=2500,
        selection="catid_1mu_boosted",
        label="1 Muon, boosted",
    )

    # resolved and boosted as parent category
    cat_resolved = config.add_category(
        name="resolved",
        id=3100,
        selection="catid_resolved",
        label="resolved",
    )
    cat_resolved_1b = cat_resolved.add_category(
        name="resolved_1b",
        id=2110,
        selection="catid_resolved_1b",
        label="resolved, 1b",
    )
    cat_resolved_2b = cat_resolved.add_category(
        name="resolved_2b",
        id=2120,
        selection="catid_resolved_2b",
        label="resolved, 2b",
    )
    cat_boosted = config.add_category(
        name="boosted",
        id=3500,
        selection="catid_boosted",
        label="boosted",
    )

    # add leaf categories to reduce number of leaf categories
    cat_resolved_1b.add_category(cat_1mu_resolved_1b)
    cat_resolved_1b.add_category(cat_1e_resolved_1b)
    cat_resolved_2b.add_category(cat_1mu_resolved_2b)
    cat_resolved_2b.add_category(cat_1e_resolved_2b)
    cat_boosted.add_category(cat_1mu_boosted)
    cat_boosted.add_category(cat_1e_boosted)
