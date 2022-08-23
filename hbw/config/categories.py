# coding: utf-8

"""
Definition of categories.
"""

import order as od


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    config.add_category(
        name="incl",
        id=1,
        selection="sel_incl",
        label="inclusive",
    )
    cat_e = config.add_category(
        name="1e",
        id=100,
        selection="sel_1e",
        label="1 electron",
    )
    cat_e.add_category(
        name="1e_eq1b",
        id=110,
        selection="sel_1e_eq1b",
        label="1e, 1 b-tag",
    )
    cat_e.add_category(
        name="1e_ge2b",
        id=120,
        selection="sel_1e_ge2b",
        label=r"1e, $\geq$ 2 b-tags",
    )
