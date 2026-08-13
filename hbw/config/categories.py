
# coding: utf-8

"""
Definition of categories.

Categorizer modules (used to determine category masks) are defined in hbw.categorization.categories

Ids for combinations of categories are built as the sum of category ids.
To avoid reusing category ids, each category block (e.g. leptons, jets, ...) uses ids of a different
power of 10.

power of 10 | category block

0: sr vs fake (SL), sr vs dycr vs ttcr (DL)
1: jet (resolved vs boosted)
2: bjet (1 vs geq 2)
3: lepton
4: dnn
5: gen-level leptons (not combined with other categories)
"""

# @call_once_on_config()
# def add_categories_ml(config, ml_model_inst,
#     add_jet_categories_func=None,
# ):
#     if config.has_tag("add_categories_production_called"):
#         raise Exception("We should not call *add_categories_production* when also building ML categories")
#     if not add_jet_categories_func:
#         add_jet_categories_func = config.x.add_jet_categories_func

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
    # NOTE: this should instead be covered by process ids if necessary
    gen_0lep = config.add_category(  # noqa: F841
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
    gen_2lep = config.add_category(  # noqa: F841
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
        label=r"$MET \geq 20$",
    )
    config.add_category(
        name="lowmet",
        id=6,
        selection="catid_lowmet",
        label=r"$MET < 20$",
    )


def add_dih_mll_categories(config: od.Config) -> None:
    """
    Adds categories based on mll.
    NOTE: this should never be used in combination with the *add_abcd_categories* function
    """
    config.add_category(
        name="sr",
        id=1,
        selection="catid_mll_low_narrow",
        label=r"$20 \leq m_{\ell\ell} < 70$",
    )
    cr = config.add_category(
        name="cr",
        id=2,
        selection="catid_cr",
        label=r"$20 \leq m_{\ell\ell} \geq 70$",
    )
    cr.add_category(
        name="dycr",
        id=3,
        selection="catid_mll_z_wide",
        label=r"$70 \leq m_{\ell\ell} < 110$",
    )
    cr.add_category(
        name="ttcr",
        id=4,
        selection="catid_mll_very_high",
        label=r"$m_{\ell\ell} \geq 110$",
    )


def add_trih_mll_categories(config: od.Config) -> None:
    """
    Adds categories based on mll.
    NOTE: this should never be used in combination with the *add_abcd_categories* function
    """
    config.add_category(
        name="sr",
        id=5,
        selection="catid_hhh_sr",
        label=r"$12 \leq m_{\ell\ell} < 80, \geq 2 N_{jets}$",
    )
    cr = config.add_category(
        name="cr",
        id=6,
        selection="catid_hhh_cr",
        label=r"$12 \leq m_{\ell\ell} \geq 80$",
    )
    cr.add_category(
        name="dycr",
        id=7,
        selection="catid_hhh_dycr",
        label=r"$80 \leq m_{\ell\ell} < 100$",
    )
    cr.add_category(
        name="ttcr",
        id=8,
        selection="catid_hhh_ttcr",
        label=r"$m_{\ell\ell} \geq 100$",
    )


@call_once_on_config()
def add_mll_categories(config, add_mll_categories_func=None):
    if not add_mll_categories_func:
        add_mll_categories_func = config.x.add_mll_categories_func
    add_mll_categories_func(config)


@call_once_on_config()
def add_lepton_categories(config: od.Config) -> None:
    config.x.lepton_channels = {
        "sl": ("1e", "1mu"),
        "dl": ("2e", "2mu", "emu", "ge3lep"),
    }[config.x.lepton_tag]

    cat_1e = config.add_category(  # noqa: F841
        name="1e",
        id=10,
        selection="catid_1e",
        label="1 Electron",
    )

    cat_1mu = config.add_category(  # noqa: F841
        name="1mu",
        id=20,
        selection="catid_1mu",
        label="1 Muon",
    )
    # dl categories
    cat_2e = config.add_category(  # noqa: F841
        name="2e",
        id=30,
        selection="catid_2e",
        label="2 Electron",
    )

    cat_2mu = config.add_category(  # noqa: F841
        name="2mu",
        id=40,
        selection="catid_2mu",
        label="2 Muon",
    )

    cat_emu = config.add_category(  # noqa: F841
        name="emu",
        id=50,
        selection="catid_emu",
        label="1 Electron 1 Muon",
    )

    cat_emu = config.add_category(  # noqa: F841
        name="ge3lep",
        id=60,
        selection="catid_ge3lep",
        label=r"$N_{lep} \geq 3$",
    )


@call_once_on_config()
def add_njet_categories(config: od.Config) -> None:
    config.add_category(
        name="njet1",
        id=100001,
        selection="catid_njet1",
        label=r"$N_{jet} >= 1$",
    )
    config.add_category(
        name="njet3",
        id=100003,
        selection="catid_njet3",
        label=r"$N_{jet} >= 3$",
    )
    config.add_category(
        name="njet2",
        id=100003,
        selection="catid_njet2",
        label=r"$N_{jet} >= 2$",
    )


@call_once_on_config()
def add_jet_categories(config: od.Config) -> None:
    cat_resolved = config.add_category(  # noqa: F841
        name="resolved",
        id=100,
        selection="catid_resolved",
        label="resolved",
    )
    cat_boosted = config.add_category(  # noqa: F841
        name="boosted",
        id=200,
        selection="catid_boosted",
        label="boosted",
    )
    cat_boosted_eq1 = config.add_category(  # noqa: F841
        name="boosted_eq1",
        id=300,
        selection="catid_boosted_eq1",
        label="boosted",
    )
    cat_boosted_geq2 = config.add_category(  # noqa: F841
        name="boosted_geq2",
        id=400,
        selection="catid_boosted_geq2",
        label="boosted",
    )
    cat_boosted_low = config.add_category(  # noqa: F841
        name="boosted_low",
        id=500,
        selection="catid_boosted_low",
        label="boosted",
    )
    cat_boosted_loose = config.add_category(  # noqa: F841
        name="boosted_loose",
        id=600,
        selection="catid_boosted_loose",
        label="boosted",
    )
    cat_boosted_glo = config.add_category(  # noqa: F841
        name="boosted_glo",
        id=700,
        selection="catid_boosted_glo",
        label="boosted",
    )
    cat_resolved_glo = config.add_category(  # noqa: F841
        name="resolved_glo",
        id=800,
        selection="catid_resolved_glo",
        label="boosted",
    )


def add_dih_bjet_categories(config: od.Config) -> None:
    cat_1b = config.add_category(  # noqa: F841
        name="1b",
        id=10000,
        selection="catid_1b",
        label=r"$\leq 1 btag$",
    )
    cat_2b = config.add_category(  # noqa: F841
        name="2b",
        id=20000,
        selection="catid_2b",
        label=r"$2 btag$",
    )


def add_trih_bjet_categories(config: od.Config) -> None:
    cat_eq2b = config.add_category(  # noqa: F841
        name="2b",
        id=19000,
        selection="catid_eq2b",
        label=r"$2 btag$",
    )
    cat_2b_1l = config.add_category(  # noqa: F841
        name="2b_1l",
        id=39050,
        selection="catid_2b_1l",
        label=r"$2 btag$",
    )
    cat_1b_1tb = config.add_category(  # noqa: F841
        name="1b_1tb",
        id=29050,
        selection="catid_1b_1tb",
        label=r"$2 btag$",
    )
    cat_3b = config.add_category(  # noqa: F841
        name="3b",
        id=9000,
        selection="catid_eq3b",
        label=r"$3 btag$",
    )
    cat_4b = config.add_category(  # noqa: F841
        name="4b",
        id=15050,
        selection="catid_geq4b",
        label=r"$\geq 4 btag$",
    )


@call_once_on_config()
def add_bjet_categories(config, add_bjet_categories_func=None):
    if not add_bjet_categories_func:
        add_bjet_categories_func = config.x.add_bjet_categories_func

    add_bjet_categories_func(config)


@call_once_on_config()
def add_categories_selection(config: od.Config) -> None:
    """
    Adds categories to a *config*, that are typically produced in `SelectEvents`.
    """

    # inclusive category separate from all other categories (important for cross checks)
    config.add_category(
        name="incl",
        id=0,
        selection="catid_incl",
        label="Inclusive",
    )

    # adds categories based on the existence of gen particles
    # NOTE: commented out because we did not use it anyways
    # add_gen_categories(config)
    if config.x.lepton_tag == "sl":
        # adds categories for ABCD background estimation
        add_abcd_categories(config)
        config.x.main_categories = ["sr", "fake"]
    elif config.x.lepton_tag == "dl":
        # adds categories based on mll
        add_mll_categories(config)
        config.x.main_categories = ["sr", "dycr", "ttcr"]
        # adds categories based on bjets
        if config.x.signal_tag == "hh":
            # add_mll_categories(config, add_dih_mll_categories)
            config.x.bjet_categories = ["1b", "2b"]
        elif config.x.signal_tag == "hhh":
            # add_mll_categories(config, add_trih_mll_categories)
            config.x.bjet_categories = ["3b", "4b"]
            config.x.jet_categories = ["resolved_glo", "boosted_glo"]

    # adds categories based on number of leptons
    add_lepton_categories(config)


def name_fn(root_cats):
    cat_name = "__".join(cat.name for cat in root_cats.values())
    return cat_name


def kwargs_fn(root_cats):
    kwargs = {
        "id": sum([c.id for c in root_cats.values()]),
        "label": "\n".join([c.label for c in root_cats.values()]),
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
        logger.warning(
            "We should not call *add_categories_production* when also building ML categories."
            "This function will not add any categories.",
        )
        # when ML categories already exist, don't do anything
        return

    # jet categories conatin resolved an boosted, should be same for everything
    add_jet_categories(config)
    # bjet categories are defined in config, but also can be given directly in function
    add_bjet_categories(config)

    #
    # define all combinations of categories
    #

    category_blocks = OrderedDict({
        "main": [config.get_category(cat) for cat in config.x.main_categories],
        "lep": [config.get_category(lep_ch) for lep_ch in config.x.lepton_channels],
        # "jet": [config.get_category("resolved"), config.get_category("boosted")],
        "jet": [config.get_category(mode) for mode in config.x.jet_categories],
        "b": [config.get_category(bjet) for bjet in config.x.bjet_categories],
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

    dycr__nonmixed = config.add_category(
        name="dycr__nonmixed",
        id=2349237509,
        label="dycr (Nonmixed)",
    )
    dycr__nonmixed.add_category(config.get_category("dycr__2e"))
    dycr__nonmixed.add_category(config.get_category("dycr__2mu"))


@call_once_on_config()
def add_categories_ml(config, ml_model_inst):
    print("test.", ml_model_inst)
    if config.has_tag("add_categories_production_called"):
        raise Exception("We should not call *add_categories_production* when also building ML categories")

    #
    # prepare non-ml categories
    #

    add_jet_categories(config)
    add_bjet_categories(config)

    #
    # add parent ml model categories
    #

    # if not already done, get the ml_model instance
    if isinstance(ml_model_inst, str):
        ml_model_inst = MLModel.get_cls(ml_model_inst)(config)

    # add ml categories directly to the config
    # NOTE: this is a bit dangerous, because our ID depends on the MLModel, but
    #       we can reconfigure our MLModel after having created these categories
    # TODO: config is empty and therefore fails
    ml_categories = []
    for proc, node_config in ml_model_inst.train_nodes.items():
        print(proc, node_config["ml_id"])
        _id = (node_config["ml_id"] + 1) * 1000
        ml_categories.append(config.add_category(
            # NOTE: name and ID is unique as long as we don't use
            #       multiple ml_models simutaneously
            name=f"ml_{proc}",
            # NOTE: the +1 is necessary to avoid reusing ID of non-ml categories
            id=_id,
            selection=f"catid_ml_{proc}",
            aux={"ml_proc": proc},
        ))

    #
    # create combination of categories
    #

    # if config.x.signal_tag == "hhh":
    #     config.x.bjet_categories = ["3b", "4b"]
    #     config.x.jet_categories = ["resolved_glo", "boosted_glo"]

    # adds categories based on number of leptons
    # NOTE: building this many categories takes forever: has to be improved...
    category_blocks = OrderedDict({
        # NOTE: when building DNN categories, we do not need the control regions
        "main": [config.get_category("sr")],
        # "jet": [config.get_category("resolved"), config.get_category("boosted")],
        "jet": [config.get_category(mode) for mode in config.x.jet_categories],
        "b": [config.get_category(bjet) for bjet in config.x.bjet_categories],
        "dnn": ml_categories,
    })

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
