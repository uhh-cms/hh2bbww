# coding: utf-8

"""
Set of producers to reconstruct categories at different states of the analysis
"""

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, InsertableDict
from columnflow.production.categories import category_ids

# from hbw.production.weights import event_weights
# from hbw.production.prepare_objects import prepare_objects
from hbw.config.categories import add_categories_production, add_categories_ml
from hbw.ml.dense_classifier import dense_test

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@producer(
    uses={category_ids},
    produces={category_ids},
)
def pre_ml_cats(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Reproduces category ids before ML Training
    """
    # category ids
    events = self[category_ids](events, **kwargs)

    return events


@pre_ml_cats.init
def pre_ml_cats_init(self: Producer) -> None:
    if self.config_inst.x("add_categories_production", True):
        # add categories but only on first call
        add_categories_production(self.config_inst)
        self.config_inst.x.add_categories_production = False


@producer(
    uses={category_ids},
    produces={category_ids},
)
def ml_cats(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Reproduces category ids after ML Training
    """
    # category ids
    events = self[category_ids](events, **kwargs)

    return events


@ml_cats.requires
def ml_cats_reqs(self: Producer, reqs: dict) -> None:
    if "ml" in reqs:
        return

    from columnflow.tasks.ml import MLEvaluation
    reqs["ml"] = MLEvaluation.req(self.task, ml_model=self.ml_model_name)


@ml_cats.setup
def ml_cats_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    reader_targets["mlcolumns"] = inputs["ml"]["mlcolumns"]


@ml_cats.init
def ml_cats_init(self: Producer) -> None:
    self.ml_model_name = "dense_test"

    if self.config_inst.x("add_categories_production", True):
        # add categories but only on first call
        add_categories_production(self.config_inst)
        self.config_inst.x.add_categories_production = False

    if self.config_inst.x("add_categories_ml", True):
        # add ml categories but only on first call
        add_categories_ml(self.config_inst, dense_test)
        self.config_inst.x.add_categories_ml = False


# TODO: derive different `ml_cats` producer for all the available ml_models
