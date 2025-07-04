# coding: utf-8

"""
Set of producers to reconstruct categories at different states of the analysis
"""

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.categories import category_ids

from hbw.config.categories import add_categories_production, add_categories_ml
from hbw.util import get_subclasses_deep

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@producer(
    # uses in init, produces should not be empty
    produces={"category_ids"},
    version=2,
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
    # add categories to config inst
    add_categories_production(self.config_inst)

    self.uses.add(category_ids)
    self.produces.add(category_ids)


@producer(
    # uses in init, produces should not be empty
    produces={"category_ids", "mlscore.max_score"},
    ml_model_name=None,
    version=2,
)
def cats_ml(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Reproduces category ids after ML Training. Calling this producer also
    automatically adds `MLEvaluation` to the requirements.
    """
    max_score = ak.fill_none(ak.max([events.mlscore[f] for f in events.mlscore.fields], axis=0), 0)
    events = set_ak_column(events, "mlscore.max_score", max_score, value_type=np.float32)
    # category ids
    events = self[category_ids](events, **kwargs)

    return events


@cats_ml.requires
def cats_ml_reqs(self: Producer, task: law.Task, reqs: dict) -> None:
    if "ml" in reqs:
        return

    from columnflow.tasks.ml import MLTraining, MLEvaluation
    if task.pilot:
        # skip MLEvaluation in pilot, but ensure that MLTraining has already been run
        reqs["mlmodel"] = MLTraining.req(task, ml_model=self.ml_model_name)
    else:
        reqs["ml"] = MLEvaluation.req(
            task,
            ml_model=self.ml_model_name,
        )


@cats_ml.setup
def cats_ml_setup(
    self: Producer, task: law.Task, reqs: dict, inputs: dict, reader_targets: law.util.InsertableDict,
) -> None:
    # self.uses |= self[category_ids].uses
    reader_targets["mlcolumns"] = inputs["ml"]["mlcolumns"]


@cats_ml.init
def cats_ml_init(self: Producer) -> None:
    if not self.ml_model_name:
        raise ValueError(f"invalid ml_model_name {self.ml_model_name} for Producer {self.cls_name}")

    # NOTE: if necessary, we could initialize the MLModel ourselves, e.g. via:
    # MLModelMixinBase.get_ml_model_inst(self.ml_model_name, self.analysis_inst, requested_configs=[self.config_inst])

    if not self.config_inst.has_variable("mlscore.max_score"):
        self.config_inst.add_variable(
            name="mlscore.max_score",
            expression="mlscore.max_score",
            binning=(1000, 0., 1.),
            x_title="DNN max output score",
            aux={
                "rebin": 25,
            },
        )

    # add categories to config inst
    add_categories_ml(self.config_inst, self.ml_model_name)

    self.uses.add(category_ids)
    self.produces.add(category_ids)


# get all the derived MLModels and instantiate a corresponding producer for each one
from hbw.ml.base import MLClassifierBase
ml_model_names = get_subclasses_deep(MLClassifierBase)
logger.info(f"deriving {len(ml_model_names)} ML categorizer...")

for ml_model_name in ml_model_names:
    cats_ml.derive(f"cats_ml_{ml_model_name}", cls_dict={"ml_model_name": ml_model_name})
