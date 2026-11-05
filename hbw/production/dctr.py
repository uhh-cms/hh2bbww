# coding: utf-8
"""
Producer to evaluate DCTR models and produce correction weights
"""


from __future__ import annotations


import law
from columnflow.types import Any
from columnflow.production import Producer, producer
from columnflow.util import maybe_import, DotDict
from columnflow.columnar_util import set_ak_column


np = maybe_import("numpy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)


@producer(
    ml_model_name=None,
    weight_name=None,
    ml_node=None,
    cap=None,
    eps=1e-6,
    version=0,
)
def dctr_reweighter(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to evaluate DCTR models and produce correction weights
    """
    if self.dataset_inst.is_data:
        return events

    def weight(probability, eps):
        p = np.asarray(probability, dtype=np.float64)
        if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid classifier probabilities")
        p = np.clip(p, eps, 1 - eps)
        return p / (1 - p)

    prob = events["mlscore"][self.ml_node]

    reweight = weight(prob, self.eps)

    if self.cap is not None:
        quantile = float(np.quantile(reweight, self.cap))
        reweight = np.minimum(reweight, quantile)

    events = set_ak_column(events, self.weight_name, reweight)

    return events


@dctr_reweighter.requires
def dctr_reweighter_requires(self: Producer, task: law.Task, reqs: dict[str, DotDict[str, Any]], **kwargs) -> None:
    if task.dataset_inst.is_data:
        return
    if "ml" in reqs:
        return

    from columnflow.tasks.ml import MLEvaluation
    reqs["ml"] = MLEvaluation.req(task, ml_model=self.ml_model_name)  # TODO: maybe move to config


@dctr_reweighter.setup
def dctr_reweighter_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
    **kwargs
) -> None:
    """
    Setup function for the DCTR producer, load the model scores
    """
    if self.dataset_inst.is_data:
        return
    reader_targets["mlcolumns"] = inputs["ml"]["mlcolumns"]


@dctr_reweighter.init
def dctr_reweighter_init(self: Producer, **kwargs) -> None:

    if None in (self.ml_model_name, self.weight_name, self.ml_node):
        raise ValueError("ml_model_name, weight_name and ml_node must be specified")

    if self.dataset_inst.is_data:
        return

    self.uses = {"mlscore.*"}
    self.produces = {self.weight_name}


# Derived DCTR reweighters #############################################################################################
dctr_24_dy_reweighter = dctr_reweighter.derive("dctr_24_dy_reweighter", cls_dict={
    "ml_model_name": "dctr_dy_24_v5",
    "weight_name": "dy_correction_weight",
    "ml_node": "dy_data",
    "cap": 0.995,
    "version": 4,
})
c2st_24incl_reweighter = dctr_reweighter.derive("c2st_24incl_reweighter", cls_dict={
    "ml_model_name": "c2st_24_incl_nocorr",
    "weight_name": "c2st_incl_correction",
    "ml_node": "data",
    "cap": 0.995,
    "version": 0,
})
c2st_24dyvr_reweighter = dctr_reweighter.derive("c2st_24dyvr_reweighter", cls_dict={
    "ml_model_name": "c2st_24_min_njlq32",
    "weight_name": "c2st_dyvr_correction",
    "ml_node": "data",
    "cap": 0.995,
    "version": 9,
})
c2st_24dyvr_reverse_reweighter = dctr_reweighter.derive("c2st_24dyvr_reverse_reweighter", cls_dict={
    "ml_model_name": "c2st_24_binary_reduced",
    "weight_name": "c2st_dyvr_correction",
    "ml_node": "mc_bkg",
    "cap": 0.995,
    "version": 0,
})
