# coding: utf-8

"""
ML models using the MLClassifierBase and Mixins
"""

from __future__ import annotations

# from columnflow.types import Union

import law

from columnflow.util import maybe_import

from hbw.ml.base import MLClassifierBase
from hbw.ml.mixins import DenseModelMixin, ClassicModelFitMixin
# from hbw.config.styling import color_palette


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


class DenseClassifierDCTR(DenseModelMixin, ClassicModelFitMixin, MLClassifierBase):
    _default__processes: tuple = (
        "data",
        "tt",
        "st",
        "dy_m10to50", "dy_m50toinf",
        "h", "vv", "ttv", "other",
    )
    train_nodes: dict = {
        "mc_bkg": {
            "ml_id": 0,
            "label": "MC Bkg",
            # "class_factor_mode": "equal",  # should not be needed
            "sub_processes": ["tt", "st", "dy_m10to50", "dy_m50toinf", "vv", "ttv", "h", "other"],
        },
        "data": {
            "ml_id": 1,
            "label": "Data",
        },
    }
    _default__class_factors: dict = {
        "mc_bkg": 1,
        "data": 1,
    }

    _default__sub_process_class_factors = {
        "data": 1,
        "tt": 1,
        "st": 1,
        "dy_m10to50": 1, "dy_m50toinf": 1,
        "h": 1, "vv": 1, "ttv": 1, "other": 1,
    }

    input_features = [
        # "mli_met_pt",
        "mli_ll_pt",
        "mli_n_jet",
    ]

    preparation_producer_name = "prepml"

    folds: int = 5
    negative_weights: str = "ignore"

    # overwriting DenseModelMixin parameters
    _default__activation: str = "relu"
    _default__layers: tuple = (64,)
    _default__batchnorm: bool = False
    _default__l2regularization: float = 0.0
    _default__dropout: float = 0.0
    _default__learningrate: float = 0.00050
    _default__weighting: str = "c2st"
    _default__optimizer: str = "sgd"

    # overwriting ModelFitMixin parameters
    _default__callbacks: set = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True
    _default__reduce_lr_factor: float = 0.2
    _default__reduce_lr_patience: int = 2
    _default__epochs: int = 50
    _default__batchsize: int = 2 ** 13
    # steps_per_epoch: Union[int, str] = "iter_smallest_process"

    # parameters to add into the `parameters` attribute to determine the 'parameters_repr' and to store in a yaml file
    bookkeep_params: set[str] = {
        # base params
        "data_loader", "input_features", "train_val_test_split",
        "processes", "sub_process_class_factors", "class_factors", "train_nodes",
        "negative_weights", "folds", "weighting",
        # DenseModelMixin
        "activation", "layers", "dropout", "learningrate", "batchnorm", "l2regularization", "optimizer",
        # ModelFitMixin
        "callbacks", "reduce_lr_factor", "reduce_lr_patience",
        "epochs", "batchsize",
    }

    # parameters that can be overwritten via command line
    settings_parameters: set[str] = {
        # base params
        "processes", "class_factors", "sub_process_class_factors",
        # DenseModelMixin
        "activation", "layers", "dropout", "learningrate", "batchnorm", "l2regularization",
        # ModelFitMixin
        "callbacks", "reduce_lr_factor", "reduce_lr_patience",
        "epochs", "batchsize",
    }

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def cast_ml_param_values(self):
        super().cast_ml_param_values()

    def setup(self) -> None:
        super().setup()
