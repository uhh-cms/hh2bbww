# coding: utf-8

"""
ML models using the MLClassifierBase and Mixins
"""

from __future__ import annotations

import law
import order as od

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production import Producer, producer
from columnflow.util import maybe_import, DotDict
from columnflow.columnar_util import set_ak_column

from hbw.ml.base import MLClassifierBase
from hbw.ml.mixins import DenseModelMixin, ModelFitMixin

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


class DenseClassifier(DenseModelMixin, MLClassifierBase):
    def __init__(
            self,
            *args,
            folds: int | None = None,
            **kwargs,
    ):

        super().__init__(*args, **kwargs)


dense_test = DenseClassifier.derive("dense_test", cls_dict={"folds": 5})
