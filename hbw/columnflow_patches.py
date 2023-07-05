# coding: utf-8

"""
Collection of patches of underlying columnflow tasks.
"""

import law
from columnflow.util import memoize


@memoize
def patch_mltraining():
    from columnflow.tasks.ml import MLTraining

    MLTraining.output_collection_cls = law.NestedSiblingFileCollection


@memoize
def patch_all():
    patch_mltraining()
