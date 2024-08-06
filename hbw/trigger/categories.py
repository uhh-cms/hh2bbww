# coding: utf-8
"""
defines categories based on the selection used for the trigger studies
"""

from __future__ import annotations

import law

from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer
from columnflow.selection import SelectionResult

np = maybe_import("numpy")
ak = maybe_import("awkward")

# categorizer muon channel


@categorizer()
def catid_trigger_mu(
    self: Categorizer,
    events: ak.Array,
    results: SelectionResult | None = None,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    if results:
        return events, results.steps.SR_mu
    else:
        raise NotImplementedError(f"Category didn't receive a SelectionResult")

# categorizer electron channel


@categorizer()
def catid_trigger_ele(
    self: Categorizer,
    events: ak.Array,
    results: SelectionResult | None = None,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    if results:
        return events, results.steps.SR_ele
    else:
        raise NotImplementedError(f"Category didn't receive a SelectionResult")

# categorizer for orthogonal measurement (muon channel)


@categorizer()
def catid_trigger_orth_mu(
    self: Categorizer,
    events: ak.Array,
    results: SelectionResult | None = None,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    if results:
        return events, (results.steps.TrigEleMatch & results.steps.SR_mu & results.steps.ref_trigger_mu)
    else:
        raise NotImplementedError(f"Category didn't receive a SelectionResult")

# categorizer for orthogonal measurement (electron channel)


@categorizer()
def catid_trigger_orth_ele(
    self: Categorizer,
    events: ak.Array,
    results: SelectionResult | None = None,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    if results:
        return events, (results.steps.TrigMuMatch & results.steps.SR_ele & results.steps.ref_trigger_e)
    else:
        raise NotImplementedError(f"Category didn't receive a SelectionResult")
