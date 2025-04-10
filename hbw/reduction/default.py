# coding: utf-8

"""
Exemplary reduction methods that can run on-top of columnflow's default reduction.
"""

from columnflow.reduction import Reducer, reducer
from columnflow.reduction.default import cf_default
from columnflow.util import maybe_import

from hbw.util import IF_TOP, IF_VJETS
from columnflow.production.cms.top_pt_weight import gen_parton_top
from hbw.production.gen_v import gen_v_boson
from hbw.production.dy import gen_dilepton

ak = maybe_import("awkward")


@reducer(
    uses={
        cf_default,
        IF_TOP(gen_parton_top), IF_VJETS(gen_v_boson), IF_VJETS(gen_dilepton),
    },
    produces={
        cf_default,
        IF_TOP(gen_parton_top), IF_VJETS(gen_v_boson), IF_VJETS(gen_dilepton),
    },
)
def default(self: Reducer, events: ak.Array, selection: ak.Array, **kwargs) -> ak.Array:
    # run cf's default reduction which handles event selection and collection creation
    events = self[cf_default](events, selection, **kwargs)

    # compute and store additional columns after the default reduction
    # (so only on a subset of the events and objects which might be computationally lighter)

    # compute gen information that will later be needed for top pt reweighting
    if self.dataset_inst.has_tag("has_top"):
        events = self[gen_parton_top](events, **kwargs)

    # compute gen information that will later be needed for vector boson pt reweighting
    if self.dataset_inst.has_tag("is_v_jets"):
        events = self[gen_v_boson](events, **kwargs)
        events = self[gen_dilepton](events, **kwargs)

    return events
