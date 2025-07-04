# coding: utf-8

"""
Column producers related to top quark pt reweighting.
"""

from __future__ import annotations

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")
np = maybe_import("numpy")


logger = law.logger.get_logger(__name__)


@producer(
    uses={"GenPartonTop.pt"},
    produces={"top_pt_theory_weight{,_up,_down}"},
    max_top_pt=None,
    # skip the producer unless the datasets has this specified tag (no skip check performed when none)
    require_dataset_tag="is_ttbar",
)
def top_pt_theory_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Compute SF to be used for top pt reweighting.

    Based on theory calculations. More details can be found here:
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting#TOP_PAG_corrections_based_on_the?rev=31

    The *GenPartonTop.pt* column can be produced with the :py:class:`gen_parton_top` Producer. The
    SF should *only be applied in ttbar MC* as an event weight and is computed based on the
    gen-level top quark transverse momenta.
    """
    # check the number of gen tops
    if ak.any((n_tops := ak.num(events.GenPartonTop, axis=1)) != 2):
        raise Exception(
            f"{self.cls_name} can only run on events with two generator top quarks, but found "
            f"counts of {','.join(map(str, sorted(set(n_tops))))}",
        )

    # clamp top pt
    top_pt = events.GenPartonTop.pt
    if self.max_top_pt and self.max_top_pt > 0:
        top_pt = ak.where(top_pt > self.max_top_pt, self.max_top_pt, top_pt)

    # params = [
    #     0.103,
    #     -0.0118,
    #     -0.000134,
    #     0.973,
    #     0.991,
    #     0.000075,
    # ]

    sf_run2 = (0.103 * np.exp(-0.0118 * top_pt) - 0.000134 * top_pt + 0.973)
    sf = (0.991 + 0.000075 * top_pt) * (sf_run2)

    # compute weight from SF product for top and anti-top
    weight = np.sqrt(np.prod(sf, axis=1))
    weight = ak.fill_none(weight, 1.0)

    # declare down variation as 1.0 and up variation as symmetric to the down variation
    weight_down = ak.ones_like(weight)
    weight_up = 2 * (weight - 1.0) + 1.0

    # write out weights
    events = set_ak_column(events, "top_pt_theory_weight", weight)
    events = set_ak_column(events, "top_pt_theory_weight_up", weight_up)
    events = set_ak_column(events, "top_pt_theory_weight_down", weight_down)

    return events


@top_pt_theory_weight.skip
def top_pt_theory_weight_skip(self: Producer, **kwargs) -> bool:
    """
    Skip if running on anything except ttbar MC simulation, evaluated via the :py:attr:`require_dataset_tag` attribute.
    """
    if self.require_dataset_tag is None:
        return self.dataset_inst.is_data

    return self.dataset_inst.is_data or not self.dataset_inst.has_tag("is_ttbar")
