# coding: utf-8

"""
Column producers related to top quark pt reweighting.
"""

from __future__ import annotations

from dataclasses import dataclass

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")
np = maybe_import("numpy")


logger = law.logger.get_logger(__name__)


@dataclass
class TopPtWeightConfig:
    params: dict[str, float]
    pt_max: float = 500.0

    @classmethod
    def new(cls, obj: TopPtWeightConfig | dict[str, float]) -> TopPtWeightConfig:
        # backward compatibility only
        if isinstance(obj, cls):
            return obj
        return cls(params=obj)


def get_top_pt_theory_weight_config(self: Producer) -> TopPtWeightConfig:
    params = self.config_inst.x.top_pt_theory_weight

    return TopPtWeightConfig(params=params, pt_max=-1.0)


@producer(
    uses={"GenPartonTop.pt"},
    produces={"top_pt_theory_weight{,_up,_down}"},
    get_top_pt_theory_weight_config=get_top_pt_theory_weight_config,
    # skip the producer unless the datasets has this specified tag (no skip check performed when none)
    require_dataset_tag="is_ttbar",
)
def top_pt_theory_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Compute SF to be used for top pt reweighting.

    Based on:
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting#TOP_PAG_corrections_based_on_the?rev=31

    The *GenPartonTop.pt* column can be produced with the :py:class:`gen_parton_top` Producer. The
    SF should *only be applied in ttbar MC* as an event weight and is computed based on the
    gen-level top quark transverse momenta.

    The top pt reweighting parameters should be given as an auxiliary entry in the config:

    .. code-block:: python

        cfg.x.top_pt_theory_weight = {
            "a": 0.0615,
            "a_up": 0.0615 * 1.5,
            "a_down": 0.0615 * 0.5,
            "b": -0.0005,
            "b_up": -0.0005 * 1.5,
            "b_down": -0.0005 * 0.5,
        }

    *get_top_pt_config* can be adapted in a subclass in case it is stored differently in the config.

    :param events: awkward array containing events to process
    """
    # check the number of gen tops
    if ak.any((n_tops := ak.num(events.GenPartonTop, axis=1)) != 2):
        raise Exception(
            f"{self.cls_name} can only run on events with two generator top quarks, but found "
            f"counts of {','.join(map(str, sorted(set(n_tops))))}",
        )

    # clamp top pt
    top_pt = events.GenPartonTop.pt
    if self.cfg.pt_max >= 0.0:
        top_pt = ak.where(top_pt > self.cfg.pt_max, self.cfg.pt_max, top_pt)

    for variation in ("", "_up", "_down"):
        a = self.cfg.params[f"a{variation}"]
        b = self.cfg.params[f"b{variation}"]
        c = self.cfg.params[f"c{variation}"]
        d = self.cfg.params[f"d{variation}"]

        # evaluate SF function
        sf = a * np.exp(b * top_pt) + c * top_pt + d

        # compute weight from SF product for top and anti-top
        weight = np.sqrt(np.prod(sf, axis=1))

        # write out weights
        events = set_ak_column(events, f"top_pt_theory_weight{variation}", ak.fill_none(weight, 1.0))

    return events


@top_pt_theory_weight.init
def top_pt_theory_weight_init(self: Producer) -> None:
    # store the top pt weight config
    self.cfg = self.get_top_pt_theory_weight_config()


@top_pt_theory_weight.skip
def top_pt_theory_weight_skip(self: Producer, **kwargs) -> bool:
    """
    Skip if running on anything except ttbar MC simulation, evaluated via the :py:attr:`require_dataset_tag` attribute.
    """
    if self.require_dataset_tag is None:
        return self.dataset_inst.is_data

    return self.dataset_inst.is_data or not self.dataset_inst.has_tag("is_ttbar")
