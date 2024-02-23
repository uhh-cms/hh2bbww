# coding: utf-8

"""
Column producers related to gen-level top quark.
"""
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, has_ak_column

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={"GenPart.pdgId", "GenPart.statusFlags"},
    # requested GenPartonTop columns, passed to the *uses* and *produces*
    produced_top_columns={"pt"},
    mc_only=True,
)
def gen_parton_top(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produce parton-level top quarks (before showering and detector simulation).
    """
    # find parton-level top quarks
    abs_id = abs(events.GenPart.pdgId)
    t = events.GenPart[abs_id == 6]
    t = t[t.hasFlags("isLastCopy")]
    t = t[~ak.is_none(t, axis=1)]

    # save the column
    events = set_ak_column(events, "GenPartonTop", t)

    return events


@gen_parton_top.init
def gen_parton_top_init(self: Producer) -> bool:
    for col in self.produced_top_columns:
        self.uses.add(f"GenPart.{col}")
        self.produces.add(f"GenPartonTop.{col}")


@gen_parton_top.skip
def gen_parton_top_skip(self: Producer) -> bool:
    """
    Custom skip function that checks whether the dataset is a MC simulation containing top
    quarks in the first place.
    """
    # never skip when there is not dataset
    if not getattr(self, "dataset_inst", None):
        return False

    return self.dataset_inst.is_data or not self.dataset_inst.has_tag("has_top")


@producer(
    uses={
        "GenPartonTop.pt",
    },
    produces={
        "top_pt_weight", "top_pt_weight_up", "top_pt_weight_down",
    },
    get_top_pt_config=(lambda self: self.config_inst.x.top_pt_reweighting_params),
)
def top_pt_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Compute SF to be used for top pt reweighting.

    The SF should *only be applied in ttbar MC* as an event weight computed
    based on the gen-level top quark transverse momenta.
    """

    # get SF function parameters from config
    params = self.get_top_pt_config()

    # obtain gen-level top quark information if not already done
    if not has_ak_column(events, "GenPartonTop.pt"):
        events = self[gen_parton_top](events, **kwargs)

    # clamp top pT < 500 GeV and evaluate SF function
    pt_clamped = ak.where(events.GenPartonTop.pt > 500.0, 500.0, events.GenPartonTop.pt)
    sf = ak.pad_none(np.exp(params["a"] + params["b"] * pt_clamped), 2)

    # compute weight from SF product for top and anti-top
    weight = np.sqrt(sf[:, 0] * sf[:, 1])

    # write out weights
    events = set_ak_column(events, "top_pt_weight", ak.fill_none(weight, 1.0))
    events = set_ak_column(events, "top_pt_weight_up", ak.fill_none(weight * 1.5, 1.0))
    events = set_ak_column(events, "top_pt_weight_down", ak.fill_none(weight * 0.5, 1.0))

    return events


@top_pt_weight.skip
def top_pt_weight_skip(self: Producer) -> bool:
    """
    Skip if running on anything except ttbar MC simulation.
    """
    # never skip when there is no dataset
    if not getattr(self, "dataset_inst", None):
        return False

    return self.dataset_inst.is_data or not self.dataset_inst.has_tag("is_ttbar")
