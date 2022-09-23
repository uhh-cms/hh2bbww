# coding: utf-8

"""
Producers that determine the generator-level particles of a HH->bbWW decay.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column


ak = maybe_import("awkward")


@producer
def gen_hbw_decay_products(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates column 'hhgen'
    """

    if self.dataset_inst.is_data or not self.dataset_inst.x("is_hbw", False):
        return events

    abs_id = abs(events.GenPart.pdgId)

    # for quick checks
    def all_or_raise(arr, msg):
        if not ak.all(arr):
            raise Exception(f"{msg} in {100 * ak.mean(~arr):.3f}% of cases")

    # only consider hard process genparticles
    gp = events.GenPart[events.GenPart.hasFlags("isHardProcess")]
    gp = gp[~ak.is_none(gp, axis=1)]
    gp["index"] = ak.local_index(gp, axis=1)
    abs_id = abs(gp.pdgId)

    # find hard Higgs bosons
    h = gp[abs_id == 25]
    nh = ak.num(h, axis=1)
    all_or_raise(nh == 2, "number of Higgs != 2")

    # bottoms from H decay
    b = gp[abs_id == 5]
    b = b[(abs(b.distinctParent.pdgId) == 25)]
    nb = ak.num(b, axis=1)
    all_or_raise(nb == 2, "number of bottom quarks from Higgs decay != 2")

    # Ws from H decay
    w = gp[abs_id == 24]
    w = w[(abs(w.distinctParent.pdgId) == 25)]
    nw = ak.num(w, axis=1)
    all_or_raise(nw == 2, "number of Ws != 2")

    # non-top quarks from W decays
    qs = gp[(abs_id >= 1) & (abs_id <= 5)]
    qs = qs[(abs(qs.distinctParent.pdgId) == 24)]
    nqs = ak.num(qs, axis=1)
    all_or_raise((nqs % 2) == 0, "number of quarks from W decays is not dividable by 2")

    # leptons from W decays
    ls = gp[(abs_id >= 11) & (abs_id <= 16)]
    ls = ls[(abs(ls.distinctParent.pdgId) == 24)]
    nls = ak.num(ls, axis=1)
    all_or_raise((nls % 2) == 0, "number of leptons from W decays is not dividable by 2")

    all_or_raise(nqs + nls == 2 * nw, "number of W decay products invalid")

    # check if decay product charges are valid
    sign = lambda part: (part.pdgId > 0) * 2 - 1
    all_or_raise(ak.sum(sign(b), axis=1) == 0, "two ss bottoms")
    all_or_raise(ak.sum(sign(w), axis=1) == 0, "two ss Ws")
    all_or_raise(ak.sum(sign(qs), axis=1) == 0, "sign-imbalance for quarks")
    all_or_raise(ak.sum(sign(ls), axis=1) == 0, "sign-imbalance for leptons")

    # TODO: sort decay products in a useful way
    # from IPython import embed; embed()

    hhgen = ak.zip({
        "hbb": h[:, 0],
        "hww": h[:, 1],
        "b1": b[:, 0],
        "b2": b[:, 1],
        "wlep": w[:, 0],
        "whad": w[:, 1],
        "l": ls[:, 0],
        "nu": ls[:, 1],
        "q1": qs[:, 0],
        "q2": qs[:, 1],
    })

    # dummy return
    events = set_ak_column(events, "hhgen", hhgen)

    return events


@gen_hbw_decay_products.init
def gen_hbw_decay_products_init(self: Producer) -> None:
    """
    Ammends the set of used and produced columns of :py:class:`gen_hbw_decay_products` in case
    a dataset including top decays is processed.
    """
    if getattr(self, "dataset_inst", None) and self.dataset_inst.x("is_hbw", False):
        self.uses |= {"nGenPart", "GenPart.*"}
        self.produces |= {"gen_hbw_decay"}
