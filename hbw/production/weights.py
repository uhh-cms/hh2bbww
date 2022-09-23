# coding: utf-8

"""
Column production methods related to generic event weights.
"""

from columnflow.production import Producer, producer
from columnflow.production.pileup import pu_weights
from columnflow.production.normalization import normalization_weights
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")


@producer(
    uses={"LHEPdfWeight"},
    produces={
        "alpha_weight", "alpha_weight_up", "alpha_weight_down",
        "pdf_weight", "pdf_weight_up", "pdf_weight_down",
    },
)
def pdf_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    determines the pdf up and down shifts
    Documentation: https://arxiv.org/pdf/1510.03865.pdf
    """

    N_pdfweights = ak.num(events.LHEPdfWeight, axis=1)
    if ak.any(N_pdfweights != 103):
        raise Exception(f"Number of LHEPdfWeights ({N_pdfweights}) is not " 
                        f"as expected (103) in dataset {self.dataset_inst.name}")

    # LHEPdfWeight value 102: alpha down; value 103: alpha down
    events = set_ak_column(events, "alpha_weight", ak.ones_like(events.event))
    events = set_ak_column(events, "alpha_weight_down", events.LHEPdfWeight[:, 101])
    events = set_ak_column(events, "alpha_weight_up", events.LHEPdfWeight[:, 102])

    # first 101 LHEPdfWeight values: pdf variations
    # NOTE: check if nominal weight is included --> change 83 to 84?
    pdfweights = events.LHEPdfWeight[:, :101]
    pdfweights = ak.sort(pdfweights)

    # PDF uncertainty by calculating the variance (underestimates uncertainty?)
    # var = ak.var(events.LHEPdfWeight, axis=1)
    # PDF uncertainty as 68% CL
    var = (pdfweights[:, 83] - pdfweights[:, 15]) / 2

    mean = ak.mean(events.LHEPdfWeight, axis=1)

    # NOTE: use mean value as nominal pdf weight? or remove the necessity of adding this nominal weight?
    events = set_ak_column(events, "pdf_weight", ak.ones_like(events.event))
    events = set_ak_column(events, "pdf_weight_down", mean + var)
    events = set_ak_column(events, "pdf_weight_up", mean - var)

    return events


@producer(
    uses={"LHEScaleWeight"},
    produces={
        "mur_weight", "mur_weight_up", "mur_weight_down",
        "muf_weight", "muf_weight_up", "muf_weight_down",
    },
)
def murmuf_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    reads out mur and muf uncertainties independently; documentation of the LHEScaleWeight columns:
    https://cms-nanoaod-integration.web.cern.ch/integration/master/mc94X_doc.html
    TODO: better documentation
    """
    N_scaleweights = ak.num(events.LHEScaleWeight, axis=1)
    if ak.any(N_scaleweights != 9):
        N_scaleweights = N_scaleweights[N_scaleweights != 9]
        raise Exception(f"Number of LHEScaleWeights ({N_scaleweights}) is not as expected (9) in dataset {self.dataset_inst.name}")

    # NOTE: nominal mur/muf weights should be always 1, so could maybe be removed?
    # NOTE: it might also be smarter to not event save them as new columns but just
    #       use the existing LHEScaleWeight columns via aliases
    events = set_ak_column(events, "mur_weight", events.LHEScaleWeight[:, 4])
    events = set_ak_column(events, "mur_weight_up", events.LHEScaleWeight[:, 7])
    events = set_ak_column(events, "mur_weight_down", events.LHEScaleWeight[:, 1])
    events = set_ak_column(events, "muf_weight", events.LHEScaleWeight[:, 4])
    events = set_ak_column(events, "muf_weight_up", events.LHEScaleWeight[:, 5])
    events = set_ak_column(events, "muf_weight_down", events.LHEScaleWeight[:, 3])

    return events


@producer(
    uses={"LHEScaleWeight"},
    produces={"scale_weight", "scale_weight_up", "scale_weight_down"},
)
def scale_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    determines the scale up and down variations; documentation of the LHEScaleWeight columns:
    https://cms-nanoaod-integration.web.cern.ch/integration/master/mc94X_doc.html
    TODO: better documentation
    """

    N_scaleweights = ak.num(events.LHEScaleWeight, axis=1)
    if ak.any(ak.num(events.LHEScaleWeight, axis=1) != 9):
        N_scaleweights = N_scaleweights[N_scaleweights != 9]
        raise Exception(f"Number of LHEScaleWeights ({N_scaleweights}) is not "
                        f"as expected (9) in dataset {self.dataset_inst.name}")

    # LHEScaleWeights[:, 4] is the nominal weight
    events = set_ak_column(events, "scale_weight", events.LHEScaleWeight[:, 4])

    # for the up/down variations, take the max/min value of all possible combinations
    # except mur=2, muf=0.5 (index 2) and mur=0.5, muf=2 (index 6) into account
    idx_mask = (ak.local_index(events.LHEScaleWeight) != 2) & (ak.local_index(events.LHEScaleWeight) != 6)
    considered_scale_weights = events.LHEScaleWeight[idx_mask]
    events = set_ak_column(events, "scale_weight_down", ak.min(considered_scale_weights, axis=1))
    events = set_ak_column(events, "scale_weight_up", ak.max(considered_scale_weights, axis=1))

    return events


@producer(
    uses={"Jet.pt"},
    produces={"top_pt_weight", "top_pt_weight_up", "top_pt_weight_down"},
)
def top_pt_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Adds a dummy top pt weight and its variations to *events* in case the internal
    py:attr:`dataset_inst` represents a ttbar dataset.
    """
    # skip when not a ttbar dataset
    if not self.dataset_inst.x("is_ttbar", False):
        return events

    # save dummy, asymmetric top pt weights
    events = set_ak_column(events, "top_pt_weight", ak.ones_like(events.event) * 1.0)
    events = set_ak_column(events, "top_pt_weight_up", ak.ones_like(events.event) * 1.1)
    events = set_ak_column(events, "top_pt_weight_down", ak.ones_like(events.event) * 1.0)

    return events


@producer(
    uses={normalization_weights, pu_weights, scale_weights, murmuf_weights, pdf_weights, top_pt_weights},
    produces={normalization_weights, pu_weights, scale_weights, murmuf_weights, pdf_weights, top_pt_weights},
    shifts={"minbias_xs_up", "minbias_xs_down", "scale_up", "scale_down", "pdf_up", "pdf_down"},
)
def event_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Opinionated wrapper of several event weight producers. It declares dependence all shifts that
    might possibly change any of the weights.
    """
    # compute normalization weights
    events = self[normalization_weights](events, **kwargs)

    # compute pu weights
    events = self[pu_weights](events, **kwargs)

    # compute scale weights (TODO cases with != 9 weights)
    # events = self[scale_weights](events, **kwargs)

    # read out mur and weights (TODO cases with != 9 weights)
    # events = self[murmuf_weights](events, **kwargs)

    # compute pdf weights (TODO: cases with != 103 weights)
    # events = self[pdf_weights](events, **kwargs)

    # compute top pt weights
    events = self[top_pt_weights](events, **kwargs)

    return events


@event_weights.init
def event_weights_init(self: Producer) -> None:
    """
    Performs an update of the :py:obj:`event_weights` producer based on, when existing, the internal
    py:attr:`dataset_inst` attribute.
    """
    # add top pt weight shifts for ttbar, or when the dataset_inst is not even set, meaning that
    # the owning task is a ConfigTask or higher
    if not getattr(self, "dataset_inst", None) or self.dataset_inst.x("is_ttbar", False):
        self.shifts |= {"top_pt_up", "top_pt_down"}
