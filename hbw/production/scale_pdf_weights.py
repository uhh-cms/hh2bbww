# coding: utf-8

"""
Column production methods related to generic event weights.
"""

from columnflow.production import Producer, producer
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

    # stop here for data
    if self.dataset_inst.is_data:
        return events

    N_pdfweights = ak.num(events.LHEPdfWeight, axis=1)
    if ak.all(N_pdfweights == 103):
        # LHEPdfWeight value 102: alpha down; value 103: alpha down
        events = set_ak_column(events, "alpha_weight", ak.ones_like(events.event))
        events = set_ak_column(events, "alpha_weight_down", events.LHEPdfWeight[:, 101])
        events = set_ak_column(events, "alpha_weight_up", events.LHEPdfWeight[:, 102])
    elif ak.all(N_pdfweights == 101):
        # dummy alpha weight (TODO)
        events = set_ak_column(events, "alpha_weight", ak.ones_like(events.event))
        events = set_ak_column(events, "alpha_weight_down", ak.ones_like(events.event))
        events = set_ak_column(events, "alpha_weight_up", ak.ones_like(events.event))
    else:
        raise Exception(f"Number of LHEPdfWeights ({N_pdfweights}) is not "
                        f"as expected (103 or 101) in dataset {self.dataset_inst.name}")

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

    # stop here for data
    if self.dataset_inst.is_data:
        return events

    N_scaleweights = ak.num(events.LHEScaleWeight, axis=1)
    # For now, make an exception for st_schannel_had dataset; should be fixed with NanoAODv10
    if ak.any(N_scaleweights != 9) and self.dataset_inst.name != "st_schannel_had_amcatnlo":
        N_scaleweights = N_scaleweights[N_scaleweights != 9]
        raise Exception(f"Number of LHEScaleWeights ({N_scaleweights}) is not "
                        f"as expected (9) in dataset {self.dataset_inst.name}")

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

    # stop here for data
    if self.dataset_inst.is_data:
        return events

    N_scaleweights = ak.num(events.LHEScaleWeight, axis=1)
    # For now, make an exception for st_schannel_had dataset; should be fixed with NanoAODv10
    if ak.any(N_scaleweights != 9) and self.dataset_inst.name != "st_schannel_had_amcatnlo":
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
