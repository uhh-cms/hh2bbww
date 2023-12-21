# coding: utf-8
"""
Producers for Neutrino reconstruction.
"""

import functools

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

# from cmsdb.constants import m_w

from hbw.util import four_vec
from hbw.production.prepare_objects import prepare_objects


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses=prepare_objects,
    produces=four_vec("Neutrino"),
)
def neutrino_reconstruction(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to reconstruct a neutrino orignating from a leptonically decaying W boson.
    Assumes that Neutrino pt can be reconstructed via MET and that the W boson has been
    produced on-shell.
    """
    # add behavior and define new collections (e.g. Lepton)
    events = self[prepare_objects](events, **kwargs)

    # TODO: might be outdated, should be defined in cmsdb
    w_mass = 80.379

    # get input variables (assuming that there is only one lepton)
    E_l = events.Lepton.E[:, 0]
    pt_l = events.Lepton.pt[:, 0]
    pz_l = events.Lepton.pz[:, 0]
    pt_nu = events.MET.pt

    delta_phi = events.Lepton[:, 0].delta_phi(events.MET)

    mu = w_mass**2 / 2 + pt_nu * pt_l * np.cos(delta_phi)

    # pz_nu = A +- sqrt(B-C)
    A = mu * pz_l / pt_l**2

    B = mu**2 * pz_l**2 / pt_l**4
    C = (E_l**2 * pt_nu**2 - mu**2) / pt_l**2

    pz_nu_1 = ak.where(
        B - C >= 0,
        # solution is real
        A + np.sqrt(B - C),
        # complex solution -> take only the real part
        A,
    )
    # pz_nu_2 = ak.where(
    #     B - C >= 0,
    #     # solution is real
    #     A - np.sqrt(B - C),
    #     # complex solution -> take only the real part
    #     A,
    # )

    p_nu_1 = np.sqrt(pt_nu**2 + pz_nu_1**2)
    eta_nu_1 = np.log((p_nu_1 + pz_nu_1) / (p_nu_1 - pz_nu_1)) / 2

    # store Neutrino 4 vector components
    events["Neutrino"] = events.MET
    events = set_ak_column_f32(events, "Neutrino.eta", eta_nu_1)

    # testing: Neutrino pz should be the same as pz_nu_1 within rounding errors

    # testing: reconstructing W mass should always (if B-C>0) give the input W mass (80.4 GeV)
    # but currently, this is at 114 most of the times, so something is off
    W_on_shell = events.Neutrino + events.Lepton[:, 0]
    print(W_on_shell.mass)

    return events
