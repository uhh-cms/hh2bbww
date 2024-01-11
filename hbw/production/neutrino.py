# coding: utf-8
"""
Producers for Neutrino reconstruction.
"""

import functools

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT

# from cmsdb.constants import m_w

from hbw.util import four_vec
from hbw.production.prepare_objects import prepare_objects
from hbw.config.variables import add_neutrino_variables, add_top_reco_variables


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses=prepare_objects,
    produces=four_vec(["Neutrino", "Neutrino1", "Neutrino2"]),
)
def neutrino_reconstruction(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to reconstruct a neutrino orignating from a leptonically decaying W boson.
    Assumes that Neutrino pt can be reconstructed via MET and that the W boson has been
    produced on-shell.

    TODO: reference
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

    delta_phi = abs(events.Lepton[:, 0].delta_phi(events.MET))
    mu = w_mass**2 / 2 + pt_nu * pt_l * np.cos(delta_phi)

    # Neutrino pz will be calculated as: pz_nu = A +- sqrt(B-C)
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

    pz_nu_2 = ak.where(
        B - C >= 0,
        # solution is real
        A - np.sqrt(B - C),
        # complex solution -> take only the real part
        A,
    )

    pz_nu_solutions = [pz_nu_1, pz_nu_2]

    for i, pz_nu in enumerate(pz_nu_solutions, start=1):
        # convert to float64 to prevent rounding errors
        pt_nu = ak.values_astype(pt_nu, np.float64)
        pz_nu = ak.values_astype(pz_nu, np.float64)

        # calculate Neutrino eta to define the Neutrino 4-vector
        p_nu_1 = np.sqrt(pt_nu**2 + pz_nu**2)
        eta_nu_1 = np.log((p_nu_1 + pz_nu) / (p_nu_1 - pz_nu)) / 2
        # store Neutrino 4 vector components
        events[f"Neutrino{i}"] = events.MET
        events = set_ak_column_f32(events, f"Neutrino{i}.eta", eta_nu_1)

        # sanity check: Neutrino pz should be the same as pz_nu within rounding errors
        sanity_check_1 = ak.sum(abs(events[f"Neutrino{i}"].pz - pz_nu) > abs(events[f"Neutrino{i}"].pz) / 100)
        if sanity_check_1:
            logger.warning(
                "Number of events with Neutrino.pz that differs from pz_nu by more than 1 percent: "
                f"{sanity_check_1} (solution {i})",
            )

        # sanity check: reconstructing W mass should always (if B-C>0) give the input W mass (80.4 GeV)
        W_on_shell = events[f"Neutrino{i}"] + events.Lepton[:, 0]
        sanity_check_2 = ak.sum(abs(ak.where(B - C >= 0, W_on_shell.mass, w_mass) - w_mass) > 1)
        if sanity_check_2:
            logger.warning(
                "Number of events with W mass from reconstructed Neutrino (real solutions only) that "
                f"differs by more than 1 GeV from the input W mass: {sanity_check_2} (solution {i})",
            )

    # sanity check: for complex solutions, only the real part is considered -> both solutions should be identical
    sanity_check_3 = ak.sum(ak.where(B - C <= 0, events.Neutrino1.eta - events.Neutrino2.eta, 0))
    if sanity_check_3:
        raise Exception(
            "When finding complex neutrino solutions, both reconstructed Neutrinos should be identical",
        )

    # combine both Neutrino solutions by taking the solution with smaller absolute eta
    events = set_ak_column_f32(
        events, "Neutrino",
        ak.where(abs(events.Neutrino1.eta) > abs(events.Neutrino2.eta), events.Neutrino2, events.Neutrino1),
    )
    return events


@neutrino_reconstruction.init
def neutrino_reconstruction_init(self: Producer) -> None:
    # add variable instances to config
    add_neutrino_variables(self.config_inst)


@producer(
    uses={neutrino_reconstruction, prepare_objects} | four_vec("Bjet"),
    produces={neutrino_reconstruction} | four_vec({"tlep_hyp1", "tlep_hyp2"}),
)
def top_reconstruction(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to reconstruct ttbar top quark masses using the neutrino_reconstruction Producer
    """
    # add behavior and define new collections (e.g. Lepton)
    events = self[prepare_objects](events, **kwargs)

    # run the neutrino reconstruction
    events = self[neutrino_reconstruction](events, **kwargs)

    # object padding (there are some boosted events that only contain one Jet)
    events = set_ak_column(events, "Bjet", ak.pad_none(events.Bjet, 2))

    dr_b1_lep = events.Bjet[:, 0].delta_r(events.Lepton[:, 0])
    dr_b2_lep = events.Bjet[:, 1].delta_r(events.Lepton[:, 0])

    blep = ak.where(dr_b1_lep < dr_b2_lep, events.Bjet[:, 0], events.Bjet[:, 1])
    bhad = ak.where(dr_b1_lep > dr_b2_lep, events.Bjet[:, 0], events.Bjet[:, 1])

    tlep_hyp1 = blep + events.Lepton[:, 0] + events.Neutrino
    tlep_hyp2 = bhad + events.Lepton[:, 0] + events.Neutrino

    # events = set_ak_column_f32(events, "tlep_hyp1", tlep_hyp1)
    # events = set_ak_column_f32(events, "tlep_hyp2", tlep_hyp2)

    # tlep vectors store columns (x, y, z, t), so set all 4-vec components by hand
    for var in ("pt", "eta", "phi", "mass"):
        events = set_ak_column_f32(events, f"tlep_hyp1.{var}", getattr(tlep_hyp1, var))
        events = set_ak_column_f32(events, f"tlep_hyp2.{var}", getattr(tlep_hyp2, var))

    # fill nan/none values of all produced columns
    for route in self.produced_columns:
        # replace nan, none, and inf values with EMPTY_FLOAT
        col = route.apply(events)
        col = ak.fill_none(ak.nan_to_none(route.apply(events)), EMPTY_FLOAT)
        col = ak.where(np.isinf(col), EMPTY_FLOAT, col)

        events = set_ak_column(events, route.string_column, col)

    return events


@neutrino_reconstruction.init
def top_reconstruction_init(self: Producer) -> None:
    # add variable instances to config
    add_top_reco_variables(self.config_inst)
