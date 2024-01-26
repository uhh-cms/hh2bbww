# coding: utf-8

"""
Selectors related to gen-level particles.
"""

import law

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


pdgId_map = {
    1: "down",
    2: "up",
    3: "strange",
    4: "charm",
    5: "bottom",
    6: "top",
    11: "electron",
    12: "e_neutrino",
    13: "muon",
    14: "mu_neutrino",
    15: "tau",
    16: "tau_neutrino",
    21: "gluon",
    22: "photon",
    23: "Z",
    24: "W",
    25: "Higgs",
}


@selector(
    uses={"GenPart.statusFlags"},
    mc_only=True,
)
def hard_gen_particles(
    self: Selector,
    events: ak.Array,
    results,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    gp_mask = events.GenPart.hasFlags("isHardProcess")

    results = results + SelectionResult(
        objects={"GenPart": {"HardGenPart": gp_mask}},
    )

    return events, results
