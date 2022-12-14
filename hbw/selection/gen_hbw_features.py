# coding: utf-8

"""
Selectors to set ak columns for gen particles of hh2bbww
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column  # , Route, EMPTY_FLOAT
from columnflow.selection import SelectionResult
from columnflow.production import Producer, producer

ak = maybe_import("awkward")


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


@producer(
    uses={
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB",
        "gen_hbw_decay",
    },
    produces=set(
        f"cutflow.{gp}_{var}"
        for gp in ["h1", "h2", "b1", "b2", "wlep", "whad", "l", "nu", "q1", "q2", "sec1", "sec2"]
        for var in ["pt", "eta", "phi", "mass"]
    ),
)
def gen_hbw_decay_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for var in ["pt", "eta", "phi", "mass"]:
        for gp in ["h1", "h2", "b1", "b2", "wlep", "whad", "l", "nu", "q1", "q2", "sec1", "sec2"]:
            events = set_ak_column(events, f"cutflow.{gp}_{var}", events.gen_hbw_decay[gp][var])

    return events


@producer(
    uses={
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.genJetIdx",
        "GenJet.pt", "GenJet.eta", "GenJet.phi", "GenJet.mass", "GenJet.partonFlavour", "GenJet.hadronFlavour",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "gen_hbw_decay",
    },
)
def gen_hbw_matching(
        self: Producer, events: ak.Array,
        results: SelectionResult = None, verbose: bool = True,
        dR_req: float = 0.4, ptdiff_req: float = 10.,
        **kwargs,
) -> ak.Array:
    """
    Function that matches HH->bbWW decay product gen particles to Reco-level jets and leptons.
    """

    gen_matches = {}

    # jet matching
    # NOTE: might be nice to remove jets that already have been matched
    for gp_tag in ("b1", "b2", "q1", "q2", "sec1", "sec2"):
        gp = events.gen_hbw_decay[gp_tag]

        dr = events.Jet.delta_r(gp)
        pt_diff = (events.Jet.pt - gp.pt) / gp.pt

        jet_match_mask = (dr < dR_req) & (abs(pt_diff) < ptdiff_req)
        jet_matches = events.Jet[jet_match_mask]

        if verbose:
            print(gp_tag, "multiple matches:", ak.sum(ak.num(jet_matches) > 1))
            print(gp_tag, "no matches:", ak.sum(ak.num(jet_matches) == 0))

        # if multiple matches found: choose match with smallest dr
        jet_match = jet_matches[ak.argsort(jet_matches.delta_r(gp))]
        gen_matches[gp_tag] = jet_match

    # lepton matching for combined electron and muon
    lepton_fields = set(events.Muon.fields).intersection(events.Electron.fields)

    lepton = ak.concatenate([
        ak.zip({f: events.Muon[f] for f in lepton_fields}),
        ak.zip({f: events.Electron[f] for f in lepton_fields}),
    ], axis=-1)
    lepton = ak.with_name(lepton, "PtEtaPhiMLorentzVector")

    gp_lep = events.gen_hbw_decay.l
    dr = lepton.delta_r(gp_lep)
    pt_diff = (lepton.pt - gp_lep.pt) / gp_lep.pt
    lep_match_mask = (dR_req < 0.4) & (abs(pt_diff) < ptdiff_req)
    lep_matches = lepton[lep_match_mask]

    if verbose:
        print("l multiple matches:", ak.sum(ak.num(lep_matches) > 1))
        print("l no matches:", ak.sum(ak.num(lep_matches) == 0))

    lep_match = lep_matches[ak.argsort(lep_matches.delta_r(gp_lep))]
    gen_matches["l"] = lep_match

    # write matches into events and return them
    events = set_ak_column(events, "gen_match", ak.zip(gen_matches, depth_limit=1))
    return events
