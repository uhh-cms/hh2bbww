# coding: utf-8

"""
Producers for FatJet scale factor weights.
"""

from __future__ import annotations

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, load_correction_set
from columnflow.columnar_util import (  # noqa: F401
    set_ak_column, has_ak_column, flat_np_view, layout_ak_array, DotDict, optional_column,
)
from columnflow.types import Any

np = maybe_import("numpy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)


@producer(
    uses={
        "FatJet.{pt,eta,phi,mass,particleNetWithMass_HbbvsQCD,hadronFlavour}",
        # optional_column("FatJet.hadronFlavour"),
    },
    # uses={"FatJet.*"},
    produces={"hbb_sf_weight"},
    # only run on mc
    mc_only=True,
    # configurable weight name
    weight_name="hbb_sf_weight",
    # function to determine the correction file
    get_hbb_sf_file=(lambda self, external_files: external_files.hbb_sf_corr),
    # function to determine the btag sf config
)
def hbb_sf_weights(
    self: Producer,
    events: ak.Array,
    task: law.Task,
    **kwargs,
) -> ak.Array:
    """
    Produce FatJet btag scale factor weights.
    Scale factors are applied per FatJet depending on their hadron flavour (bb or cc).
    The scale factors are only applied for jets passing the HbbvsQCD cut (0.92).
    Additionally, up and down variations are computed, combining the nominal
    scale factor variations with the tau21 variations in quadrature.
    Finally, additional variations with inflated and flat 20% uncertainties are created.
    1. inflated: the combined uncertainty is inflated by a factor 2
    2. flat: if the combined uncertainty is < 20%, a flat uncertainty of 20% is used instead.
    The resulting weights are stored as event-level weights, multiplying the per-FatJet scale factors.
    """

    cpn_tag = task.config_inst.x.cpn_tag.replace("post", "_post").replace("pre", "_pre")

    bb_corrector = self.hbb_sf_corrector[f"HHbbww_{cpn_tag}_SF_bb"]
    cc_corrector = self.hbb_sf_corrector[f"HHbbww_{cpn_tag}_SF_cc"]

    variable_map = {
        "pt": events.FatJet.pt,
    }

    inputs = [variable_map[inp.name] for inp in bb_corrector.inputs[:-1]]

    # define masks for applying the scale factors
    apply_cc_sf = (
        (events.FatJet.hadronFlavour == 4) &
        (events.FatJet.particleNetWithMass_HbbvsQCD >= 0.92)
    )

    apply_bb_sf = (
        (events.FatJet.hadronFlavour == 5) &
        (events.FatJet.particleNetWithMass_HbbvsQCD >= 0.92)
    )
    apply_lf_sf = (
        (events.FatJet.hadronFlavour == 0) &
        (events.FatJet.particleNetWithMass_HbbvsQCD >= 0.92)
    )

    # store number of tagged jets
    num_cc_hbbjets = ak.sum(apply_cc_sf, axis=1)
    events = set_ak_column(
        events,
        f"{self.weight_name}_num_cc_hbbjets",
        num_cc_hbbjets,
        value_type=np.int32,
    )
    num_bb_hbbjets = ak.sum(apply_bb_sf, axis=1)
    events = set_ak_column(
        events,
        f"{self.weight_name}_num_bb_hbbjets",
        num_bb_hbbjets,
        value_type=np.int32,
    )
    num_lf_hbbjets = ak.sum(apply_lf_sf, axis=1)
    events = set_ak_column(
        events,
        f"{self.weight_name}_num_lf_hbbjets",
        num_lf_hbbjets,
        value_type=np.int32,
    )

    def get_sf_per_fatjet(variation):
        """
        Helper function to get the scale factor per FatJet for a given variation.
        1. evaluate both bb and cc correctors
        2. assign corrector based on hadron flavour
        3. apply only for jets passing the HbbvsQCD cut
        4. return scale factors per FatJet
        """
        # first, evaluate both bb and cc correctors for all FatJets
        sf_bb = bb_corrector.evaluate(*inputs, variation)
        sf_cc = cc_corrector.evaluate(*inputs, variation)

        # assign cc or bb sf based on hadron flavour
        sf = ak.where(
            apply_cc_sf,
            sf_cc,
            1.0,
        )
        sf = ak.where(
            apply_bb_sf,
            sf_bb,
            sf,
        )
        return sf

    sf_nominal = get_sf_per_fatjet("nominal")

    events = set_ak_column(
        events,
        self.weight_name,
        ak.prod(sf_nominal, axis=1),
        value_type=np.float32,
    )

    for variation, tau_variation in (
        ("up", "tau21Up"),
        ("down", "tau21Down"),
    ):
        sf = get_sf_per_fatjet(variation)
        sf_tau21 = get_sf_per_fatjet(tau_variation)

        sign = 1.0 if variation == "up" else -1.0

        if ak.any(
            (sign * (sf_nominal - sf) > 0) | (sign * (sf_nominal - sf_tau21) > 0),
        ):
            raise ValueError(
                f"Hbb SF variation '{variation}' or '{tau_variation}' produces a shift in the wrong direction, "
                "hinting at an invalid scale factor variations (we assume that up variations are always larger than "
                "nominal and down variations are always smaller).",
            )

        # combine the uncertainties in quadrature
        combined_sf = sf_nominal + sign * np.sqrt((sf_nominal - sf) ** 2 + (sf_nominal - sf_tau21) ** 2)
        events = set_ak_column(
            events,
            f"{self.weight_name}_{variation}",
            ak.prod(combined_sf, axis=1),
            value_type=np.float32,
        )

        # # inflate the combined uncertainty by a factor 2
        # combined_sf_inflated = sf_nominal + sign * np.sqrt((sf_nominal - sf) ** 2 + (sf_nominal - sf_tau21) ** 2) * 2
        # events = set_ak_column(
        #     events,
        #     f"{self.weight_name}_inflated_{variation}",
        #     ak.prod(combined_sf_inflated, axis=1),
        #     value_type=np.float32,
        # )

    # create additional variations with inflated and flat 20% uncertainties (per event)
    sf_nominal = events[self.weight_name]
    for variation in ("up", "down"):

        sf_variation = events[f"{self.weight_name}_{variation}"]
        sign = 1.0 if variation == "up" else -1.0

        if ak.any(sign * (sf_nominal - sf_variation) > 0):
            raise ValueError(
                f"Hbb SF variation '{variation}' produces a shift in the wrong direction, "
                "hinting at an invalid scale factor variations (we assume that up variations are always larger than "
                "nominal and down variations are always smaller).",
            )

        # TODO: update to per-object inflating
        # inflate the uncertainty by a factor 2
        events = set_ak_column(
            events,
            f"{self.weight_name}_inflated_{variation}",
            sf_nominal + (sf_variation - sf_nominal) * 2,
            value_type=np.float32,
        )

        # flat 20% uncertainty if the combined uncertainty is < 20%
        events = set_ak_column(
            events,
            f"{self.weight_name}_flat_{variation}",
            ak.where(
                (sf_nominal != 1.0) & (abs(sf_variation - sf_nominal) < 0.20),
                sf_nominal * 1.20 ** sign,
                sf_variation,
            ),
            value_type=np.float32,
        )
    return events


@hbb_sf_weights.post_init
def hbb_sf_weights_post_init(self: Producer, task: law.Task, **kwargs) -> None:
    self.produces.add(self.weight_name)
    for column in (
        "up",
        "down",
        "inflated_up",
        "inflated_down",
        "flat_up",
        "flat_down",
        "num_cc_hbbjets",
        "num_bb_hbbjets",
        "num_lf_hbbjets",
    ):
        self.produces.add(f"{self.weight_name}_{column}")


@hbb_sf_weights.requires
def hbb_sf_weights_requires(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    **kwargs,
) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@hbb_sf_weights.setup
def hbb_sf_weights_setup(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    inputs: dict[str, Any],
    reader_targets: law.util.InsertableDict,
    **kwargs,
) -> None:
    # load the btag sf corrector
    hbb_sf_file = self.get_hbb_sf_file(reqs["external_files"].files)
    self.hbb_sf_corrector = load_correction_set(hbb_sf_file)
