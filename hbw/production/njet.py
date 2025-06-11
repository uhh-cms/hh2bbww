# coding: utf-8

"""
Event weight producer.
"""

import law

from columnflow.util import maybe_import

from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column
from hbw.production.prepare_objects import prepare_objects

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


# Apply njet correction from hbw.GetNJetCorrections binned in
@producer(
    uses={
        "{Jet,ForwardJet}.{pt,eta,phi,mass}",
        prepare_objects,
    },
    # both used columns and dependent shifts are defined in init below
    weight_columns=None,
    # only run on mc
    mc_only=True,
    # optional categorizer to obtain baseline event mask
    categorizer_cls=None,
    # variable="n_jet",
    corrected_process=None,
    from_file=False,
)
def njet_weight(
    self: Producer,
    events: ak.Array,
    task: law.Task,
    **kwargs,
) -> ak.Array:
    """
    Creates trigger scale factor weights using correctionlib. Requires external file.
    """
    events = self[prepare_objects](events, **kwargs)

    var_map = {
        "n_jet": ak.num(events.Jet.pt, axis=1),
        "n_forwardjet": ak.num(events.ForwardJet.pt, axis=1),
    }
    corrector = self.correction_set[f"{self.corrected_process}_njet_corrections"]

    inputs = [var_map[inp.name] for inp in corrector.inputs]
    njet_weight = corrector.evaluate(*inputs)

    events = set_ak_column(events, "njet_weight", njet_weight)
    return events


@njet_weight.setup
def njet_weight_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
) -> None:
    # create the corrector
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    if self.from_file:
        # used when the correction is stored as a JSON dict
        self.correction_set = correctionlib.CorrectionSet.from_file(
            inputs["njet_corrections"][f"{self.corrected_process}_njet_corrections"].fn,
        )
    else:
        # used when correction is stored as a JSON string
        self.correction_set = correctionlib.CorrectionSet.from_string(
            inputs["njet_corrections"][f"{self.corrected_process}_njet_corrections"].load(formatter="json"),
        )


@njet_weight.init
def njet_weight_init(self: Producer) -> None:

    if not self.dataset_inst.has_tag("is_dy"):
        return
    self.corrected_process = "dy"
    self.produces.add("njet_weight")


@njet_weight.requires
def njet_weight_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    from hbw.tasks.corrections import GetNJetCorrections
    processes = [
        "vv", "w_lnu", "st",
        "dy_m4to10", "dy_m10to50", "dy_m50toinf",
        "tt", "ttv", "h", "data",
    ]
    reqs["njet_corrections"] = GetNJetCorrections.req(
        task,
        processes=processes,
        hist_producer="with_dy_weight",
        categories=["dycr_nonmixed"],
        corrected_processes=self.corrected_process,
    )


@njet_weight.skip
def njet_weight_skip(self: Producer) -> bool:
    """
    Skip if running on anything except ttbar MC simulation.
    """
    # never skip when there is no dataset
    if not getattr(self, "dataset_inst", None):
        return False

    return self.dataset_inst.is_data or not (
        self.dataset_inst.has_tag("is_dy")
    )
