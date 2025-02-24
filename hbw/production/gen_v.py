# coding: utf-8

"""
Producers for L1 prefiring weights.
"""

from __future__ import annotations

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, DotDict
from columnflow.columnar_util import set_ak_column


np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={
        "GenPart.genPartIdxMother", "GenPart.pdgId", "GenPart.statusFlags",
        "GenPart.pt", "GenPart.eta", "GenPart.phi", "GenPart.mass",
    },
    # requested GenVBoson columns, passed to the *uses* and *produces*
    produced_v_columns={"pt"},
    mc_only=True,
)
def gen_v_boson(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produce gen-level Z or W bosons from `GenParticle` collection.
    """
    # try to find parton-level electroweak boson
    abs_id = abs(events.GenPart.pdgId)
    v_boson = events.GenPart[((abs_id == 23) | (abs_id == 24))]
    v_boson = v_boson[v_boson.hasFlags("isLastCopy")]
    v_boson = v_boson[~ak.is_none(v_boson, axis=1)]
    if ak.any(ak.num(v_boson) > 1):
        raise Exception("found more than one parton-level Z/W boson")
    v_boson = ak.firsts(v_boson)

    # if not found, reconstruct pair of leptons from hard process
    leptons = events.GenPart[((abs_id >= 11) & (abs_id <= 16))]
    leptons = leptons[leptons.hasFlags("isHardProcess")]
    if ak.any(ak.num(leptons) != 2):
        raise Exception("number of hard process leptons != 2")
    dilepton_pair = (leptons[:, 0] + leptons[:, 1])

    # assign PDG ID based on lepton flavors: 23 (Z), +/- 24 (W+/W-)
    dilepton_pdg_id = ak.zeros_like(dilepton_pair.pt)
    dilepton_pdg_id = ak.where(
        abs(leptons.pdgId[:, 0]) == abs(leptons.pdgId[:, 1]),
        23,
        dilepton_pdg_id,
    )
    dilepton_pdg_id = ak.where(
        abs(leptons.pdgId[:, 0]) != abs(leptons.pdgId[:, 1]),
        24 * ak.where(
            leptons.pdgId[:, 0] % 2,
            np.sign(leptons.pdgId[:, 1]),
            np.sign(leptons.pdgId[:, 0]),
        ),
        dilepton_pdg_id,
    )
    dilepton_pair["pdgId"] = dilepton_pdg_id

    # check if the V boson type is consistent
    unique_pdg_id = (set(abs(v_boson.pdgId)) | set(abs(dilepton_pair.pdgId))) - {None}
    if len(unique_pdg_id) != 1:
        raise Exception(f"found multiple boson types: {unique_pdg_id}")

    # save the column
    for field in self.produced_v_columns:
        value = ak.where(
            ak.is_none(v_boson),
            getattr(dilepton_pair, field),
            getattr(v_boson, field),
        )
        events = set_ak_column(
            events,
            f"GenVBoson.{field}",
            ak.fill_none(value, 0.0),
        )

    return events


@gen_v_boson.init
def gen_v_boson_init(self: Producer) -> bool:
    for col in self.produced_v_columns:
        self.uses.add(f"GenPart.{col}")
        self.produces.add(f"GenVBoson.{col}")


@gen_v_boson.skip
def gen_v_boson_skip(self: Producer) -> bool:
    """
    Custom skip function that checks whether the dataset is a MC simulation containing
    V+jets events.
    """
    # never skip when there is not dataset
    if not getattr(self, "dataset_inst", None):
        return False

    return self.dataset_inst.is_data or not (
        self.dataset_inst.has_tag("is_v_jets")
    )


@producer(
    uses={
        "GenVBoson.pt",
    },
    produces={
        "vjets_weight", "vjets_weight_up", "vjets_weight_down",
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_vjets_reweighting_file=(lambda self, external_files: external_files.vjets_reweighting),
    # function to determine the correction file
    get_vjets_reweighting_config=(lambda self: self.config_inst.x.vjets_reweighting),
)
def vjets_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer for V+jets K factor weights. Requires an external file in the config as under ``vjets_reweighting``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "vjets_reweighting": "data/json/vjets_reweighting.json.gz",
        })

    *get_vjets_reweighting_file* can be adapted in a subclass in case it is stored differently in the external
    files.

    The name of the corrections as a function of the generator-level *W* and *Z* transverse momentum, as well as the
    associated uncertainty should be given as an auxiliary entry in the config:

    .. code-block:: python

        cfg.x.vjets_reweighting = DotDict.wrap({
            "w": {
                "value": "wjets_kfactor_value",
                "error": "wjets_kfactor_error",
            },
            "z": {
                "value": "zjets_kfactor_value",
                "error": "zjets_kfactor_error",
            },
        })
    """

    # fail if not run in ttbar simulation
    if not self.dataset_inst.has_tag("is_v_jets"):
        raise Exception(f"vjets_weight should only run for W+Jets and Z+Jets datasets, got {self.dataset_inst}")

    def get_kfactor(obj_name, key, obj):
        """Obtain K factor."""
        # keep track of missing inputs
        input_is_none = ak.is_none(obj.pt)
        pt = ak.fill_none(obj.pt, 0.0)
        # get values from correctionlib evaluator
        val = self.vjets_reweighting_evaluators[obj_name][key](pt)
        # mask values where inputs were missing
        val = ak.where(~input_is_none, val, 1.0)
        return val

    # get config key by boson type
    if self.dataset_inst.has_tag("is_z_jets"):
        boson = "z"
    elif self.dataset_inst.has_tag("is_w_jets"):
        boson = "w"
    else:
        assert False

    # compute the prefiring probablities and related statistical uncertainties
    kfactor = DotDict.wrap({})
    for key in ("value", "error"):
        kfactor[key] = get_kfactor(boson, key, events.GenVBoson)

    weights = {
        # NOTE: 1-kfactor for "ew" correction
        "nominal": 1 - kfactor.value,
        "up": 1 - kfactor.value + kfactor.error,
        "down": 1 - kfactor.value - kfactor.error,
    }

    # save the weights
    events = set_ak_column(events, "vjets_weight", weights["nominal"], value_type=np.float32)
    events = set_ak_column(events, "vjets_weight_up", weights["up"], value_type=np.float32)
    events = set_ak_column(events, "vjets_weight_down", weights["down"], value_type=np.float32)

    return events


@vjets_weight.skip
def vjets_weight_skip(self: Producer) -> bool:
    """
    Skip if running on anything except ttbar MC simulation.
    """
    # never skip when there is no dataset
    if not getattr(self, "dataset_inst", None):
        return False

    return self.dataset_inst.is_data or not (
        self.dataset_inst.has_tag("is_v_jets")
    )


@vjets_weight.requires
def vjets_weight_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@vjets_weight.setup
def vjets_weight_setup(
    self: Producer, task: law.Task, reqs: dict, inputs: dict, reader_targets: law.util.InsertableDict,
) -> None:
    bundle = reqs["external_files"]

    # create the L1 prefiring weight evaluator
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    correction_set = correctionlib.CorrectionSet.from_string(
        self.get_vjets_reweighting_file(bundle.files).load(formatter="gzip").decode("utf-8"),
    )
    corrections = self.get_vjets_reweighting_config()

    self.vjets_reweighting_evaluators = {
        obj_name: {
            key: correction_set[correction_name]
            for key, correction_name in corrections_obj.items()
        }
        for obj_name, corrections_obj in corrections.items()
    }
