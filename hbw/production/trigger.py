# coding: utf-8

"""
Trigger related event weights.
"""

from __future__ import annotations

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, InsertableDict
from columnflow.columnar_util import set_ak_column, flat_np_view, layout_ak_array

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={
        "Trigger.pt", "Trigger.eta",
    },
    # produces in the init
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_trigger_file=(lambda self, external_files: external_files.trigger_sf),
    # function to determine the trigger weight config
    # get_trigger_config=(lambda self: self.config_inst.x.trigger_sf_names),
    weight_name="trigger_weight",
)
def trigger_weights(
    self: Producer,
    events: ak.Array,
    trigger_mask: ak.Array | type(Ellipsis) = Ellipsis,
    **kwargs,
) -> ak.Array:
    """
    Creates trigger weights using the correctionlib. Requires an external file in the config under
    ``trigger_sf``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "trigger_sf": "/afs/cern.ch/work/m/mrieger/public/mirrors/jsonpog-integration-9ea86c4c/POG/MUO/2017_UL/trigger_z.json.gz",  # noqa
        })

    *get_trigger_file* can be adapted in a subclass in case it is stored differently in the external
    files.

    The name of the correction set and the year string for the weight evaluation should be given as
    an auxiliary entry in the config:

    .. code-block:: python

        cfg.x.trigger_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", "2017_UL")

    *get_trigger_config* can be adapted in a subclass in case it is stored differently in the config.

    Optionally, a *trigger_mask* can be supplied to compute the scale factor weight based only on a
    subset of triggers.
    """
    # flat absolute eta and pt views
    abs_eta = flat_np_view(abs(events.Trigger.eta[trigger_mask]), axis=1)
    pt = flat_np_view(events.Trigger.pt[trigger_mask], axis=1)

    variable_map = {
        "year": self.year,
        "abseta": abs_eta,
        "eta": abs_eta,
        "pt": pt,
    }

    # loop over systematics
    for syst, postfix in [
        ("sf", ""),
        ("systup", "_up"),
        ("systdown", "_down"),
    ]:
        # get the inputs for this type of variation
        variable_map_syst = {
            **variable_map,
            "scale_factors": "nominal" if syst == "sf" else syst,  # syst key in 2022
            "ValType": syst,  # syst key in 2017
        }
        inputs = [variable_map_syst[inp.name] for inp in self.trigger_sf_corrector.inputs]
        sf_flat = self.trigger_sf_corrector(*inputs)

        # add the correct layout to it
        sf = layout_ak_array(sf_flat, events.Trigger.pt[trigger_mask])

        # create the product over all triggers in one event
        weight = ak.prod(sf, axis=1, mask_identity=False)

        # store it
        events = set_ak_column(events, f"{self.weight_name}{postfix}", weight, value_type=np.float32)

    return events


@trigger_weights.requires
def trigger_weights_requires(self: Producer, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@trigger_weights.setup
def trigger_weights_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    bundle = reqs["external_files"]

    # create the corrector
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    correction_set = correctionlib.CorrectionSet.from_string(
        self.get_trigger_file(bundle.files),
    )
    corrector_name, self.year = self.get_trigger_config()
    self.trigger_sf_corrector = correction_set[corrector_name]

    # check versions
    if self.supported_versions and self.trigger_sf_corrector.version not in self.supported_versions:
        raise Exception(f"unsuppprted trigger sf corrector version {self.trigger_sf_corrector.version}")


@trigger_weights.init
def trigger_weights_init(self: Producer, **kwargs) -> None:
    weight_name = self.weight_name
    self.produces |= {weight_name, f"{weight_name}_up", f"{weight_name}_down"}
