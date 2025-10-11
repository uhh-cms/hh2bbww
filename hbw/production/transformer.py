# coding: utf-8
"""
Producer to evaluate external transformer model.
"""


from __future__ import annotations

import law
from columnflow.types import Any
from columnflow.production import Producer, producer
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.columnar_util import set_ak_column
from hbw.production.prepare_objects import prepare_objects


np = maybe_import("numpy")
ak = maybe_import("awkward")


from hbw.util import timeit_multiple


@producer(
    uses={
        prepare_objects,
        "{Electron,Muon,Jet}.{pt,eta,phi,mass}",
        "PuppiMET.{pt,phi}",
        "Jet.btagPNetB",
        "{Electron,Muon}.{charge,pdgId}",
    },
    produces={"transformerScore{,_even,_odd}"},
    sandbox=dev_sandbox("bash::$HBW_BASE/sandboxes/venv_onnx.sh"),
    only_combined=False,
    zero_pad_value=0,
)
@timeit_multiple
def transformer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to evaluate external transformer model.
    """
    # prepare inputs for model evaluation
    events = self[prepare_objects](events, **kwargs)

    # pad jets to length 2
    events = set_ak_column(events, "Jet", ak.pad_none(events.Jet, 2))

    # dummy base mask
    base_mask = events.event > 0
    masked_events = events[base_mask]

    lepton_1 = masked_events.Lepton[:, 0]
    lepton_2 = masked_events.Lepton[:, 1]
    jet_1 = masked_events.Jet[:, 0]
    jet_2 = masked_events.Jet[:, 1]
    met = masked_events.PuppiMET

    def reshape(x, fields):
        arr = np.asarray(ak.fill_none(
            ak.values_astype(ak.zip([getattr(x, f) for f in fields]), np.float32),
            self.zero_pad_value,
        ))
        arr = arr.astype(
            [(name, np.float32) for name in arr.dtype.names], copy=False,
        ).view(np.float32).reshape((-1, len(arr.dtype)))
        return arr

    inputs = {
        "lepton_1": reshape(lepton_1, ("px", "py", "pz", "E", "pdgId", "charge")),
        "lepton_2": reshape(lepton_2, ("px", "py", "pz", "E", "pdgId", "charge")),
        "jet_1": reshape(jet_1, ("px", "py", "pz", "E", "btagPNetB")),
        "jet_2": reshape(jet_2, ("px", "py", "pz", "E", "btagPNetB")),
        "met": reshape(met, ("px", "py", "E")),
    }

    if self.only_combined:
        # prepare array for output score
        unmasked = np.asarray(ak.values_astype(ak.zeros_like(base_mask), np.float32)) - 1
        mask_inputs = lambda mask: {k: v[mask[base_mask]] for k, v in inputs.items()}

        for mask, session in (
            (events.event % 2 == 1, self.session_even),  # evaluate even model with odd events
            (events.event % 2 == 0, self.session_odd),   # evaluate odd model with even events
        ):
            outp = session.run(None, mask_inputs(mask))[0][:, 1]
            unmasked[mask & base_mask] = outp
        events = set_ak_column(events, "transformerScore", unmasked)
        return events
    else:
        # evaluate models for odd and even events
        for label, session in (
            ("odd", self.session_odd),
            ("even", self.session_even),
        ):
            unmasked = np.asarray(ak.values_astype(ak.zeros_like(base_mask), np.float32)) - 1
            outp = session.run(None, inputs)[0][:, 0]
            unmasked[base_mask] = outp
            events = set_ak_column(events, f"transformerScore_{label}", unmasked)

        # even model was trained on even events --> evaluate on odd events
        events = set_ak_column(events, "transformerScore", ak.where(
            events.event % 2 == 0,
            events.transformerScore_odd,
            events.transformerScore_even,
        ))
        return events


@transformer.requires
def transformer_requires(
    self: Producer,
    task: law.Task,
    reqs: dict[str, DotDict[str, Any]],
    **kwargs,
) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(task)


@transformer.setup
def transformer_setup(self: Producer, reqs, **kwargs) -> None:
    """
    Initialize the transformer model.
    """
    # load the model
    import onnxruntime as ort

    model_odd = reqs["external_files"].files.transformer_odd.fn
    self.session_odd = ort.InferenceSession(model_odd)

    model_even = reqs["external_files"].files.transformer_even.fn
    self.session_even = ort.InferenceSession(model_even)

    # Get input details
    for inp in self.session_odd.get_inputs():
        print(f"Input name: {inp.name}, shape: {inp.shape}, type: {inp.type}")


@transformer.init
def transformer_init(self: Producer, **kwargs) -> None:
    if not self.config_inst.has_variable("tnn_score"):
        for postfix in ("", "_even", "_odd"):
            self.config_inst.add_variable(
                name=f"tnn_score{postfix}",
                expression=f"transformerScore{postfix}",
                null_value=-1,
                binning=(100, 0., 1.),
                x_title=f"TNN score {postfix.strip('_')}",
            )


@transformer.teardown
def transformer_teardown(self: Producer, **kwargs) -> None:
    """
    Stops the model session.
    """
    delattr(self, "session_even")
    delattr(self, "session_odd")
