# coding: utf-8

"""
Column production methods related trigger studies
"""


from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    produces={"trig_bits", "trig_bits_orth"},
    channel=["mu", "e"],
)
def trigger_prod(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces column where each bin corresponds to a certain trigger
    """
    
    trig_bits = ak.Array([["allEvents"]] * len(events))
    trig_bits_orth = ak.Array([["allEvents"]] * len(events))

    for channel in self.channel:
        ref_trig = self.config_inst.x.ref_trigger[channel]
        for trigger in self.config_inst.x.trigger[channel]:

            trig_passed = ak.where(events.HLT[trigger], [[trigger]], [[]])
            trig_bits = ak.concatenate([trig_bits, trig_passed], axis=1)

            trig_passed_orth = ak.where((events.HLT[ref_trig] & events.HLT[trigger]), [[trigger]], [[]])
            trig_bits_orth = ak.concatenate([trig_bits_orth, trig_passed_orth], axis=1)


    events = set_ak_column(events, "trig_bits", trig_bits)
    events = set_ak_column(events, "trig_bits_orth", trig_bits_orth)
    
    return events

@trigger_prod.init 
def trigger_prod_init(self: Producer) -> None:

    for channel in self.channel:
        for trigger in self.config_inst.x.trigger[channel]:
            self.uses.add(f"HLT.{trigger}")
        self.uses.add(f"HLT.{self.config_inst.x.ref_trigger[channel]}")

# producers for single channels
mu_trigger_prod = trigger_prod.derive("mu_trigger_prod", cls_dict={"channel": ["mu"], "produces": {"trig_bits", "trig_bits_orth"}})
ele_trigger_prod = trigger_prod.derive("ele_trigger_prod", cls_dict={"channel": ["e"]})