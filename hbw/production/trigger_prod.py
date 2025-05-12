# coding: utf-8

"""
Column production methods related trigger studies
"""


from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from hbw.selection.nn_trigger import NN_trigger_inference

np = maybe_import("numpy")
ak = maybe_import("awkward")

def my_mask(events):
    ele_pt = events.Electron.pt
    ele_pt = ak.pad_none(ele_pt, 2, axis=1)
    ele_pt = ak.fill_none(ele_pt, -1)
    mask = (ele_pt[:, 0] > 31) | (ele_pt[:, 0] < 0)
    print(20*'#', sum(mask))
    return mask
# produce trigger columns for debugging
@producer(
    produces={"trig_ids"},
    version=2,
)
def trigger_prod(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces column filled for each event with the triggers triggering the event.
    This column can then be used to fill a Histogram where each bin corresponds to a certain trigger.
    """
    events = self[NN_trigger_inference](events, **kwargs)
    topo_mask = events.L1NNscore >= 0.8351974487304688 
    ele_mask = events.HLT.Ele30_WPTight_Gsf
    ele_realistic = (my_mask(events)) & (events.HLT.Ele30_WPTight_Gsf)
    # TODO: check if trigger were fired by unprescaled L1 seed
    trig_ids = ak.Array([[0.]] * len(events))

    trig_passed = ak.where(ele_mask, [[1.]], [[]])
    trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    trig_passed = ak.where(topo_mask, [[2.]], [[]])
    trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    trig_passed = ak.where(ele_realistic, [[3.]], [[]])
    trig_ids = ak.concatenate([trig_ids, trig_passed], axis=1)

    events = set_ak_column(events, "trig_ids", trig_ids)

    return events


# initialize the trigger producer, triggers can be set in the trigger config
@trigger_prod.init
def trigger_prod_init(self: Producer) -> None:
    self.uses.add("HLT.Ele30_WPTight_Gsf")
    self.uses.add(NN_trigger_inference)
    self.uses.add("Electron.pt")