# coding: utf-8

"""
Trigger selection methods.
"""
import law

from functools import partial
from collections import defaultdict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, optional_column as optional

from hbw.util import ak_any

np = maybe_import("numpy")
ak = maybe_import("awkward")


# helper functions
set_ak_bool = partial(set_ak_column, value_type=np.bool_)
set_ak_int32 = partial(set_ak_column, value_type=np.int32)

# logger
logger = law.logger.get_logger(__name__)


@selector(
    uses={
        "run",
        # nano columns
        "TrigObj.id", "TrigObj.pt", "TrigObj.eta", "TrigObj.phi", "TrigObj.filterBits",
    },
    produces={
        # new columns
        "trigger_ids",
        "trigger_data.any_fired", "trigger_data.HLT*.{all_legs_match,fired,fired_and_all_legs_match}",
    },
)
def trigger_selection(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    HLT trigger path selection.

    NOTE: to use this selector, the *triggers* auxiliary must be set in the config.
    """
    trigger_ids = ak.singletons(ak.zeros_like(events.run, dtype=np.int64))

    trigger_data = ak.Array({
        "any_fired": ak.zeros_like(events.run, dtype=np.bool_),
    })

    # index of TrigObj's to repeatedly convert masks to indices
    index = ak.local_index(events.TrigObj)

    for trigger in self.config_inst.x.triggers:
        # # skip the trigger if it does not apply to the dataset
        # if not trigger.applies_to_dataset(self.dataset_inst):
        #     continue

        # get bare decisions
        fired = events.HLT[trigger.hlt_field] == 1
        if trigger.run_range:
            fired = fired & (
                ((trigger.run_range[0] is None) | (trigger.run_range[0] <= events.run)) &
                ((trigger.run_range[1] is None) | (trigger.run_range[1] >= events.run))
            )
        trigger_data = set_ak_bool(trigger_data, f"{trigger.name}.fired", fired)
        trigger_data = set_ak_bool(trigger_data, "any_fired", trigger_data.any_fired | fired)

        # get trigger objects for fired events per leg
        leg_masks = []
        all_legs_match = True
        leg_match_or = False
        for i, leg in enumerate(trigger.legs):
            # start with a True mask
            leg_mask = abs(events.TrigObj.id) >= 0
            # pdg id selection
            if leg.pdg_id is not None:
                leg_mask = leg_mask & (abs(events.TrigObj.id) == leg.pdg_id)
            # pt cut
            if leg.min_pt is not None:
                leg_mask = leg_mask & (events.TrigObj.pt >= leg.min_pt)
            # trigger bits match
            if leg.trigger_bits is not None:
                # OR across bits themselves, AND between all decision in the list
                for bits in leg.trigger_bits:
                    # NOTE: changed
                    # leg_mask = leg_mask & ((events.TrigObj.filterBits & bits) > 0)
                    leg_mask = leg_mask & ((events.TrigObj.filterBits & bits) == bits)
            leg_masks.append(index[leg_mask])

            # at least one object must match this leg
            # NOTE: it could in theory happen, that two legs are matched to the same object.
            # However, as long as the correct trigger bits are required (e.g. 2mu), this
            # might not be an actual problem (leg2 is always a sub-set of leg1 (?))
            all_legs_match = all_legs_match & ak.any(leg_mask, axis=1)
            leg_match_or = leg_match_or | leg_mask

        # final trigger decision
        fired_and_all_legs_match = fired & all_legs_match

        # check if an unprescaled L1 seed has fired as well
        l1_seeds_fired = ak_any([events.L1[l1_seed] for l1_seed in trigger.x.L1_seeds])

        fired_and_l1_fired = fired & l1_seeds_fired

        # store all intermediate results for subsequent selectors
        trigger_data = set_ak_bool(
            trigger_data,
            f"{trigger.name}.all_legs_match",
            all_legs_match,
        )
        trigger_data = set_ak_bool(
            trigger_data,
            f"{trigger.name}.fired_and_all_legs_match",
            fired_and_all_legs_match,
        )
        trigger_data = set_ak_bool(
            trigger_data,
            f"{trigger.name}.fired_and_all_legs_match_and_l1_fired",
            fired_and_l1_fired,
        )
        ids = ak.where(fired_and_l1_fired, np.float32(trigger.id), np.float32(np.nan))
        trigger_ids = ak.concatenate([trigger_ids, ak.singletons(ak.nan_to_none(ids))], axis=1)

    # store the fired trigger ids
    # trigger_ids = ak.concatenate(trigger_ids, axis=1)
    events = set_ak_int32(events, "trigger_ids", trigger_ids)
    events = set_ak_bool(events, "trigger_data", trigger_data)
    # check_trigger_data(trigger_data)
    return events, SelectionResult(
        steps={
            "trigger": trigger_data.any_fired,
        },
        # aux={
        #     "trigger_data": trigger_data,
        # },
    )


def check_trigger_data(trigger_data):
    for field in trigger_data.fields:
        if field == "any_fired":
            continue
        fired = trigger_data[field].fired
        fired_and_all_legs_match = trigger_data[field].fired_and_all_legs_match
        print(field)
        print("fired", ak.sum(fired))
        print("fired_and_all_legs_match", ak.sum(fired_and_all_legs_match))
        print("ratio", ak.sum(fired_and_all_legs_match) / ak.sum(fired))


@trigger_selection.init
def trigger_selection_init(self: Selector) -> None:
    if getattr(self, "config_inst", None) is None:
        return
    if getattr(self, "dataset_inst", None) is None:
        return

    # full used columns
    self.uses |= {
        optional(trigger.name)
        for trigger in self.config_inst.x.triggers
        # if trigger.applies_to_dataset(self.dataset_inst)
    }
    # add L1 seed columns
    for trigger in self.config_inst.x.triggers:
        self.uses |= {
            f"L1.{l1_seed}"
            for l1_seed in trigger.x.L1_seeds
        }

    # add L1 seed columns
    for trigger in self.config_inst.x.triggers:
        if not trigger.x("L1_seeds", None):
            logger.warning(f"Trigger '{trigger.name}' does not have L1 seeds defined")
        self.uses |= {
            f"L1.{l1_seed}"
            for l1_seed in trigger.x.L1_seeds
        }

    # testing: add HLT columns to keep columns
    self.config_inst.x.keep_columns["cf.ReduceEvents"] |= {
        f"HLT.{trigger.hlt_field}"
        for trigger in self.config_inst.x.triggers
        # if trigger.applies_to_dataset(self.dataset_inst)
    }


@selector(
    uses={"run", "trigger_ids"},
)
def trigger_lepton_crosscheck(
    self: Selector,
    events: ak.Array,
    lepton_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    This Selector ensures that the trigger selection is consistent with the lepton selection.
    e.g. events can only pass the ee selection if one of the ee triggers fired and at least two
    electrons have been selected.

    Requires the *trigger_selection* and *lepton_selection* selectors to be run before.
    The lepton_results are expected to contain steps starting with "Lep_".

    The trigger config objects are expected to have a *channels* attribute, which is a list of
    lepton channels that the trigger applies to (same names as the "Lep" channels).
    """
    # initialize the result object
    trigger_lepton_result = SelectionResult()

    # get all lepton channels from the lepton selection results
    channels = [step.replace("Lep_", "") for step in lepton_results.steps if step.startswith("Lep_")]

    triggers_dict = {
        channel: [trigger for trigger in self.config_inst.x.triggers if channel in trigger.x.channels]
        for channel in channels
    }

    def trigger_fired(events, triggers):
        if not triggers:
            return np.zeros(len(events), dtype=np.bool_)
        # check if one of the triggers fired
        return ak_any([ak.any(events.trigger_ids == trig.id, axis=1) for trig in triggers])

    for channel, triggers in triggers_dict.items():
        if not triggers:
            logger.warning(f"no triggers found for channel '{channel}'")
            continue

        # start with mask of events where triggers for this process fired
        trigger_mask = trigger_fired(events, triggers_dict[channel])
        trigger_lepton_result.steps[f"Trigger_{channel}"] = trigger_mask

        # check if the trigger selection is consistent with the lepton selection
        trigger_lepton_result.steps[f"TriggerAndLep_{channel}"] = trigger_mask & lepton_results.steps[f"Lep_{channel}"]

    # combine results of each individual channel
    trigger_lepton_result.steps["Trigger"] = ak_any([
        trigger_lepton_result.steps[f"Trigger_{channel}"]
        for channel in channels
    ])
    trigger_lepton_result.steps["TriggerAndLep"] = ak_any([
        trigger_lepton_result.steps[f"TriggerAndLep_{channel}"]
        for channel in channels
    ])

    return events, trigger_lepton_result


@selector(
    uses={"run", "trigger_ids"},
)
def data_double_counting(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Data double counting removal.

    The idea is as follows:
    - Each trigger defines one data stream that it belongs to via a tag.
    - We start with a mask that removes all events where triggers from this stream did not fire.
    - In the config, we define the order of the data stream priorities. When a trigger from a higher
        priority data stream fires, the event is removed
    """

    if self.dataset_inst.is_mc:
        # return always true for MC
        return events, SelectionResult(steps={"data_double_counting": np.ones(len(events), dtype=np.bool_)})

    process_inst = self.dataset_inst.processes.get_first()

    if process_inst.name == "data_jethtmet":
        # return always true for data_jethtmet
        return events, SelectionResult(steps={"data_double_counting": np.ones(len(events), dtype=np.bool_)})

    data_processes_ordered = [f"data_{stream}" for stream in self.config_inst.x.data_streams]
    if process_inst.name not in data_processes_ordered:
        raise Exception(f"data process {process_inst.name} not known for double counting removal")

    double_counting_result = SelectionResult()

    triggers_dict = {
        data_process: [trigger for trigger in self.config_inst.x.triggers if trigger.x.data_stream == data_process]
        for data_process in data_processes_ordered
    }

    def trigger_fired(events, triggers):
        if not triggers:
            return np.zeros(len(events), dtype=np.bool_)
        # check if one of the triggers fired
        return ak_any([ak.any(events.trigger_ids == trig.id, axis=1) for trig in triggers])

    # start with mask of events where triggers for this process fired
    double_counting_mask = trigger_fired(events, triggers_dict[process_inst.name])

    skip_veto = False
    for data_process in data_processes_ordered:
        if data_process == process_inst.name:
            # when arriving at the current process, we are done; loop over the triggers of the
            # remaining processes is only done for cross checks
            skip_veto = True

        # check if one of the triggers fired (and therefore belongs to some other data stream)
        fired = trigger_fired(events, triggers_dict[data_process])
        double_counting_result.steps[f"Trigger_{data_process}"] = fired

        if not skip_veto:
            # remove events that fired triggers from other data streams
            double_counting_mask = double_counting_mask & ~fired

    double_counting_result.steps["data_double_counting"] = double_counting_mask

    return events, double_counting_result


@selector(
    uses={trigger_selection, trigger_lepton_crosscheck, data_double_counting},
    produces={trigger_selection, trigger_lepton_crosscheck, data_double_counting},
)
def hbw_trigger_selection(
    self: Selector,
    events: ak.Array,
    lepton_results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Wrapper of all trigger selection methods.
    """
    # apply trigger selection methods
    events, trigger_result = self[trigger_selection](events, stats, **kwargs)
    events, trigger_lepton_result = self[trigger_lepton_crosscheck](events, lepton_results, **kwargs)
    events, double_counting_result = self[data_double_counting](events, **kwargs)

    # combine results
    hbw_trigger_result = trigger_result + trigger_lepton_result + double_counting_result

    return events, hbw_trigger_result
