# coding: utf-8


from typing import Sequence
import order as od

from columnflow.util import maybe_import


ak = maybe_import("awkward")


def assign_dataset_to_process(
        dataset_inst: od.Dataset,
        process_insts: list[od.Process],
) -> None:
    """
    Assigns the dataset to exactly one process from a list of processes
    """

    if len(dataset_inst.processes) != 1:
        raise Exception("only 1 process inst is expected for each dataset")

    for i, proc_inst in enumerate(process_insts):
        leaf_procs = [p.name for p, _, _ in proc_inst.walk_processes(include_self=True)]
        if dataset_inst.processes.get_first() in leaf_procs:
            dataset_inst.x.ml_process = proc_inst
            return

    raise Exception(f"dataset {dataset_inst.name} is not matched to any of the given processes")


def strip_to_columns(
        events: ak.Array,
        columns: Sequence[str],
) -> ak.Array:
    from columnflow.columnar_util import remove_ak_column

    # NOTE: we might want to use Routes here
    # check that all relevant input features are present
    if not set(columns).issubset(set(events.fields)):
        raise Exception(
            f"The columns {set(events.fields).difference(set(columns)) "
            "are not present in the ML input events"
        )

    # remove columns that are not requested
    for var in events.fields:
        if var not in columns:
            events = remove_ak_column(events, var)

    return events
