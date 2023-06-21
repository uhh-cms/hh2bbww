# coding: utf-8


import order as od

from columnflow.util import maybe_import


ak = maybe_import("awkward")
np = maybe_import("numpy")
tf = maybe_import("tensorflow")


def assign_dataset_to_process(
        dataset_inst: od.Dataset,
        process_insts: list[od.Process],
) -> bool:
    """
    Assigns the dataset to exactly one process from a list of processes.
    Returns True when the matching was successful and False if not
    """

    if len(dataset_inst.processes) != 1:
        raise Exception("only 1 process inst is expected for each dataset")

    for i, proc_inst in enumerate(process_insts):
        leaf_procs = [p.name for p, _, _ in proc_inst.walk_processes(include_self=True)]
        if dataset_inst.processes.get_first() in leaf_procs:
            dataset_inst.x.ml_process = proc_inst
            return True

    return False


def predict_numpy_on_batch(
    model: tf.keras.Model,
    inputs: np.array,
    batch_size: int = 2 ** 16,
) -> np.array:
    """
    Helper function to allow predicting numpy arrays in batches
    """
    num_samples = inputs.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))

    # store list of predictions first
    predictions = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_inputs = inputs[start_idx:end_idx]
        batch_pred = model.predict_on_batch(batch_inputs)
        predictions.append(batch_pred)

    # concatenate batched predictions
    predictions = np.concatenate(predictions, axis=0)

    return predictions
