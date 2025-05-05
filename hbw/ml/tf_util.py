# coding: utf-8

from __future__ import annotations
import gc

import math
import numpy as np
import tensorflow as tf

import law

from columnflow.util import DotDict

import order as od

logger = law.logger.get_logger(__name__)


class MultiDataset(object):
    # def auto_resolve_weights(self, data, max_diff_int: float = 0.3):
    #     """
    #     Represents cross-section weighting
    #     """
    #     rel_sumw_dict = {proc_inst: {} for proc_inst in data.keys()}
    #     factor = 1
    #     smallest_sumw = None
    #     for proc_inst, arrays in data.items():
    #         sumw = np.sum(arrays.weights)
    #         if not smallest_sumw or smallest_sumw >= sumw:
    #             smallest_sumw = sumw

    #     for proc_inst, arrays in data.items():
    #         sumw = np.sum(arrays.weights)
    #         rel_sumw = sumw / smallest_sumw
    #         rel_sumw_dict[proc_inst] = rel_sumw

    #         if (rel_sumw - round(rel_sumw)) / rel_sumw > max_diff_int:
    #             factor = 2

    #     rel_sumw_dict = {proc_inst: int(rel_sumw * factor) for proc_inst, rel_sumw in rel_sumw_dict.items()}

    #     return list(rel_sumw_dict.values())

    def __init__(
        self,
        data: DotDict[od.Process, DotDict[str, np.array]],
        batch_size: DotDict[od.Process, int] | int = 128,
        correct_batch_size: bool | str = "down",
        kind: str = "train",
        seed: int | None = None,
        buffersize: int = 0,  # buffersize=0 means no shuffle
    ):
        super().__init__()

        assert correct_batch_size in ("up", "down", True, False)

        assert kind in ["train", "valid"]
        self.kind = kind
        self.seed = seed
        self.buffersize = buffersize

        # create datasets, store counts and relative weights
        self.processes = tuple(data.keys())
        self.datasets = []
        self.counts = []
        class_factors = []

        for proc_inst, arrays in data.items():
            arrays = (arrays.features, arrays.target, arrays.train_weights)
            # ML WEIGHTING
            self.tuple_length = len(arrays)
            self.datasets.append(tf.data.Dataset.from_tensor_slices(arrays))
            self.counts.append(len(arrays[0]))
            class_factors.append(proc_inst.x.sub_process_class_factor)

        # if class_factor_mode == "sub_process_class_factor":
        #     class_factors = [proc_inst.x.sub_process_class_factor for proc_inst in data.keys()]
        # elif class_factor_mode == "auto":
        #     class_factors = self.auto_resolve_weights(data)

        # state attributes
        self.batches_seen = None

        # determine batch sizes per dataset
        sum_weights = sum(class_factors)

        if isinstance(batch_size, int):
            total_batch_size = batch_size
        else:
            total_batch_size = sum([batch_size[proc_inst] for proc_inst in data.keys()])

        if isinstance(batch_size, int):
            # check if requested batch size and weights are compatible
            if remainder := total_batch_size % sum_weights:
                msg = (
                    f"total_batch_size ({total_batch_size}) should be dividable by sum of "
                    f"process weights ({sum_weights}) to correctly weight processes as requested. "
                )
                if correct_batch_size:
                    if isinstance(correct_batch_size, str) and correct_batch_size.lower() == "down":
                        total_batch_size -= remainder
                    else:
                        total_batch_size += sum_weights - remainder
                    msg += f"total_batch_size has been corrected to {total_batch_size}"
                logger.warning(msg)

            self.batch_sizes = []
            carry = 0.0
            for weight in class_factors:
                bs = weight / sum_weights * total_batch_size - carry
                bs_int = int(round(bs))
                carry = bs_int - bs
                self.batch_sizes.append(bs_int)
        else:
            # NOTE: when passing a dict, the batch size and sub_process_class_factors have been pre-calculated
            # therefore no check is needed
            self.batch_sizes = [batch_size[proc_inst] for proc_inst in data.keys()]

        if total_batch_size != sum(self.batch_sizes):
            print(f"total_batch_size is {sum(self.batch_sizes)} but should be {batch_size}")

        self.max_iter_valid = math.ceil(max([c / bs for c, bs in zip(self.counts, self.batch_sizes)]))
        self.iter_smallest_process = math.ceil(min([c / bs for c, bs in zip(self.counts, self.batch_sizes)]))
        gc.collect()

        for proc_inst, batch_size, count, weight in zip(self.processes, self.batch_sizes, self.counts, class_factors):
            logger.info(
                f"Data of process {proc_inst.name} needs {math.ceil(count / batch_size)} steps to be seen completely "
                f"(count {count}, weight {weight}, batch size {batch_size})",
            )

    @property
    def n_datasets(self):
        return len(self.datasets)

    def __iter__(self):
        self.batches_seen = 0

        datasets = self.datasets

        if self.buffersize > 0 and self.kind == "train":
            # shuffling
            datasets = [
                dataset.shuffle(int(self.buffersize * count), reshuffle_each_iteration=False, seed=self.seed)
                for dataset, count in zip(datasets, self.counts)
            ]

        # repitition
        datasets = [
            dataset.repeat(-1)
            for dataset in datasets
        ]

        # batching
        datasets = [
            dataset.batch(bs_size)
            for dataset, bs_size in zip(datasets, self.batch_sizes)
        ]

        its = [iter(dataset) for dataset in datasets]
        while True:
            dataset_batches = []
            do_continue = False
            do_break = False
            for i, it in enumerate(its):
                try:
                    dataset_batches.append(next(it))
                except tf.errors.DataLossError as e:
                    print(f"\nDataLossError in dataset {i}:\n{e}\n")
                    do_continue = True
                    break
                except StopIteration:
                    do_break = True
                    break

            if do_continue:
                continue
            if do_break:
                break

            yield tuple(tf.concat([batch[i] for batch in dataset_batches], axis=0) for i in range(self.tuple_length))

            self.batches_seen += 1
            logger.debug(self.kind, self.batches_seen, self.iter_smallest_process, self.max_iter_valid)
            # if self.kind == "valid" and self.batches_seen >= self.iter_smallest_process:
            #     break

    def map(self, *args, **kwargs):
        for key, dataset in list(self._datasets.items()):
            self._datasets[key] = dataset.map(*args, **kwargs)


_cumulated_crossentropy_epsilon = 1e-7


@tf.function
def _cumulated_crossenropy_from_logits(y_true, y_pred, axis):
    # implementation of the log-sum-exp trick that makes computing log of a sum of softmax outputs numerically stable
    # paper: Muller & Smith, 2020: A Hierarchical Loss for Semantic Segmentation
    # target = tf.math.subtract(1., y_true)

    b = tf.math.reduce_max(y_pred, axis=axis, keepdims=True)
    b_C = tf.math.reduce_max(y_pred * y_true, axis=axis, keepdims=True)

    numerator = b_C + tf.math.log(tf.math.reduce_sum(y_true * tf.math.exp(y_pred - b_C), axis=axis, keepdims=True))
    denominator = b + tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred - b), axis=axis, keepdims=True))

    return ((numerator - denominator) / tf.math.reduce_sum(y_true, axis=axis))[:, 0]


@tf.function
def cumulated_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
    if from_logits:
        return _cumulated_crossenropy_from_logits(y_true, y_pred, axis=axis)

    epsilon = tf.constant(_cumulated_crossentropy_epsilon, dtype=y_pred.dtype)
    output = y_pred / (tf.reduce_sum(y_pred, axis, True) + epsilon)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)

    return -tf.reduce_sum(y_true * tf.math.log(output), axis=axis)
