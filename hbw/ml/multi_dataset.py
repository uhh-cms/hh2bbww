# coding: utf-8

from __future__ import annotations
import gc

import math
import numpy as np
import tensorflow as tf

from columnflow.util import DotDict

import order as od


class MultiDataset(object):

    def __init__(
        self,
        data: DotDict[od.Process, DotDict[str, np.array]],
        batch_size: int = 128,
        kind: str = "train",
        seed: int | None = None,
        buffersize: int = 0,  # buffersize=0 means no shuffle
    ):
        super().__init__()

        assert kind in ["train", "valid"]
        self.kind = kind
        self.seed = seed
        self.buffersize = buffersize

        # create datasets, store counts and relative weights
        self.datasets = []
        self.counts = []
        self.weights = []

        for proc_inst, arrays in data.items():
            arrays = (arrays.inputs, arrays.target, arrays.weights)
            self.tuple_length = len(arrays)
            self.datasets.append(tf.data.Dataset.from_tensor_slices(arrays))
            self.counts.append(len(arrays[0]))
            self.weights.append(proc_inst.x.ml_process_weight)

        # state attributes
        self.batches_seen = None

        # determine batch sizes per dataset
        self.batch_sizes = []
        sum_weights = sum(self.weights)

        carry = 0.0
        for weight in self.weights:
            bs = weight / sum_weights * batch_size - carry
            bs_int = int(round(bs))
            carry = bs_int - bs
            self.batch_sizes.append(bs_int)

        if batch_size != sum(self.batch_sizes):
            print(f"batch size is {sum(self.batch_sizes)} but should be {batch_size}")

        self.max_iter_valid = int(math.ceil(max([c / bs for c, bs in zip(self.counts, self.batch_sizes)])))
        self.iter_smallest_process = int(math.ceil(min([c / bs for c, bs in zip(self.counts, self.batch_sizes)])))
        gc.collect()

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
            if self.kind == "valid" and self.batches_seen >= self.max_iter_valid:
                break

    def map(self, *args, **kwargs):
        for key, dataset in list(self._datasets.items()):
            self._datasets[key] = dataset.map(*args, **kwargs)
