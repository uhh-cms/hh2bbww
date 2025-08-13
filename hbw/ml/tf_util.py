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
        self.data = data  # Store original data for alternative method
        self.datasets = []
        self.counts = []
        class_factors = []

        for proc_inst, arrays in data.items():
            # ML WEIGHTING
            self.counts.append(len(arrays.features))
            if self.kind == "train":
                arrays = (arrays.features, arrays.target, arrays.train_weights)
            else:
                arrays = (arrays.features, arrays.target, arrays.validation_weights)
            self.tuple_length = len(arrays)
            self.datasets.append(tf.data.Dataset.from_tensor_slices(arrays))
            class_factors.append(proc_inst.x.sub_process_class_factor)

        # state attributes
        self.batches_seen = None

        if self.kind == "valid":
            # always batch validation data to the same size (100 batches)
            self.max_iter_valid = 100
            self.iter_smallest_process = 100
            return

        # determine batch sizes per dataset
        sum_weights = sum(class_factors)

        if isinstance(batch_size, int):
            total_batch_size = batch_size
        else:
            total_batch_size = sum([batch_size[proc_inst] for proc_inst in data.keys()])

        self.total_batch_size = total_batch_size  # Store for alternative method

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

        for proc_inst, batch_size, count, weight in zip(
            self.processes, self.batch_sizes, self.counts, class_factors,
        ):
            logger.info(
                f"Data of process {proc_inst.name} needs {math.ceil(count / batch_size)} steps to be seen "
                f"completely (count {count}, weight {weight}, batch size {batch_size})",
            )

    @property
    def n_datasets(self):
        return len(self.datasets)

    def __iter__(self):
        """Optimized iterator with GPU-friendly data pipeline and prefetching"""
        self.batches_seen = 0

        # Create the optimized dataset pipeline
        dataset = self._create_optimized_dataset()

        # Add crucial optimizations for GPU performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # If GPU is available, prefetch to GPU for even better performance
        if tf.config.list_physical_devices("GPU"):
            try:
                dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
            except Exception:
                # Fallback if prefetch_to_device fails
                pass

        return iter(dataset)

    def _create_optimized_dataset(self):
        """Create an optimized tf.data.Dataset that handles multi-process batching efficiently"""
        datasets = self.datasets

        if self.buffersize > 0 and self.kind == "train":
            # shuffling
            datasets = [
                dataset.shuffle(int(self.buffersize * count), reshuffle_each_iteration=True, seed=self.seed)
                for dataset, count in zip(datasets, self.counts)
            ]

        # Finite repetition to prevent infinite memory growth
        # Use a large but finite number for training, exact count for validation
        if self.kind == "train":
            # Large finite repetition - enough for long training but prevents infinite accumulation
            repeat_count = 10000  # Should be more than enough epochs
            datasets = [
                dataset.repeat(repeat_count)
                for dataset in datasets
            ]
        else:
            # For validation, calculate exact repetitions needed
            max_validation_epochs = 1000  # Reasonable upper bound
            datasets = [
                dataset.repeat(max_validation_epochs)
                for dataset in datasets
            ]

        if self.kind == "train":
            # batching
            datasets = [
                dataset.batch(bs_size)
                for dataset, bs_size in zip(datasets, self.batch_sizes)
            ]
        else:
            # for validation, use reasonable batch sizes instead of massive batches
            datasets = [
                dataset.batch(count // self.iter_smallest_process)
                for dataset, count in zip(datasets, self.counts)
            ]

        # Create a generator function that handles the batching logic with proper cleanup
        def batch_generator():
            its = [iter(dataset) for dataset in datasets]
            batches_seen = 0
            max_batches = getattr(self, "_max_batches", None)  # Allow external batch limit

            try:
                while True:
                    # Break if we've hit a batch limit (useful for debugging/testing)
                    if max_batches and batches_seen >= max_batches:
                        break

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

                    # Use optimized concatenation function
                    result = self._concat_batches_optimized(dataset_batches)
                    yield result
                    batches_seen += 1

                    # Periodic memory cleanup
                    if batches_seen % 100 == 0:
                        gc.collect()

            finally:
                # Cleanup iterators
                del its
                gc.collect()

        # Define output signature based on batched datasets (not original)
        batched_spec = datasets[0].element_spec
        output_signature = tuple(
            tf.TensorSpec(shape=(None,) + batched_spec[i].shape[1:],
                         dtype=batched_spec[i].dtype)
            for i in range(len(batched_spec))
        )

        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            batch_generator,
            output_signature=output_signature,
        )

        return dataset

    @tf.function(experimental_relax_shapes=True)
    def _concat_batches_optimized(self, dataset_batches):
        """GPU-optimized batch concatenation using tf.function for better performance

        Note: experimental_relax_shapes=True helps prevent memory accumulation
        from different batch shapes creating new computation graphs
        """
        return tuple(
            tf.concat([batch[i] for batch in dataset_batches], axis=0)
            for i in range(self.tuple_length)
        )

    def cleanup_resources(self):
        """Explicit cleanup method to prevent memory leaks"""
        # Clear any cached data
        if hasattr(self, "data"):
            del self.data
        if hasattr(self, "datasets"):
            del self.datasets

        # Force garbage collection
        gc.collect()

        # Clear TensorFlow function cache if needed
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass

    def set_batch_limit(self, max_batches):
        """Set a maximum number of batches for debugging/testing"""
        self._max_batches = max_batches

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
