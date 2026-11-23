# coding: utf-8
"""
Optuna search over class factors (process imbalance) and minor hyperparameters.

Design choices:
- Class factors are drawn from simple fractions (denominators = powers of 2), so the exact
  LCM-based base batch stays small and the batch sizes stay integer.
- Before any training, the resulting batch sizes and steps per epoch are computed from the event
  counts (same logic as get_batch_sizes). Trials below MIN_STEPS are pruned immediately, which
  makes the search cheap and keeps the ~100+ steps requirement.
- The objective must NOT be the class-factor-weighted training/validation loss: class factors
  change the loss itself, so it is not comparable between trials. Use a sampling-independent
  metric, here the macro one-vs-rest AUC (or a balanced metric of your choice).

Usage:
    pip install optuna
    python optimize_hparams.py --n-trials 40 --storage sqlite:///hhh_hpo.db
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import shlex
import subprocess
from fractions import Fraction
from pathlib import Path

import optuna

# ---------------------------------------------------------------------------------------------
# 1. Setup: edit to your model
# ---------------------------------------------------------------------------------------------

MIN_STEPS = 65  # minimal number of natural steps per epoch
# key in the result json written by fit_ml_model that is maximized ("macro_auc" or "sig_over_sqrtb"), see --metric
METRIC_KEY = "macro_auc"
TARGET_BATCHSIZE = 1024

# training event counts per process (from your log)
# Cat_3b (5 nodes incl. tt_custom); taken from the "Data of process ... (count N" log lines.
# Check them against YOUR log of this exact model, the counts depend on the category.
COUNTS = {
    "hhh_4b2w_2l2nu_c30_d40": 57998,
    "hhh_4b2w_2l2nu_c30_d499": 17767,
    "hhh_4b2w_2l2nu_c319_d419": 15733,
    "hhh_4b2w_2l2nu_c31_d40": 21979,
    "hhh_4b2w_2l2nu_c31_d42": 21565,
    "hhh_4b2w_2l2nu_c32_d4m1": 19596,
    "hhh_4b2w_2l2nu_c34_d49": 17249,
    "hhh_4b2w_2l2nu_c3m1_d40": 6882,
    "hhh_4b2w_2l2nu_c3m1_d4m1": 18106,
    "hhh_4b2w_2l2nu_c3m1p5_d4m0p5": 15785,
    "tthh": 27604,
    "tth": 333882,
    "tt_custom": 206159,
    "ttbb_custom": 64507,
}

# node -> sub processes. Single-process nodes carry their own class factor.
# optional processes: simply add or remove the entry in COUNTS, e.g.
# COUNTS["tt_custom"] = <number of training events>
_ALL_NODES = {
    "hhh_signal": [p for p in COUNTS if p.startswith("hhh_")],
    "tthh": ["tthh"],
    "tth": ["tth"],
    "ttbb_custom": ["ttbb_custom"],
    "tt_custom": ["tt_custom"],
}
# keep only nodes whose processes all have a count; everything else is ignored
NODES = {node: procs for node, procs in _ALL_NODES.items(
) if procs and all(p in COUNTS for p in procs)}
REFERENCE_NODE = "hhh_signal"

# hidden layer layouts (cast to tuple of ints by cast_ml_param_values)
LAYER_CHOICES = {
    "256x3": [256] * 3,
    "512x3": [512] * 3,
    "512x4": [512] * 4,
    "128x3": [128] * 3,
}

# integer class factors: exact batch sizes with the live (truncating) get_batch_sizes.
# only the ratios between the nodes matter (all-equal = all 1)
CF_CHOICES = [1, 2, 3]
# keep the hhh node sum at 9.5 (LCM 19) -> small, exact batch sizes
SUB_CF_CHOICES = [0.5, 1.0]

# Class factors are searched as NON-DECREASING along this order (rarest process first), e.g. never
# hhh=3 > tt=1. Nodes in one inner list are not ordered relative to each other. Nodes that are not in
# NODES are skipped, nodes missing here are searched freely with CF_CHOICES.
CF_ORDER = [["hhh_signal"], ["tthh"], ["tth"], ["ttbb_custom"], ["tt_custom"]]
FIRST_CF_CHOICES = [1, 2]   # factor of the first level
# increase of each following node w.r.t. the previous level
CF_STEP_CHOICES = [0, 1]
# the hhh process with very few events gets a reduced inner weight (searched)
LOW_STAT_HHH = "hhh_4b2w_2l2nu_c3m1_d40"


# ---------------------------------------------------------------------------------------------
# 2. Batch size / steps precheck (exact fractions, same logic as get_batch_sizes)
# ---------------------------------------------------------------------------------------------

def batch_sizes_from_config(class_factors: dict, sub_factors: dict, batchsize: int) -> dict:
    shares = {}
    for node, procs in NODES.items():
        cf = Fraction(class_factors.get(node, 1)).limit_denominator(10_000)
        weights = {p: Fraction(sub_factors.get(
            p, 1)).limit_denominator(10_000) for p in procs}
        total = sum(weights.values())
        for p, w in weights.items():
            shares[p] = cf * w / total

    base = functools.reduce(math.lcm, (s.denominator for s in shares.values()))
    sizes = {p: int(s * base) for p, s in shares.items()}
    scaler = batchsize // sum(sizes.values()) or 1
    return {p: bs * scaler for p, bs in sizes.items()}


# True if get_batch_sizes still uses float_lcm over node sums + int() truncation of the batch factor
USE_LEGACY_BATCH_LOGIC = True


def batch_sizes_legacy(class_factors: dict, sub_factors: dict, batchsize: int) -> dict:
    """Replica of the float_lcm version (class factors NOT in the LCM, int() truncation)."""
    def _lcm(x: Fraction, y: Fraction) -> Fraction:
        return Fraction(math.lcm(x.numerator, y.numerator), math.gcd(x.denominator, y.denominator))

    weights = {
        node: {p: Fraction(sub_factors.get(p, 1)).limit_denominator(
            10_000) for p in procs}
        for node, procs in NODES.items()
    }
    sums = {node: sum(w.values()) for node, w in weights.items()}
    lcm = functools.reduce(_lcm, sums.values())

    sizes = {}
    for node, w in weights.items():
        cf = Fraction(class_factors.get(node, 1)).limit_denominator(10_000)
        batch_factor = int(cf * lcm / sums[node])
        for p, wp in w.items():
            sizes[p] = int(wp * batch_factor)
    scaler = batchsize // sum(sizes.values()) or 1
    return {p: bs * scaler for p, bs in sizes.items()}


def min_steps(batch_sizes: dict) -> float:
    return min(COUNTS[p] / bs for p, bs in batch_sizes.items())


# ---------------------------------------------------------------------------------------------
# 3. Training hook: THIS is what you connect to your pipeline
# ---------------------------------------------------------------------------------------------

def run_training(settings: dict, trial: optuna.Trial | None = None) -> float:
    """
    Train one model with `settings` and return the validation metric (higher = better).

    `settings` contains:
        class_factors, sub_process_class_factors, batchsize, dropout, learning_rate,
        l2, epochs

    Options to implement this:
    a) Call your law task via subprocess with these values as --ml-model-settings, then load the
       validation predictions from the task output and compute `macro_auc` below.
    b) Call the training code in-process and return macro_auc on the validation set.

    NOTE: dropout / learning_rate / l2 must be added to `settings_parameters` and `bookkeep_params`
    of your MLClassifierBase (and defined as `_default__dropout` etc.), otherwise they cannot be
    changed per run and will not enter the output hash.

    For pruning, use OptunaPruningCallback below as a keras callback.
    """
    # your usual training command, but with the HPO model, e.g.
    #   export HPO_LAW_CMD="law run cf.MLTraining --version v1 --ml-model Cat_4b_hpo --config ... --workers 1"
    law_cmd = os.environ.get("HPO_LAW_CMD")
    if not law_cmd:
        raise RuntimeError(
            "set the HPO_LAW_CMD environment variable to your law training command")

    number = trial.number if trial is not None else 0
    result_file = Path(os.environ.get(
        "HPO_RESULT_DIR", "hpo_results")).resolve() / f"trial_{number}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.unlink(missing_ok=True)

    # identical settings were already trained in this run -> reuse
    key = json.dumps(settings, sort_keys=True)
    if key in _CACHE:
        return _CACHE[key]

    env = os.environ.copy()
    # read by the Cat_4b_hpo model definition
    env["HPO_SETTINGS"] = key
    # written at the end of fit_ml_model
    env["HPO_RESULT_FILE"] = str(result_file)
    proc = subprocess.run(shlex.split(law_cmd), env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"training failed (return code {proc.returncode}) for settings {key}")
    if not result_file.exists():
        raise RuntimeError(
            f"no result file {result_file}. If the training output already existed, law skipped the "
            "fit; remove the output of this model (--remove-output) and retry.",
        )

    result = json.loads(result_file.read_text())
    if METRIC_KEY not in result:
        raise RuntimeError(
            f"metric '{METRIC_KEY}' not in {result_file} (keys: {list(result)})")
    value = result[METRIC_KEY]
    _CACHE[key] = value
    return value


_CACHE: dict = {}


def macro_auc(y_true_onehot, y_pred, sample_weight=None) -> float:
    """Sampling-independent metric: mean one-vs-rest AUC over classes."""
    from sklearn.metrics import roc_auc_score
    n_classes = y_true_onehot.shape[1]
    aucs = [
        roc_auc_score(y_true_onehot[:, i],
                      y_pred[:, i], sample_weight=sample_weight)
        for i in range(n_classes)
    ]
    return float(sum(aucs) / len(aucs))


class OptunaPruningCallback:
    """
    Use as keras callback (subclass keras.callbacks.Callback in your code):
    reports the validation metric every epoch so bad trials are stopped early.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val_metric"):
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.monitor)
        if value is None:
            return
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# ---------------------------------------------------------------------------------------------
# 4. Objective
# ---------------------------------------------------------------------------------------------

def suggest_class_factors(trial: optuna.Trial) -> dict:
    """
    Integer class factors that are non-decreasing along CF_ORDER. The first level draws from
    FIRST_CF_CHOICES, every following node is max(previous level) + a step from CF_STEP_CHOICES,
    so every proposal is valid and no trial is wasted on an unordered combination.
    """
    class_factors = {}
    prev = None
    for level in CF_ORDER:
        level = [node for node in level if node in NODES]
        if not level:
            continue
        for node in level:
            if prev is None:
                class_factors[node] = trial.suggest_categorical(
                    f"cf_first_{node}", FIRST_CF_CHOICES)
            else:
                class_factors[node] = prev + \
                    trial.suggest_categorical(
                        f"cf_step_{node}", CF_STEP_CHOICES)
        prev = max(class_factors[node] for node in level)

    # nodes that are not part of the ordering are searched freely
    for node in NODES:
        if node not in class_factors:
            class_factors[node] = trial.suggest_categorical(
                f"cf_{node}", CF_CHOICES)
    return class_factors


def objective(trial: optuna.Trial) -> float:
    # integer class factors, non-decreasing along CF_ORDER (hhh <= tthh <= tth <= tt, ttbb)
    class_factors = suggest_class_factors(trial)
    # only ratios matter: (2, 2, 2, 2, 2) is the same as (1, 1, 1, 1, 1) -> skip duplicates
    if functools.reduce(math.gcd, class_factors.values()) > 1:
        raise optuna.TrialPruned(
            "class factors have a common divisor (duplicate of a smaller set)")
    sub_factors = {}
    if LOW_STAT_HHH in COUNTS:
        sub_factors[LOW_STAT_HHH] = trial.suggest_categorical(
            "sub_cf_low_stat_hhh", SUB_CF_CHOICES)
    batchsize = trial.suggest_categorical("batchsize", [512, 1024, 2048])

    # cheap precheck: skip configurations that violate the minimum steps
    batch_fn = batch_sizes_legacy if USE_LEGACY_BATCH_LOGIC else batch_sizes_from_config
    sizes = batch_fn(class_factors, sub_factors, batchsize)
    if any(bs < 1 for bs in sizes.values()):
        raise optuna.TrialPruned("a process would get a batch size of 0")
    steps = min_steps(sizes)
    trial.set_user_attr("min_steps", steps)
    trial.set_user_attr("batch_sizes", {p: int(b) for p, b in sizes.items()})
    if steps < MIN_STEPS:
        raise optuna.TrialPruned(f"min steps {steps:.0f} < {MIN_STEPS}")

    settings = {
        "class_factors": class_factors,
        "sub_process_class_factors": sub_factors,
        "batchsize": batchsize,
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.05),
        "layers": LAYER_CHOICES[trial.suggest_categorical("layers", list(LAYER_CHOICES))],
        # names as in DenseModelMixin / CallbacksBase
        "learningrate": trial.suggest_float("learningrate", 1e-4, 3e-3, log=True),
        "reduce_lr_patience": trial.suggest_int("reduce_lr_patience", 2, 6),
        "epochs": 100,
    }
    return run_training(settings, trial=trial)


def make_storage(spec: str):
    """
    "sqlite:///x.db" is used as is. A path ending in ".log" uses Optuna's file-based journal storage,
    which is safer than sqlite when several workers write from a network file system.
    """
    if not spec.endswith(".log"):
        return spec
    from optuna.storages import JournalStorage
    try:
        from optuna.storages.journal import JournalFileBackend
        return JournalStorage(JournalFileBackend(spec))
    except ImportError:  # older optuna versions
        from optuna.storages import JournalFileStorage
        return JournalStorage(JournalFileStorage(spec))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--storage", default="sqlite:///hhh_hpo.db")
    parser.add_argument("--study-name", default="hhh_class_factors")
    parser.add_argument("--seed", type=int, default=1,
                        help="use a DIFFERENT seed for each parallel process")
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="trainings run in parallel from this one process (each trial is its own law subprocess)",
    )
    parser.add_argument(
        "--metric", default="macro_auc", choices=["macro_auc", "sig_over_sqrtb"],
        help="quantity to maximize; use a different study name for each metric",
    )
    args = parser.parse_args()

    global METRIC_KEY
    METRIC_KEY = args.metric

    study = optuna.create_study(
        study_name=args.study_name,
        storage=make_storage(args.storage),
        load_if_exists=True,  # resumable, and several workers can share the same storage
        direction="maximize",
        # constant_liar: parallel trials do not all propose the same point
        sampler=optuna.samplers.TPESampler(
            seed=args.seed, multivariate=True, constant_liar=True),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=8, n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=args.n_trials,
                   n_jobs=args.n_jobs, catch=(RuntimeError,))

    print("best value:", study.best_value)
    print("best params:", study.best_params)
    print("best min_steps:", study.best_trial.user_attrs.get("min_steps"))
    try:
        print(optuna.importance.get_param_importances(study))
    except Exception:
        pass


if __name__ == "__main__":
    main()
