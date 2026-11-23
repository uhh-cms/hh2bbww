# coding: utf-8

"""
First implementation of DNN for HH analysis, generalized (TODO)
"""

from __future__ import annotations
import logging

from typing import Any
import yaml

import law
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd

from columnflow.types import Sequence
from columnflow.util import maybe_import, dev_sandbox, DotDict
from columnflow.columnar_util import set_ak_column
from hbw.config.dl.variables import add_gatja_scores_variables

from hbw.ml.data_loader import input_features_sanity_checks
from hbw.ml.base import MLClassifierBase
from hbw.config.processes import prepare_ml_processes

np = maybe_import("numpy")
ak = maybe_import("awkward")
pickle = maybe_import("pickle")

logger = law.logger.get_logger(__name__)

logger = logging.getLogger("luigi-interface")

# patch, allowing user to fall back to old versions
use_old_version = law.config.get_expanded("analysis", "use_old_version", False)


# load all ml modules here
if law.config.has_option("analysis", "ml_modules"):
    for m in law.config.get_expanded("analysis", "ml_modules", [], split_csv=True):
        logger.debug(f"loading ml module '{m}'")
        maybe_import(m.strip())


class Training_GATJA_Base(MLClassifierBase):
    ml_inputs_producer_name = "input_features_gatja"
    preparation_producer_name = "gatja_prepml"
    _default__seed: int = 1

    class_weight_power = 0.0
    dataset_weight_power = 0.0
    weight_clip_percentile = None
    use_physics_weights = False
    early_stopping_start_epoch = 0

    store_name = "test_v2"

    train_val_test_split = (0.75, 0.15, 0.1)
    folds = 5
    _default__epochs = 300
    _default__batchsize = 2048
    _default__negative_weights = "handle"

    stage_one_index_node: int = 5
    stage_one_index_neigh1: int = 4
    stage_one_index_neigh2: int = 4
    initial_lr = 1e-4
    warmup_epochs = 2
    warmup_lr = 1e-9

    _downsample = {"tt_custom": 0.1}

    _jet_downsample = {
        "tt_custom": (1.0, 1.0, 1.0),
        "ttbb_custom": (1.0, 1.0, 1.0),
        "tth": (1.0, 1.0, 1.0),
        "tthh_4b": (1.0, 1.0, 1.0),
        "hhh_4b2w_2l2nu_c30_d40": (1.0, 1.0, 1.0),
    }

    jet_classes = ("higgs", "top", "other")
    n_jets = 8
    padding_value = -6.0

    monitor_metric = "val_loss"
    monitor_mode = "min"
    early_stopping_min_delta = 0.0
    early_stopping_patience = 50
    warmup_fraction = None
    lamb_weight_decay = None

    settings_parameters = MLClassifierBase.settings_parameters | {"seed"}
    bookkeep_params = MLClassifierBase.bookkeep_params | {"seed"}

    @property
    def n_scores(self) -> int:
        return self.n_jets * len(self.jet_classes)

    def cast_ml_param_values(self):
        super().cast_ml_param_values()
        self.seed = int(self.seed)

    def training_producers(self, analysis_inst, requested_producers):
        return ["gatja_event_weight", self.ml_inputs_producer_name]

    def evaluation_producers(self, analysis_inst, requested_producers: Sequence[str]) -> list[str]:
        return [self.ml_inputs_producer_name]

    _default__processes = (
        # "tt", "ttbb_custom",
        "hhh_4b2w_2l2nu_c30_d40",
        "tthh_4b",
        # "ttb_custom",
        # "tt2b_custom",
        "ttbb_custom",
        "tt_custom",
        "tth",
        # "ttzz",
        # "ttzh",
    )

    train_nodes = {
        "hhh_4b2w_2l2nu_c30_d40": {"ml_id": 0},
        "tthh_4b": {"ml_id": 1},
        "ttbb_custom": {"ml_id": 2},
        "tt_custom": {"ml_id": 3},
        "tth": {"ml_id": 4},
    }

    _default__class_factors = {"hhh_4b2w_2l2nu_c30_d40": 1,
                               "tthh_4b": 1, "ttbb_custom": 1, "tt_custom": 1, "tth": 1}
    _default__sub_process_class_factors = {
        "hhh_4b2w_2l2nu_c30_d40": 1, "tthh_4b": 1, "ttbb_custom": 1, "tt_custom": 1, "tth": 1,
    }

    input_features = [
        "jetPT1", "jetPT2", "jetPT3", "jetPT4", "jetPT5", "jetPT6", "jetPT7", "jetPT8",
        "jetEta1", "jetEta2", "jetEta3", "jetEta4", "jetEta5", "jetEta6", "jetEta7", "jetEta8",
        "jetPhi1", "jetPhi2", "jetPhi3", "jetPhi4", "jetPhi5", "jetPhi6", "jetPhi7", "jetPhi8",
        "bjetAverageMass", "jetAverageMass",
        "bjetAverageMassSqr", "jetNumber", "bjetNumber",
        "minDeltaRbb", "btag_weight", "weights",
        "jetHT", "bjetHT", "lightjetHT",
        "leptonPT1", "leptonEta1", "leptonPhi1",
        "leptonPT2", "leptonEta2", "leptonPhi2",
        "met", "metPhi",
        "jetMinChiHiggsIndex1", "jetSecMinChiHiggsIndex1", "jetMinChiHiggsIndex2", "jetSecMinChiHiggsIndex2",
        "jetMinChiHiggsIndex3", "jetSecMinChiHiggsIndex3", "jetMinChiHiggsIndex4",
        "jetSecMinChiHiggsIndex4", "jetMinChiHiggsIndex5", "jetSecMinChiHiggsIndex5", "jetMinChiHiggsIndex6",
        "jetSecMinChiHiggsIndex6", "jetMinChiHiggsIndex7",
        "jetSecMinChiHiggsIndex7", "jetMinChiHiggsIndex8", "jetSecMinChiHiggsIndex8",
        "jetBTagDisc1", "jetBTagDisc2", "jetBTagDisc3", "jetBTagDisc4",
        "jetBTagDisc5", "jetBTagDisc6", "jetBTagDisc7", "jetBTagDisc8",
        "jetTopMatched1", "jetTopMatched2", "jetTopMatched3", "jetTopMatched4",
        "jetTopMatched5", "jetTopMatched6", "jetTopMatched7", "jetTopMatched8",
        "jetHiggsMatched1", "jetHiggsMatched2", "jetHiggsMatched3", "jetHiggsMatched4",
        "jetHiggsMatched5", "jetHiggsMatched6", "jetHiggsMatched7", "jetHiggsMatched8",
    ]

    graph_feature_names = (
        "btag_weight",
        "node_pt", "node_eta", "node_phi", "node_minChiIdx", "node_btagDisc",
        "jetHT", "jetNumber", "jetAverageMass",
        "leptonPT1", "leptonEta1", "leptonPhi1",
        "leptonPT2", "leptonEta2", "leptonPhi2", "met",
        "n1_pt", "n1_eta", "n1_phi", "n1_btagDisc",
        "n2_pt", "n2_eta", "n2_phi", "n2_btagDisc",
    )

    rest_cols = (
        "jetHT",  # "bjetHT", "lightjetHT",
        "jetNumber", "jetAverageMass",
        "leptonPT1", "leptonEta1", "leptonPhi1",
        "leptonPT2", "leptonEta2", "leptonPhi2",
        "met",
    )

    def sandbox(self, task: law.Task) -> str:
        return dev_sandbox("bash::$HBW_BASE/sandboxes/venv_ml_plotting.sh")

    @property
    def n_features(self) -> int:
        # return len(self.graph_feature_names) - 1
        return len(self.graph_feature_names) - 1

    def uses(self, config_inst) -> set:
        columns = set(self.input_features)
        columns |= {
            "process_id",
            "normalization_weight",
            # "stitched_normalization_weight",
            "event_weight",
        }
        return columns

    def produces(self, config_inst) -> set:
        return {f"gatja_output_{i}" for i in range(self.n_jets * len(self.jet_classes))}

    def setup(self) -> None:
        for config_inst in self.config_insts:
            if config_inst.has_tag(f"{self.cls_name}_called"):
                continue

            add_gatja_scores_variables(config_inst, self.n_scores)
            prepare_ml_processes(config_inst, self.train_nodes,
                                 self.sub_process_class_factors)

            for i in range(self.n_scores):
                name = f"gatja_output_{i}"
                if name in config_inst.variables:
                    continue
                jet_slot = i // len(self.jet_classes) + 1
                jet_class = self.jet_classes[i % len(self.jet_classes)]
                config_inst.add_variable(
                    name=name,
                    expression=name,
                    null_value=-10,
                    binning=(40, 0., 1.),
                    x_title=f"GATJA {jet_class} score (jet {jet_slot})",
                )

            config_inst.add_tag(f"{self.cls_name}_called")

        self.valid_ml_id_sanity_check()

    def _proc_data_to_dataframe(self, proc_data, process_name: str) -> "pd.DataFrame":
        import pandas as pd
        df = pd.DataFrame(
            np.asarray(proc_data.features),
            columns=list(proc_data.input_features),
        )
        frac = self._downsample.get(process_name, 1.0)
        if frac < 1.0:
            import zlib
            seed = self.seed + zlib.crc32(process_name.encode())
            rng = np.random.default_rng(seed % 2**32)
            # rng = np.random.default_rng(abs(hash(process_name)) % 2**32)
            df = df.loc[rng.random(len(df)) < frac].reset_index(drop=True)
        df["process"] = process_name
        return df

    def _safe_lookup(self, frame: "pd.DataFrame", row_labels: Sequence[int], column_names: Sequence[str]) -> np.ndarray:
        if len(row_labels) == 0:
            return np.array([], dtype=float)
        subset = frame.loc[row_labels]
        column_index = subset.columns.get_indexer(column_names)
        row_index = np.arange(len(row_labels))
        return subset.to_numpy()[row_index, column_index]

    def matched_column(self, cls: str, slot: int) -> str:
        return f"jet{cls.capitalize()}Matched{slot}"

    def compute_padding_mask(self, working: "pd.DataFrame", index: int) -> np.ndarray:
        slot = index + 1
        by_count = (slot > working["jetNumber"].to_numpy())
        return by_count

    def _create_graphs_core(self, df: "pd.DataFrame", index: int, drop_empty: bool = False):
        import pandas as pd
        # working = df.copy()
        working = df.drop(columns=["process"], errors="ignore").copy()

        low_index_column = f"jetMinChiHiggsIndex{index + 1}"
        second_index_column = f"jetSecMinChiHiggsIndex{index + 1}"

        working.loc[working[low_index_column] > 7, low_index_column] = index
        working.loc[working[low_index_column] == -6, low_index_column] = index
        sec_col = working[second_index_column]
        working.loc[sec_col > 7, second_index_column] = index
        working.loc[sec_col == -6, second_index_column] = index

        node_cols = [
            "jetPT" + str(index + 1),
            "jetEta" + str(index + 1),
            "jetPhi" + str(index + 1),
            "jetMinChiHiggsIndex" + str(index + 1),
            "jetBTagDisc" + str(index + 1),
        ]

        rest_cols = list(self.rest_cols)

        if drop_empty:
            working = working.loc[working[f"jetPT{index + 1}"] >= 0].copy()

        btag_weight = working["btag_weight"].to_numpy()

        node_part = working[node_cols].to_numpy()
        rest_part = working[rest_cols].to_numpy()

        low_partner = (working[low_index_column] + 1).astype(int).astype(str)
        second_partner = (
            working[second_index_column] + 1).astype(int).astype(str)

        # slot = index + 1
        # per_class, matched_any = [], None
        # for cls in self.jet_classes:
        #     if cls == "other":
        #         per_class.append(None)
        #         continue
        #     m = working[self.matched_column(cls, slot)].to_numpy(dtype=bool)
        #     matched_any = m if matched_any is None else (matched_any | m)
        #     per_class.append(m)

        # labels = np.vstack([
        #     (~matched_any if m is None else m).astype(int)
        #     for m in per_class
        # ])

        low_rows = pd.Series("jetPT" + low_partner, index=working.index)
        neighbour = [
            self._safe_lookup(working, low_rows.index, low_rows),
            # .drop(empty_index)) for the following lines is droped -> use instead padding
            self._safe_lookup(working, low_rows.index, pd.Series(
                "jetEta" + low_partner, index=working.index)),
            self._safe_lookup(working, low_rows.index, pd.Series(
                "jetPhi" + low_partner, index=working.index)),
            self._safe_lookup(working, low_rows.index, pd.Series(
                "jetBTagDisc" + low_partner, index=working.index,
            )),
        ]

        second_rows = pd.Series("jetPT" + second_partner, index=working.index)
        neighbour2 = [
            self._safe_lookup(working, second_rows.index, second_rows),
            # .drop(empty_index)) for the following lines is droped -> use instead padding
            self._safe_lookup(working, second_rows.index, pd.Series(
                "jetEta" + second_partner, index=working.index)),
            self._safe_lookup(working, second_rows.index, pd.Series(
                "jetPhi" + second_partner, index=working.index)),
            self._safe_lookup(working, second_rows.index, pd.Series(
                "jetBTagDisc" + second_partner, index=working.index,
            )),
        ]
        graph_data = np.hstack(
            (btag_weight[:, None], node_part, rest_part,
             np.array(neighbour).T, np.array(neighbour2).T),
        ).astype(np.float32)

        padding_mask = self.compute_padding_mask(working, index)
        event_weights = working["weights"].to_numpy(dtype=np.float32)

        return graph_data, padding_mask, event_weights

    def create_graphs(
        self,
        df: "pd.DataFrame",
        index: int,
        drop_empty: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._create_graphs_core(df, index=index, drop_empty=drop_empty)

    def prepare_stage_one_graph_tensors(
        self,
        df_sample: "pd.DataFrame",
        max_njets: int = 8,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    ]:
        sample_blocks = []
        label_blocks = []
        btag_weights_blocks = []
        process_blocks = []
        padding_blocks = []
        event_weights_blocks = []

        for index in range(max_njets):
            sample_block, label_block, padding_mask, event_weight_block = self.create_graphs(
                df_sample, index)
            real = ~padding_mask
            btag_weights_blocks.append(sample_block[real, 0])
            sample_blocks.append(sample_block[real, 1:])
            label_blocks.append(label_block.T[real].astype(np.float32))
            process_blocks.append(df_sample["process"].to_numpy()[real])
            padding_blocks.append(padding_mask[real])
            event_weights_blocks.append(event_weight_block[real])

        sample = np.concatenate(sample_blocks, axis=0)
        label = np.concatenate(label_blocks, axis=0)
        btag_weights = np.concatenate(btag_weights_blocks, axis=0)
        process_names = np.concatenate(process_blocks, axis=0)
        padded = np.concatenate(padding_blocks, axis=0)
        event_weights = np.concatenate(event_weights_blocks, axis=0)

        assert label.shape[1] == len(
            self.jet_classes), f"label shape {label.shape} != (n, 3)"
        assert len(sample) == len(label) == len(btag_weights) == len(
            process_names) == len(padded) == len(event_weights)

        return sample, label, btag_weights, process_names, padded, event_weights

    def _log_class_composition(self, label, process_names, event_weights=None, tag: str = "") -> None:
        groups = [("ALL", np.ones(len(label), dtype=bool))]
        groups += [(str(p), process_names == p)
                   for p in np.unique(process_names)]

        for name, sel in groups:
            n = int(sel.sum())
            if n == 0:
                continue
            counts = label[sel].sum(axis=0)
            parts = [
                f"{cls}={int(c):,} ({c / n:.2%})"
                for cls, c in zip(self.jet_classes, counts)
            ]
            line = f"  {name:<22} n_jets={n:>10,}   " + "   ".join(parts)

            if event_weights is not None:
                w_counts = (label[sel] * event_weights[sel, None]).sum(axis=0)
                line += "   | weighted: " + " ".join(
                    f"{cls}={w:.1f}" for cls, w in zip(self.jet_classes, w_counts)
                )
            logger.info(line)

    def make_model_gnn(self, input_shape, index_node: int, index_neigh1: int, index_neigh2: int):
        import tensorflow
        from tensorflow import keras
        layers = tensorflow.keras.layers
        from tensorflow.keras.layers import BatchNormalization

        inputs = keras.Input(shape=input_shape)
        input_node_value = inputs[:, :index_node]
        input_neigh1_value = inputs[:, -
                                    (index_neigh1 + index_neigh2):-index_neigh2]
        input_neigh2_value = inputs[:, -index_neigh2:]
        input_rest = inputs[:, index_node:-(index_neigh1 + index_neigh2)]

        def dense_layer(values, units: int):
            values = layers.Dense(units)(values)
            values = layers.LeakyReLU()(values)
            return values

        node_value = dense_layer(input_node_value, 256)
        node_value = layers.Concatenate()([node_value, input_node_value])
        node_value = dense_layer(node_value, 128)

        neigh1_value = dense_layer(input_neigh1_value, 256)
        neigh1_value = layers.Concatenate()([neigh1_value, input_neigh1_value])
        neigh1_value = dense_layer(neigh1_value, 128)

        neigh2_value = dense_layer(input_neigh2_value, 256)
        neigh2_value = layers.Concatenate()([neigh2_value, input_neigh2_value])
        neigh2_value = dense_layer(neigh2_value, 128)

        weight_main = layers.Softmax()(keras.ops.matmul(
            keras.ops.transpose(node_value), node_value))
        weight_neigh1 = layers.Softmax()(keras.ops.matmul(
            keras.ops.transpose(node_value), neigh1_value))
        weight_neigh2 = layers.Softmax()(keras.ops.matmul(
            keras.ops.transpose(node_value), neigh2_value))

        rest = dense_layer(input_rest, 256)
        rest = layers.Concatenate()([rest, input_rest])
        rest = dense_layer(rest, 128)

        node = node_value * weight_main[:, 0]
        neigh1 = neigh1_value * weight_neigh1[:, 0]
        neigh2 = neigh2_value * weight_neigh2[:, 0]

        max_embed = layers.Concatenate()(
            [node, layers.Maximum()([neigh1, neigh2])])
        x_dense = layers.Concatenate()([rest, max_embed])
        x_dense = layers.Dropout(0.15)(x_dense)

        def dropout_layer(values, units: int):
            values = layers.Dense(units)(values)
            values = BatchNormalization()(values)
            values = layers.LeakyReLU()(values)
            values = layers.Dropout(0.15)(values)
            values = layers.Concatenate()([values, x_dense])
            return values

        x = dropout_layer(x_dense, 2048)
        x = dropout_layer(x, 2048)
        x = dropout_layer(x, 2048)
        x = dropout_layer(x, 2048)
        x = dropout_layer(x, 1024)
        x = dropout_layer(x, 512)
        x = dropout_layer(x, 128)
        x = dropout_layer(x, 32)
        # outputs = layers.Dense(3, activation="softmax")(x)
        outputs = layers.Dense(len(self.jet_classes), activation="softmax")(x)

        return keras.Model(inputs, outputs)

    def compute_dataset_weights(self, process_names: np.ndarray) -> dict[str, float]:
        procs, counts = np.unique(process_names, return_counts=True)
        counts = counts.astype(np.float64)

        w_ds = counts ** (-self.dataset_weight_power)
        w_ds /= np.average(w_ds, weights=counts)
        return dict(zip(procs, w_ds))

    def _dataset_weight_array(self, process_names: np.ndarray, ds_weights: dict) -> np.ndarray:
        procs, inverse = np.unique(process_names, return_inverse=True)
        lut = np.array([ds_weights.get(p, 1.0)
                       for p in procs], dtype=np.float32)
        return lut[inverse]

    def _finalize_weights(self, w: np.ndarray, tag: str, label: np.ndarray | None = None) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64)

        if self.weight_clip_percentile is not None:
            positive = w[w > 0]
            if positive.size:
                cap = np.percentile(positive, self.weight_clip_percentile)
                w = np.minimum(w, cap)
        nonzero = w > 0
        if np.any(nonzero):
            w /= w[nonzero].mean()
        return w.astype(np.float32)

    def compute_class_weights(self, label: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        if weights is None:
            class_sums = label.sum(axis=0).astype(np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            class_sums = (label.astype(np.float64) * w[:, None]).sum(axis=0)

        w_cls = class_sums ** (-self.class_weight_power)
        w_cls /= np.average(w_cls, weights=class_sums)

        return w_cls.astype(np.float32)

    def make_optimizer(self, train_data_length: int):
        import tensorflow as tf

        class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, initial_lr, decay_steps, warmup_steps=2, warmup_lr=1e-9, name=None):
                super().__init__()
                self.initial_lr = float(initial_lr)
                self.decay_steps = int(decay_steps)
                self.warmup_steps = int(warmup_steps)
                self.warmup_lr = float(warmup_lr)
                self.name = name
                self.cosine = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=self.initial_lr,
                    decay_steps=self.decay_steps,
                )

            def __call__(self, step):
                import tensorflow as tf
                step = tf.cast(step, tf.float32)
                warmup_steps = tf.cast(self.warmup_steps, tf.float32)

                def warmup():
                    return self.warmup_lr + (self.initial_lr - self.warmup_lr) * (step / warmup_steps)

                def decay():
                    return self.cosine(step - warmup_steps)

                return tf.cond(step < warmup_steps, warmup, decay)

            def get_config(self):
                return {
                    "initial_lr": self.initial_lr,
                    "decay_steps": self.decay_steps,
                    "warmup_steps": self.warmup_steps,
                    "warmup_lr": self.warmup_lr,
                    "name": self.name,
                }

        import math
        steps_per_epoch = math.ceil(train_data_length / self.batchsize)
        total_steps = self.epochs * steps_per_epoch

        if self.warmup_fraction is not None:
            warmup_steps = max(1, int(total_steps * self.warmup_fraction))
        else:
            warmup_steps = int(self.warmup_epochs * steps_per_epoch)

        logger.info(
            f"schedule: {steps_per_epoch:,} steps/epoch, total={total_steps:,}, "
            f"warmup={warmup_steps:,} ({warmup_steps / total_steps:.2%})",
        )

        lr_schedule = WarmupCosineDecay(
            initial_lr=self.initial_lr,
            decay_steps=total_steps,
            warmup_steps=warmup_steps,
            warmup_lr=self.warmup_lr,
        )

        kwargs = {"learning_rate": lr_schedule}
        if self.lamb_weight_decay is not None:
            kwargs["weight_decay"] = self.lamb_weight_decay
        return tf.keras.optimizers.Lamb(**kwargs)

    def compile_stage_one(self, model, optimizer=None):
        import tensorflow as tf
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            # loss=custom_focal_loss,
            metrics=[
                "accuracy",
                "categorical_crossentropy",
                tf.keras.metrics.AUC(
                    name="auc_higgs", multi_label=True, num_labels=len(self.jet_classes),
                    label_weights=[1.0 if c ==
                                   "higgs" else 0.0 for c in self.jet_classes],
                    num_thresholds=50,
                ),
            ],
        )
        return model

    def prepare_ml_model(self, task: Any):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.seed + task.fold)
        model = self.make_model_gnn(
            input_shape=(self.n_features,),
            index_node=self.stage_one_index_node,
            index_neigh1=self.stage_one_index_neigh1,
            index_neigh2=self.stage_one_index_neigh2,
        )
        return model

    def _downsample_items(self, fracs):

        if isinstance(fracs, dict):
            out = []
            for cls, frac in fracs.items():
                if cls not in self.jet_classes:
                    raise ValueError(
                        f"unbekannte Jet-Klasse '{cls}' in _jet_downsample; "
                        f"bekannt: {self.jet_classes}",
                    )
                out.append((self.jet_classes.index(cls), frac))
            return out
        if len(fracs) != len(self.jet_classes):
            raise ValueError(
                f"_jet_downsample-Eintrag hat {len(fracs)} Werte, "
                f"erwartet {len(self.jet_classes)}",
            )
        return list(enumerate(fracs))

    def _downsample_jets(self, sample, label, btag, process_names, padded, event_weights, tag="", fold: int = 0):
        if not self._jet_downsample:
            return sample, label, btag, process_names, padded, event_weights

        class_ids = label.argmax(axis=1)
        rng = np.random.default_rng(self.seed + fold)
        u = rng.random(len(label))
        keep = np.ones(len(label), dtype=bool)

        for proc, fracs in self._jet_downsample.items():
            for cls_idx, frac in self._downsample_items(fracs):
                if frac >= 1.0:
                    continue
                sel = (process_names == proc) & (class_ids == cls_idx)
                n_before = int(sel.sum())
                if n_before == 0:
                    continue
                keep &= ~(sel & (u >= frac))

        return (sample[keep], label[keep], btag[keep],
                process_names[keep], padded[keep], event_weights[keep])

    def _normalize_per_process(self, w_phys, process_names):
        w = np.asarray(w_phys, dtype=np.float64).copy()
        for proc in np.unique(process_names):
            sel = process_names == proc
            s = w[sel].sum()
            if s <= 0:
                logger.warning(f"{proc}: Gewichtssumme {s:.3g} -- setze auf 1")
                w[sel] = 1.0
                continue
            w[sel] *= sel.sum() / s
            logger.info(
                f"normalize {proc}: mean {s / sel.sum():.4f} -> 1.000  (n={sel.sum():,})")
        return w.astype(np.float32)

    def fit_ml_model(
        self,
        task: Any,
        model,
        train: DotDict,
        validation: DotDict,
        output,
    ) -> None:
        import tensorflow as tf
        from sklearn.preprocessing import RobustScaler, QuantileTransformer, MinMaxScaler
        import pandas as pd
        import gc

        df_train = pd.concat(
            [self._proc_data_to_dataframe(proc_data, proc_inst.name)
             for proc_inst, proc_data in train.items()],
            ignore_index=True,
        )
        df_val = pd.concat(
            [self._proc_data_to_dataframe(proc_data, proc_inst.name)
             for proc_inst, proc_data in validation.items()],
            ignore_index=True,
        )
        logger.info(
            f"fold {task.fold}: {len(df_train)} train / {len(df_val)} val events "
            f"({dict(df_train['process'].value_counts())})",
        )
        for proc_data in list(train.values()) + list(validation.values()):
            proc_data.cleanup()

        gc.collect()

        train_sample, train_label, train_btag_weights, train_process, train_padded, train_event_weights = \
            self.prepare_stage_one_graph_tensors(
                df_train, max_njets=self.n_jets)

        train_sample, train_label, train_btag_weights, train_process, train_padded, train_event_weights = \
            self._downsample_jets(
                train_sample, train_label, train_btag_weights,
                train_process, train_padded, train_event_weights,
                tag="train", fold=task.fold,
            )

        val_sample, val_label, val_btag_weights, val_process, val_padded, val_event_weights = \
            self.prepare_stage_one_graph_tensors(df_val, max_njets=self.n_jets)

        assert train_sample.shape[1] == self.n_features, (
            f"graph width {train_sample.shape[1]} != n_features {self.n_features}"
        )

        self._log_class_composition(
            train_label, train_process, train_event_weights, tag=f"fold{task.fold} train")
        self._log_class_composition(
            val_label, val_process, val_event_weights, tag=f"fold{task.fold} val")

        del df_train, df_val
        gc.collect()

        robust_scaler = RobustScaler()
        quantile_scaler = QuantileTransformer(
            random_state=self.seed + task.fold)
        minmax_scaler = MinMaxScaler()

        x_train = minmax_scaler.fit_transform(
            quantile_scaler.fit_transform(
                robust_scaler.fit_transform(train_sample),
            ),
        )
        x_val = minmax_scaler.transform(
            quantile_scaler.transform(
                robust_scaler.transform(val_sample),
            ),
        )

        def _as_f32(name, arr):
            a = np.asarray(arr)
            if a.dtype != np.float32:
                logger.info(f"{name}: casting {a.dtype} -> float32")
                a = a.astype(np.float32)
            return a

        x_train = _as_f32("x_train", x_train)
        x_val = _as_f32("x_val", x_val)
        del train_sample, val_sample
        gc.collect()
        train_label = _as_f32("train_label", train_label)
        val_label = _as_f32("val_label", val_label)
        train_btag_weights = _as_f32("train_btag_weights", train_btag_weights)
        val_btag_weights = _as_f32("val_btag_weights", val_btag_weights)

        for name, ev, bt in (
            ("train", train_event_weights, train_btag_weights),
            ("val", val_event_weights, val_btag_weights),
        ):
            raw = ev.astype(np.float64) * bt
            n_neg = int((raw < 0).sum())
            logger.info(
                f"{name} weights: n={len(raw):,}, negativ={n_neg:,} ({n_neg / len(raw):.3%}), "
                f"sum={raw.sum():.4g}, sum|w|={np.abs(raw).sum():.4g}, "
                f"min={raw.min():.4g}, max={raw.max():.4g}",
            )
            logger.info(
                f"{name} davon: event_weight<0 -> {int((ev < 0).sum()):,}, "
                f"btag_weight<0 -> {int((bt < 0).sum()):,}",
            )

        if self.use_physics_weights:
            w_train = np.clip(train_event_weights *
                              train_btag_weights, 0.0, None)
            w_val = np.clip(val_event_weights * val_btag_weights, 0.0, None)
        else:
            w_train = np.clip(train_btag_weights, 0.0, None)
            w_val = np.clip(val_btag_weights, 0.0, None)

        w_train = self._normalize_per_process(w_train, train_process)
        w_val = self._normalize_per_process(w_val, val_process)

        ds_weights = self.compute_dataset_weights(train_process)
        w_train = w_train * \
            self._dataset_weight_array(train_process, ds_weights)
        w_val = w_val * self._dataset_weight_array(val_process, ds_weights)

        class_weight_vec = self.compute_class_weights(
            train_label, weights=w_train)
        train_weights = w_train * class_weight_vec[train_label.argmax(axis=1)]
        val_weights = w_val * class_weight_vec[val_label.argmax(axis=1)]

        train_weights = self._finalize_weights(
            train_weights, tag="train", label=train_label)
        val_weights = self._finalize_weights(
            val_weights, tag="val", label=val_label)

        optimizer = self.make_optimizer(train_data_length=len(x_train))
        model = self.compile_stage_one(model, optimizer)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=output["checkpoint"].abspath,
                monitor=self.monitor_metric, mode=self.monitor_mode, save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=self.monitor_metric, mode=self.monitor_mode,
                min_delta=self.early_stopping_min_delta,
                patience=self.early_stopping_patience,
                start_from_epoch=self.early_stopping_start_epoch,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(
                output["mlmodel"].child(
                    f"history_fold{task.fold}.csv", type="f").abspath,
                append=False,
            ),
        ]

        output["mlmodel"].child("robust_scaler.pkl", type="f").dump(
            robust_scaler, formatter="pickle")
        output["mlmodel"].child("quantile_scaler.pkl", type="f").dump(
            quantile_scaler, formatter="pickle")
        output["mlmodel"].child("minmax_scaler.pkl", type="f").dump(
            minmax_scaler, formatter="pickle")

        output["mlmodel"].child(f"gatja_config_fold{task.fold}.yaml", type="f").dump({
            "jet_classes": list(self.jet_classes),
            "n_jets": self.n_jets,
            "n_features": self.n_features,
            "padding_value": self.padding_value,
            "stage_one_index_node": self.stage_one_index_node,
            "stage_one_index_neigh1": self.stage_one_index_neigh1,
            "stage_one_index_neigh2": self.stage_one_index_neigh2,
            "processes": list(self.processes),
            "downsample": dict(self._downsample),
            "jet_downsample": {
                proc: (dict(fracs) if isinstance(fracs, dict) else list(fracs))
                for proc, fracs in self._jet_downsample.items()
            },
            "train_val_test_split": list(self.train_val_test_split),
            "folds": self.folds,
            "fold": task.fold,
            "n_train_jets": int(len(x_train)),
            "n_val_jets": int(len(x_val)),
            "use_physics_weights": self.use_physics_weights,
            "class_weight_power": self.class_weight_power,
            "negative_weights": self.negative_weights,
            "class_weights": {c: float(w) for c, w in zip(self.jet_classes, class_weight_vec)},
            "initial_lr": self.initial_lr,
            "warmup_epochs": self.warmup_epochs,
            "warmup_lr": self.warmup_lr,
            "epochs": self.epochs,
            "batchsize": self.batchsize,
            "seed": self.seed,
            "dataset_weight_power": self.dataset_weight_power,
            "weight_clip_percentile": self.weight_clip_percentile,
            "dataset_weights": {str(p): float(w) for p, w in ds_weights.items()},
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "early_stopping_patience": self.early_stopping_patience,
            "warmup_fraction": self.warmup_fraction,
            "early_stopping_start_epoch": self.early_stopping_start_epoch,
            "lamb_weight_decay": self.lamb_weight_decay,
            "steps_per_epoch": int(np.ceil(len(x_train) / self.batchsize)),
            "total_steps": int(np.ceil(len(x_train) / self.batchsize) * self.epochs),
        }, formatter="yaml")

        history = model.fit(
            x=x_train,
            y=train_label,
            sample_weight=train_weights,
            validation_data=(x_val, val_label, val_weights),
            epochs=self.epochs,
            batch_size=self.batchsize,
            callbacks=callbacks,
            verbose=2,
        )

        output["mlmodel"].child("history.pkl", type="f").dump(
            history.history, formatter="pickle")

        del x_train, x_val
        gc.collect()

    def open_model(self, target) -> dict[str, Any]:
        import tensorflow as tf

        models = {}

        models["input_features"] = tuple(
            target["mlmodel"].child(
                "input_features.pkl", type="f").load(formatter="pickle"),
        )

        with open(target["mlmodel"].child("parameters.yaml", type="f").fn) as f:
            models["parameters"] = yaml.load(f.read(), Loader=yaml.Loader)

        models["model"] = tf.keras.models.load_model(
            target["mlmodel_file"].abspath, compile=False,
        )
        models["best_model"] = tf.keras.models.load_model(
            target["checkpoint"].abspath, compile=False,
        )

        for scaler in ("robust_scaler", "quantile_scaler", "minmax_scaler"):
            models[scaler] = target["mlmodel"].child(
                f"{scaler}.pkl", type="f").load(formatter="pickle")

        return models

    def evaluate(
        self,
        task,
        events: ak.Array,
        models: list,
        fold_indices: ak.Array,
        events_used_in_training: bool = True,
    ) -> ak.Array:

        import pandas as pd

        use_best_model = True
        n_classes = len(self.jet_classes)
        n_out = self.n_jets * n_classes

        for model in models:
            input_features_sanity_checks(self, model["input_features"])

        nets = [model["best_model" if use_best_model else "model"]
                for model in models]
        scalers = [
            (model["robust_scaler"],
             model["quantile_scaler"], model["minmax_scaler"])
            for model in models
        ]

        # __import__("IPython").embed()
        reduced_features = {
            f for f in self.input_features if "Matched" not in f}

        df_all = pd.DataFrame({
            col: np.asarray(ak.to_numpy(events[col]))
            for col in sorted(reduced_features)
        })

        evt_pos = np.arange(len(events), dtype=np.int64)
        fold_np = np.asarray(ak.to_numpy(fold_indices), dtype=np.int64)

        allowed = df_all["jetNumber"].to_numpy() >= 3
        df_f = df_all.loc[allowed].reset_index(drop=True)
        pos_f = evt_pos[allowed]
        folds_f = fold_np[allowed]

        outputs = np.full((len(events), n_out), -10.0, dtype=np.float32)

        for jet_idx in range(self.n_jets):

            keep = df_f[f"jetPT{jet_idx + 1}"].to_numpy() != self.padding_value
            if not np.any(keep):
                continue
            df_kept = df_f.loc[keep].reset_index(drop=True)
            pos_kept = pos_f[keep]
            folds_kept = folds_f[keep]

            sample_block, _, event_weights = self.create_graphs(
                df_kept, jet_idx, drop_empty=False)
            assert len(sample_block) == len(df_kept), (
                f"create_graphs changed row count for jet {jet_idx}: "
                f"{len(sample_block)} != {len(df_kept)}"
            )
            x_raw = sample_block[:, 1:]

            for fold in range(self.folds):
                sel = folds_kept == fold
                if not np.any(sel):
                    continue
                # __import__("IPython").embed()
                robust_scaler, quantile_scaler, minmax_scaler = scalers[fold]
                x = minmax_scaler.transform(
                    quantile_scaler.transform(
                        robust_scaler.transform(x_raw[sel]),
                    ),
                )
                pred = nets[fold].predict(
                    x, batch_size=self.batchsize, verbose=0)
                outputs[pos_kept[sel], jet_idx *
                        n_classes:(jet_idx + 1) * n_classes] = pred

        for i in range(n_out):
            events = set_ak_column(
                events, f"gatja_output_{i}", np.ascontiguousarray(
                    outputs[:, i]),
            )

        return events


gatja_baseline_test = Training_GATJA_Base.derive(
    "Test_Gatja_Training_Base_cw00_ds00",
    cls_dict={
        "weight_clip_percentile": None,
        "store_name": "test",
        "_default__epochs": 1,
        "warmup_epochs": 0,
    },
)


gatja_ds05_75_300 = Training_GATJA_Base.derive(
    "Gatja_Training_Base_cw00_ds05_tt020_ep500_pat75_monLoss",
    cls_dict={
        "store_name": "Baseline",
        "_downsample": {"tt_custom": 0.2},
        "dataset_weight_power": 0.5,
        "monitor_metric": "val_loss",
        "monitor_mode": "min",
        "early_stopping_min_delta": 0.0,
        "early_stopping_patience": 75,
        "_default__epochs": 300,
    },
)

gatja_ds05_lr5e5 = Training_GATJA_Base.derive(
    "Gatja_Training_Base_cw05_ds00_tt020_wu05_pat75_ep500_lr5e5",
    cls_dict={
        "store_name": "Baseline",
        "_downsample": {"tt_custom": 0.2},
        "class_weight_power": 0.5,
        "monitor_metric": "val_loss",
        "monitor_mode": "min",
        "early_stopping_min_delta": 0.0,
        "early_stopping_patience": 75,
        "early_stopping_start_epoch": 25,
        "warmup_fraction": 0.05,
        "initial_lr": 5e-5,
        "_default__epochs": 500,
    },
)


gatja_ds05_lr5e5 = Training_GATJA_Base.derive(
    "Gatja_Training_Base_cw05_ds00_tt020_wu05_pat75_ep500_lr5e5_more_features",
    cls_dict={
        "store_name": "Baseline_more_features",
        "_downsample": {"tt_custom": 0.2},
        "class_weight_power": 0.5,
        "monitor_metric": "val_loss",
        "monitor_mode": "min",
        "early_stopping_min_delta": 0.0,
        "early_stopping_patience": 75,
        "early_stopping_start_epoch": 25,
        "warmup_fraction": 0.05,
        "initial_lr": 5e-5,
        "_default__epochs": 500,
        # jetAverageMass is already in the base graph_feature_names, so it is not repeated here
        # "graph_feature_names": (*Training_GATJA_Base.graph_feature_names,
        #     'bjetAverageMass', 'bjetAverageMassSqr', 'bjetHT', 'lightjetHT',
        #     'bjetNumber', 'minDeltaRbb',
        # ),
        "input_features": (
            *Training_GATJA_Base.input_features,
            "bjetAverageMass", "jetAverageMass", "bjetAverageMassSqr", "bjetHT",
            "lightjetHT", "bjetNumber", "minDeltaRbb",
        ),
        # "rest_cols": (*Training_GATJA_Base.rest_cols,
        #     'bjetAverageMass', 'jetAverageMass', 'bjetAverageMassSqr', 'bjetHT',
        #     'lightjetHT', 'bjetNumber', 'minDeltaRbb',
        # ),
    },
)


# additional event-level features used by the 'more_features' trainings
_extra_event_features = (
    "bjetAverageMass", "bjetAverageMassSqr", "bjetHT", "lightjetHT", "bjetNumber",
    "averageDeltaEtabb", "minDeltaRjj", "minDeltaRbb", "maxDeltaEtabb", "maxDeltaEtajj",
    "maxDeltaEtabj", "minDeltaRbj", "averageDeltaEtabj", "averageDeltaRbj", "minDeltaRMassjj",
    "minDeltaRMassbb", "minDeltaRMassbj", "minDeltaRpTjj", "minDeltaRpTbb", "minDeltaRpTbj",
    "maxPTmassjjj", "maxPTmassjbb",
)


Corrected_Gatja_Training_Base_cw75_ds00_tt100_wu05_pat100_ep500_lr1e5_more_features = Training_GATJA_Base.derive(
    "Corrected_Gatja_Training_Base_cw75_ds00_tt100_wu05_pat100_ep500_lr1e5_more_features",
    cls_dict={
        # Corrected, because the prevoius had doubbled jetAverageMass
        "store_name": "Corrected_Baseline_more_features_corrected_fulltt",
        "_downsample": {"tt_custom": 1.0},
        "class_weight_power": 0.75,
        "monitor_metric": "val_loss",
        "monitor_mode": "min",
        "early_stopping_min_delta": 0.0,
        "early_stopping_patience": 100,
        "early_stopping_start_epoch": 25,
        "warmup_fraction": 0.05,
        "initial_lr": 1e-5,
        "_default__epochs": 500,
        "graph_feature_names": (*Training_GATJA_Base.graph_feature_names, *_extra_event_features),
        "input_features": (*Training_GATJA_Base.input_features, *_extra_event_features),
        "rest_cols": (*Training_GATJA_Base.rest_cols, *_extra_event_features),
        "warm_start_dir": (
            "/data/dust/user/weidnerb/"
            "gatja_resume_Corrected_Gatja_Training_Base_cw75_ds00_tt100_wu05_pat100_ep500_lr1e5_more_features_"
        ),

    },
)
