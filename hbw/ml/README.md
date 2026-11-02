# Machine learning

This folder contains the DNN classifiers of the analysis. They are columnflow `MLModel`s, so they are
not run by hand but are trained and evaluated through law tasks (`cf.MLTraining`, `cf.MLEvaluation`, ...).
A model is a Python class that declares its inputs, processes, training nodes and hyperparameters as class
attributes. Variants are made with `.derive("name", cls_dict={...})`, and the name you pass to `--ml-model`
is the name given to `derive`.

For the columnflow side, see the [columnflow ML guide](../../modules/columnflow/docs/user_guide/ml.md).


## Workflow

Each of these tasks requires the ones before it, so running the last task you need is enough:

```
ProduceColumns (event_weights + ML input producer)
  -> cf.PrepareMLEvents         per dataset and file: applies the preparation producer, assigns folds
  -> cf.MergeMLStats            per dataset: event counts and weight sums per process and fold
  -> cf.SimpleMergeMLEvents     per dataset: one parquet file per fold        (hbw/tasks/ml.py)
  -> hbw.MLPreTraining          per (fold, process): numpy arrays split into train/val/test
  -> cf.MLTraining              one branch per fold
  -> cf.MLEvaluation            per dataset: writes the mlscore.* columns
  -> ProduceColumns (cats_ml_<model>) -> CreateHistograms -> ... -> CreateDatacards
```

`cf.MLTraining` also requires columnflow's own `cf.MergeMLEvents` per fold. The hbw models don't read that
output, though. Their `requires()` adds `hbw.MLPreTraining`, and `load_data` reads its arrays.

**Folds.** `cf.PrepareMLEvents` assigns each event `fold_indices = deterministic_seed % folds`, with
`folds = 5` for all current models. Branch `f` of `cf.MLTraining` trains on events whose fold index is not
`f`. Within those events, `train_val_test_split = (0.75, 0.15, 0.10)` decides which go into training and
validation; the test part is held out. `cf.MLEvaluation` then applies model `f` only to events with fold
index `f`. Every MC event is therefore scored by a model that never saw it. All folds must be trained
before `cf.MLEvaluation` can run.

**Preparation producer.** The `prepml` producer in [stats.py](stats.py) runs inside `cf.PrepareMLEvents`.
It keeps only the training region: `catid_sr` (SL), `catid_mll_low_narrow` (DL HH) or `catid_hhh_sr`
(DL HHH). Derived variants such as `prepml_lep2pt15`, `prepml_met40` or `prepml_eq2b` apply one extra
categorizer on top (see the `derive` calls at the end of [stats.py](stats.py)). `prepml` also stores the
full event weight as `event_weight`, computed with the default hist producer, and fills the stats that
`cf.MergeMLStats` merges. A model picks its producer with `preparation_producer_name`.


## What `MLClassifierBase` adds

[base.py](base.py) builds on columnflow's `MLModel` and implements everything except the network itself:

- **Requirements.** `training_producers` always returns `["event_weights", <ml_inputs_producer>]`,
  whatever is passed on the command line. The input producer is `config_inst.x.ml_inputs_producer`:
  `sl_ml_inputs`, `dl_ml_inputs` or `hhh_dl_ml_inputs`, chosen by the analysis tags in
  [defaults_and_groups.py](../config/defaults_and_groups.py). `training_calibrators` falls back to the
  default calibrators. The selector isn't overridden, so the default selector is used (`sl1` / `dl1`).
- **Configs.** A model may define `training_configs` (a function returning config names). If it does, that
  list replaces whatever you pass to `--configs`.
- **Processes and datasets.** `datasets()` maps every entry of `processes` to its datasets. Each dataset
  may belong to only one process.
- **Training nodes.** `train_nodes` defines the output nodes. Each node has an `ml_id` (0..N-1), an
  optional `label`/`color`, a `class_factor_mode` (`"equal"` or `"xsec"`) and optional `sub_processes`.
  `setup()` calls `prepare_ml_processes` in [processes.py](../config/processes.py). It creates a parent
  process for every node name that isn't already a process (e.g. `sig_ggf_binary`).
- **Variables.** `setup()` adds config variables for each node: `mlscore.<node>`, `logit_mlscore.<node>`,
  `rebinlogit_mlscore.<node>` and `rebinned_logit{2,3,4}b.<node>`.
- **Inputs.** All `input_features` must start with `mli_`. The column order used in training is stored
  in `input_features.pkl` and checked again at evaluation time.
- **Sandbox.** Training, evaluation and the single-fold tasks run in
  [venv_ml_plotting.sh](../../sandboxes/venv_ml_plotting.sh) (TensorFlow 2.16.1, scikit-learn, shap; see
  [ml_plotting.txt](../../sandboxes/ml_plotting.txt)). With `CF_DEV` set, the `_dev` twin of the sandbox
  is used. [venv_onnx.sh](../../sandboxes/venv_onnx.sh) is not used by these models; the `transformer`
  and `nlo_reweighting` producers use it.
- **Bookkeeping.** All attributes listed in `bookkeep_params` are hashed into the output path
  (`ml__<model>__<hash>`) and written to `parameters.yaml`.
  - The `hbw.MLPreTraining` outputs are keyed separately, by `preparation_producer_name` plus a hash of
    `data_loader`, `input_features` and `train_val_test_split`.
  - So models that differ only in hyperparameters or weights share the same pre-training arrays.

The network and the training loop come from [mixins.py](mixins.py):

- `DenseModelMixin` builds a dense network: a `BatchNormalization` input layer, hidden layers set by
  `layers`, `activation` and `dropout`, a softmax output, and the Adam optimizer.
- `ModelFitMixin` trains with the `MultiDataset` from [tf_util.py](tf_util.py). Every batch holds a fixed
  number of events per process. That number comes from `class_factors` (between nodes) and
  `sub_process_class_factors` (inside a node). `ModelFitMixin` also adds the Keras callbacks (`backup`,
  `checkpoint`, `reduce_lr`, `early_stopping`) and writes history plots.

`DenseClassifierDL` ([derived/dl.py](derived/dl.py)) and `DenseClassifierSL`
([derived/sl.py](derived/sl.py)) combine `DenseModelMixin`, `ModelFitMixin` and `MLClassifierBase`.
Every model in use is derived from one of them.


## Overriding settings from the CLI

Columnflow's `--ml-model-settings` parameter passes a dict to the model's `__init__`. Only attributes
listed in `settings_parameters` are accepted. For `DenseClassifierDL`/`SL` these are:

- `processes`, `class_factors`, `sub_process_class_factors`
- `activation`, `layers`, `dropout`, `learningrate`
- `callbacks`, `reduce_lr_factor`, `reduce_lr_patience`
- `epochs`, `batchsize`

To make this work, the class stores each of these as `_default__<attr>`. `MLClassifierBase.derive()` does
the renaming for you, so in `cls_dict` you just write `"epochs": 50`. At init, the CLI value wins and
the `_default__` value is the fallback. Settings are comma-separated, and tuple values use `;`:

```shell
--ml-model-settings "epochs=20,learningrate=0.001,layers=256;256;256"
```

The settings change the parameter hash, so the outputs go to a new directory. That makes them handy for
quick tests. If a variant will be used downstream (histograms, datacards), derive a named model instead.


## Models

`ml_modules` in [law.cfg](../../law.cfg) lists the modules that get registered: `hbw.ml.base` and
`hbw.ml.derived.{sl,dl,ml_dl_dih,ml_dl_trih}`. A model in any other file can't be found by name.

**DL HH (dih, [derived/ml_dl_dih.py](derived/ml_dl_dih.py)).** Models come in sets of three:

- a multiclass DNN with the nodes `sig_ggf`, `sig_vbf`, `tt`, `st`, `dy_m10toinf` and `h`;
- a ggF-vs-background binary DNN (`sig_ggf_binary`, `bkg_binary_for_ggf`);
- a VBF-vs-background binary DNN (`sig_vbf_binary`, `bkg_binary_for_vbf`).

The default inference model `default_unblind` in [inf_dl_dih.py](../inference/inf_dl_dih.py) uses
`multiclassv3`, `ggfv3` and `vbfv3`, trained on the 2022+2023 configs (`configs.full`). The newest sets are
`{multiclass,ggf,vbf}_2425_v3l2pt15` and `{multiclass,ggf,vbf}_2425_v32l2pt15`. They are used by e.g.
`rate24_251` and `full_2425`, and train on `c24v15` + `c25v15` with `prepml_lep2pt15`.
`multiclass_2425_v32l2pt15` is the default ML model of the training tasks (see below). Reusable input
sets (`v1`, `v2`, `v3_discrete`, ...), process lists and `configs` presets are defined at the top of the
file.

The file also contains DCTR reweighting models (`dctr_*`). Their preparation producers
(`prepml_dyvr_mumu*`) are not defined in [stats.py](stats.py), so they can't run as they are.

**DL HHH (trih, [derived/ml_dl_trih.py](derived/ml_dl_trih.py)).** There are two multiclass models:

- `multiclass_eq2b`: HHH signal vs. ttHH, ttH, tt, st and DY;
- `multiclass_geq3b`: the same without st and DY.

`Cat_eq2b_V1`, `Cat_eq3b_V1` and `Cat_geq4b_V1` are variants restricted to one b-jet category through
`prepml_eq2b`/`prepml_eq3b`/`prepml_geq4b`. None of these define `training_configs`, so they train on
the configs you pass with `--configs`.

**SL ([derived/sl.py](derived/sl.py)).** There are `sl_22post` and `sl_22post_test`. Their
`training_configs` point to `c22post` / `l22post`, which are not registered config names.

**Caveat for this branch.** Some inference models point to ML models that are commented out:

- `rate24`/`full24` in [inf_dl_dih.py](../inference/inf_dl_dih.py) expect the 2024-only
  `multiclass24`/`ggf24`/`vbf24`;
- the trih inference models expect `Bin_V1` ([ml_dl_trih.py](derived/ml_dl_trih.py)).

**Default ML model.** Without `--ml-model`, tasks of the training pipeline (and `cf.MLEvaluation`, via the
analysis-level `default_ml_model` aux set in [create_analysis.py](../analysis/create_analysis.py)) use the
first entry of `default_ml_models` from the law config if it is set, and `multiclass_2425_v32l2pt15`
otherwise (`default_ml_model` in [defaults_and_groups.py](../config/defaults_and_groups.py); SL configs use
`sl_22post`). This also applies to `hbw_hhh_dl`, so pass `--ml-model` there. Tasks that use an inference
model take the ML models from the inference model instead.


## Recipes

The examples assume the DL HH law config (`law.dl.nocert.cfg`, analysis `hbw_dl`). For HHH, add
`--analysis hbw_hhh_dl`. `MLTraining` and the `hbw.*` ML tasks take `--configs` (plural).

```shell
# inspect the pre-training arrays of one (fold, process) branch
law run hbw.MLPreTraining --version v1 --ml-model multiclass_eq2b --analysis hbw_hhh_dl \
    --configs c24v15 --branch 0

# train a single fold locally (branch = fold)
law run cf.MLTraining --version v1 --ml-model multiclass_eq2b --analysis hbw_hhh_dl \
    --configs c24v15 --branch 0

# train all folds on htcondor (upstream tasks too); resources default to [resources] in law.cfg
law run cf.MLTraining --version v1 --ml-model multiclass_eq2b --analysis hbw_hhh_dl \
    --configs c24v15 --workflow htcondor --htcondor-gpus 1 \
    --cf.PrepareMLEvents-workflow htcondor --hbw.MLPreTraining-workflow htcondor \
    --workers 20

# quick hyperparameter test (writes to a separate ml__<model>__<hash> directory)
law run cf.MLTraining --version v1 --ml-model multiclass_eq2b --analysis hbw_hhh_dl \
    --configs c24v15 --branch 0 --ml-model-settings "epochs=10,dropout=0.3"

# ML plots for the model of one fold (confusion matrices, ROCs, output nodes,
# input features, SHAP/sensitivity ranking) + stats.json with the metrics
law run hbw.PlotMLResultsSingleFold --version v1 --ml-model multiclass_eq2b --analysis hbw_hhh_dl \
    --configs c24v15 --fold 0

# apply all folds to one dataset (adds mlscore.<node> columns)
law run cf.MLEvaluation --version v1 --ml-model multiclass_eq2b --analysis hbw_hhh_dl \
    --config c24v15 --dataset tt_dl_powheg --branch 0

# where did it go? (works for every task)
law run cf.MLTraining ... --print-output 0
```

`hbw.PlotMLResultsSingleFold` needs `hbw.MLEvaluationSingleFold`. That task stores the predictions of
model `--fold` for each process and data split (`--data-split`, default `test`). With
`hbw.PlotMLResultsSingleFoldTest`, only the test split is plotted.

For downstream histograms and plots, pass `--ml-models <model>` (e.g. to `cf.PlotVariables1D` with
`--variables mlscore.<node>`). The default producers then append `cats_ml_<model>` (see
[categories.py](../production/categories.py)), which requires `cf.MLEvaluation` and adds the ML
categories. For datacards, set the models through the inference model's `ml_model_name`.

**Outputs.** Paths are built from `store_parts` (see `hbw_parts` in
[create_analysis.py](../analysis/create_analysis.py)): roughly
`<store>/<analysis>/<calib>/<sel>/<red>/<configs>/<producers>/ml__<model>__<hash>/cf.MLTraining/<version>/`.
Each fold gets its own `mlmodel_f<i>of5/` directory containing:

- `mlmodel.keras`: the final model, which is the one used in evaluation;
- `checkpoint.model.keras`: the best model by `val_loss`;
- `parameters.yaml` and `input_features.pkl`;
- `plots/`: the loss/accuracy history plots.

`--print-output 0` prints the exact location.

### Deriving a new model

Add the derived model to one of the registered files (or register a new file in `ml_modules`). For
example, a 2024-only version of the ggF binary model in [ml_dl_dih.py](derived/ml_dl_dih.py):

```python
ggf24_test = ggfv1.derive("ggf24_test", cls_dict={
    "training_configs": configs.twentyfour,           # -> ["c24v15"]
    "input_features": input_features["v3_discrete"],  # every name must be an mli_* column
    "preparation_producer_name": "prepml_lep2pt15",   # training region, see stats.py
    "layers": (256, 256, 256),                        # settings_parameters can be set directly
    "epochs": 50,
})
```

- Check that every entry in `processes` exists in the `c24v15` config. The commented-out
  `multiclass24`/`ggf24`/`vbf24` blocks show the process lists that were used for 2024.
- A new input feature must be produced by the ML input producer: `common_ml_inputs` / `sl_ml_inputs` in
  [ml_inputs.py](../production/ml_inputs.py), `dl_ml_inputs` in
  [ml_dih_inputs.py](../production/ml_dih_inputs.py) or `hhh_dl_ml_inputs` in
  [ml_trih_inputs.py](../production/ml_trih_inputs.py).
- The new feature also needs a variable definition, in [ml_variables.py](../config/ml_variables.py)
  (common + SL) or [dl/variables.py](../config/dl/variables.py) (DL, HHH).
- Run `law index` after adding models so autocompletion finds them.


## Files

| File | Content |
| --- | --- |
| [base.py](base.py) | `MLClassifierBase`, plus `ExampleDNN` / `example_test` as a minimal example. Importing it loads all `ml_modules`. |
| [mixins.py](mixins.py) | `DenseModelMixin`, `CallbacksBase`, `ModelFitMixin`, `ClassicModelFitMixin` (plain `tf.data` loop) |
| [data_loader.py](data_loader.py) | `MLDatasetLoader` (event array to split numpy arrays, training weights), `MLProcessData` (loads `MLPreTraining` outputs per fold) |
| [tf_util.py](tf_util.py) | `MultiDataset` (per-process batch composition) and the `cumulated_crossentropy` loss |
| [plotting.py](plotting.py) | history, confusion, ROC (one-vs-rest, one-vs-one), output node, input feature and introspection plots |
| [introspection.py](introspection.py) | feature ranking: `sensitivity_analysis`, `gradient_times_input`, `shap_ranking` |
| [helper.py](helper.py) | small helpers (`assign_dataset_to_process`, `predict_numpy_on_batch`) |
| [stats.py](stats.py) | `prepml` preparation producer and its variants (registered via `production_modules`) |
| [derived/dl.py](derived/dl.py), [derived/sl.py](derived/sl.py) | `DenseClassifierDL`, `DenseClassifierSL` base models; `sl.py` also has the SL derived models |
| [derived/ml_dl_dih.py](derived/ml_dl_dih.py) | DL HH models, input feature sets, config presets |
| [derived/ml_dl_trih.py](derived/ml_dl_trih.py) | DL HHH models |
| [derived/legacy_sl.py](derived/legacy_sl.py), [derived/legacy_dl.py](derived/legacy_dl.py) | legacy, **not registered** (old `dense_*` / `dl_22*` models) |
| [derived/grid_search.py](derived/grid_search.py) | legacy, **not registered**; imports the removed `hbw.ml.dense_classifier` |
| [old_ml_model.py](old_ml_model.py) | legacy `SimpleDNN`, **not registered** |
| [../tasks/ml.py](../tasks/ml.py) | `SimpleMergeMLEvents`, `hbw.MLPreTraining`, `hbw.MLEvaluationSingleFold`, `hbw.PlotMLResultsSingleFold(Test)`, `hbw.MLOptimizer` (compares `stats` outputs of several trainings; current models write no such output) |
