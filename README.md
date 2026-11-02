# HH → bbWW

Analysis based on [columnflow](https://github.com/uhh-cms/columnflow), [law](https://github.com/riga/law) and [order](https://github.com/riga/order).

New to the analysis? Read this file for the setup, then the README of each folder, roughly in the order the data flows through the analysis:

| README | topic |
| --- | --- |
| [hbw/config](hbw/config/README.md) | analyses, configs and campaigns, datasets, processes, categories, variables, defaults and groups, where outputs land |
| [hbw/selection](hbw/selection/README.md) | calibration, event selection (`dl1`, `sl1`) and reduction (`cf.CalibrateEvents` to `cf.ReduceEvents`) |
| [hbw/production](hbw/production/README.md) | producers, event weights, categorization and hist producers (`cf.ProduceColumns` to histograms) |
| [hbw/ml](hbw/ml/README.md) | DNN classifiers: training, evaluation, deriving new models |
| [hbw/inference](hbw/inference/README.md) | inference models, datacards and the statistical-inference helpers |
| [hbw/tasks](hbw/tasks/README.md) | law usage tips, the `hbw.` tasks, helper scripts, plot functions, linting and tests |


## Quickstart

A couple test tasks are listed below.
They might require a **valid voms proxy** for accessing input data.

```shell
# clone the project
git clone --recursive git@github.com:uhh-cms/hh2bbww.git
cd hh2bbww

# source the setup and store decisions in .setups/dev.sh (arbitrary name)
source setup.sh dev

# index existing tasks once to enable auto-completion for "law run"
law index --verbose

# build the campaign summary once per config
# (configs are created lazily from this output, so every other task fails without it;
#  building the full config "c24v15" also unlocks its limited twin "l24v15")
law run hbw.BuildCampaignSummary \
    --config c24v15 \
    --remove-output 0,a,y

# run your first task
# (they are all shipped with columnflow and thus have the "cf." prefix)
# "l24v15" is the limited version of "c24v15" and only reads 2 files per dataset
law run cf.ReduceEvents \
    --version v1 \
    --config l24v15 \
    --dataset tt_dl_powheg \
    --branch 0

# create some plots
law run cf.PlotVariables1D \
    --version v1 \
    --config l24v15 \
    --datasets tt_dl_powheg \
    --variables jet0_pt,jet1_pt \
    --categories incl,2mu
```

Which defaults are used when `--config`, `--analysis` or `--dataset` are omitted is decided by
the law config file selected during `source setup.sh`, see [Law config](#law-config) below.
Datacards are created with `cf.cms.CreateDatacards`, e.g. with the 2024/2025 inference model `full_2425`
(`--configs c24v15,c25v15 --inference-model full_2425`); see the [inference README](hbw/inference/README.md).

## Most important files
(Note: please tell me, if a link or task does not work. I did not test all of them)


### Config
Files relevant for the configuration of an analysis are mainly to be found in [this](hbw/config) folder. The main analysis object is defined [here](hbw/analysis/create_analysis.py). The analysis object contains multiple configs (at least one config per campaign). Most of the configuration takes place [here](hbw/config/config_run2.py) and defines meta-data like the used datasets, processes, shifts, categories, variables and much more.
Configs are registered via `add_lazy_config` in [create_analysis.py](hbw/analysis/create_analysis.py). Each call adds *two* configs: the full one and a limited one (2 files per dataset, for testing) whose name starts with `l` instead of `c`:

| full | limited | campaign |
| --- | --- | --- |
| `c17v9` | `l17v9` | Run 2, 2017, nano v9 |
| `c22prev14` | `l22prev14` | Run 3, 2022 preEE, nano v14 |
| `c22postv14` | `l22postv14` | Run 3, 2022 postEE, nano v14 |
| `c23prev14` | `l23prev14` | Run 3, 2023 preBPix, nano v14 |
| `c23postv14` | `l23postv14` | Run 3, 2023 postBPix, nano v14 |
| `c24v15` | `l24v15` | Run 3, 2024, nano v15 |
| `c25v15` | `l25v15` | Run 3, 2025, nano v15 |
| `c26v15` | `l26v15` | Run 3, 2026, nano v15 |

The configs are currently designed for **2024 and 2025** (`c24v15`, `c25v15`).
The 2022/2023 configs are not maintained, 2026 (`c26v15`) is not yet fully configured, and the
Run 2 config `c17v9` is legacy; see the [config README](hbw/config/README.md) for details.

Most tasks can use a `--config` parameter as an input, e.g.
```
law run cf.SelectEvents --version v1 --config c24v15
```


### Selectors
Calibrators, selectors and reducers live in [hbw/calibration](hbw/calibration), [hbw/selection](hbw/selection) and [hbw/reduction](hbw/reduction).
The main selectors are `dl1` ([dl_remastered.py](hbw/selection/dl_remastered.py)) and `sl1` ([sl_remastered.py](hbw/selection/sl_remastered.py)); variants are derived from them with `.derive(...)`.
```
law run cf.SelectEvents --version v1 --config l24v15 --dataset tt_dl_powheg --selector dl1 --branch 0
```
See the [selection README](hbw/selection/README.md) for the selection steps, variants and cutflow plots.


### Producers
Modules that produce new columns (weights, ML inputs, categories, ...) are called `producers` and live in [hbw/production](hbw/production).
Each `cf.ProduceColumns` task runs one producer; downstream tasks combine several via `--producers`:
```
law run cf.ProduceColumns --version v1 --config l24v15 --dataset tt_dl_powheg --producer event_weights --branch 0
```
See the [production README](hbw/production/README.md) for the default producers, how event weights and categories are built and how hist producers ([hbw/weight](hbw/weight)) pick the histogram weights.


### Machine learning
ML models are defined in [hbw/ml](hbw/ml). The base class is `MLClassifierBase` ([base.py](hbw/ml/base.py)); the models in use are derived from `DenseClassifierDL` / `DenseClassifierSL` in [hbw/ml/derived](hbw/ml/derived) via `.derive("name", cls_dict={...})`.
A model declares which configs, selector and producers it needs, so you don't pass them on the command line:
```
law run cf.MLTraining --version v1 --ml-model <model name> --configs c24v15 --branch 0
```
See the [ML README](hbw/ml/README.md) for the training workflow, the current models and how to derive new ones.


### Inference
Inference models (categories, processes and systematics of the datacards) are defined in [hbw/inference](hbw/inference) and derived from `HBWInferenceModelBase` ([base.py](hbw/inference/base.py)).
Datacards are created with
```
law run cf.cms.CreateDatacards --version v1 --configs c24v15 --inference-model <model name>
```
The inference model already defines which ML model is required, and the producers are resolved from it, so don't add `--producers` or `--ml-models` to this task.
See the [inference README](hbw/inference/README.md) for the available models, rebinning and the statistical-inference commands.


### Tasks
Analysis-specific tasks (namespace `hbw.`) are defined in [hbw/tasks](hbw/tasks).
See the [tasks README](hbw/tasks/README.md) for an overview of all tasks, everyday law parameters and the helper scripts in [hbw/scripts](hbw/scripts).


### Columnflow
The analysis uses many functionalities of [columnflow](https://github.com/uhh-cms/columnflow).
We rely on:
- [tasks](https://github.com/uhh-cms/columnflow/tree/master/columnflow/tasks) defined from columnflow.
-  columnflow modules for [calibration](https://github.com/uhh-cms/columnflow/tree/master/columnflow/calibration), [selection](https://github.com/uhh-cms/columnflow/tree/master/columnflow/selection) and [production](https://github.com/uhh-cms/columnflow/tree/master/columnflow/production)
- convenience functions defined in columnflow ([util](https://github.com/uhh-cms/columnflow/blob/master/columnflow/util.py), [columnar_util](https://github.com/uhh-cms/columnflow/blob/master/columnflow/columnar_util.py), [config_util](https://github.com/uhh-cms/columnflow/blob/master/columnflow/config_util.py))


### Law config
The law config is located [here](law.cfg) and takes care of information available to law when calling tasks.
In here, we can for example:
- set some defaults for parameters (e.g. when not setting the `--dataset` parameter in a task that needs this parameter, we use the `default_dataset` parameter instead)
- define, in which file system to store outputs for each task
- define, which tasks should be loaded for the analysis

## Development

- Source hosted at [GitHub](https://github.com/uhh-cms/hh2bbww)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/uhh-cms/hh2bbww/issues)
