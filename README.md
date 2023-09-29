# HH → bbWW

Analysis based on [columnflow](https://github.com/uhh-cms/columnflow), [law](https://github.com/riga/law) and [order](https://github.com/riga/order).


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

# run your first task
# (they are all shipped with columnflow and thus have the "cf." prefix)
law run cf.ReduceEvents \
    --version v1 \
    --dataset st_tchannel_t_powheg \
    --branch 0

# create some plots
law run cf.PlotVariables1D \
    --version v1 \
    --datasets st_tchannel_t_powheg \
    --producers features \
    --variables jet1_pt,jet2_pt \
    --categories 1e

# create a (test) datacard (CMS-style)
law run cf.CreateDatacards \
    --version v1 \
    --inference-model default \
    --workers 3
```

## Most important files
(Note: please tell me, if a link or task does not work. I did not test all of them)


### Config
Files relevant for the configuration of an analysis are mainly to be found in [this](hbw/config) folder. The main analysis object is defined [here](hbw/config/analysis_hbw.py). The analysis object contains multiple configs (at least one config per campaign). Most of the configuration takes place [here](hbw/config/config_run2.py) and defines meta-data like the used datasets, processes, shifts, categories, variables and much more.
At the moment, there are only two configs, the default config `config_2017` and a config with reduced event statistics (for test purposes) named `config_2017_limited`. Most tasks can use a `--config` parameter as an input, e.g.
```
law run cf.SelectEvents --version v1 --config config_2017
```


### Selectors
Modules that are used to define event selections are usually named `selectors` and can be found in [this](hbw/selection) folder.
The main selector is called `default` and can be found [here](hbw/selection/default.py).
You can call the `SelectEvents` task using this selector, e.g. via
```
law run cf.SelectEvents --version v1 --selector default
```


### Producers
Modules that are used to produce new columns are usually named `producers` and can be found in [this](hbw/production) folder.
A producer can be used as a part of another calibrator/selector/producer.
You can also call a producer on it's own as part of the `ProduceColumns` task, e.g. via
```
law run cf.ProduceColumns --version v1 --producer example
```


### Machine learning
Modules for machine learning are located in [this](hbw/ml) folder.
Our base ML Model ist defined [here](hbw/ml/base.py) and parameters are defined as class parameters.
From the base class, new ML Models can be built via class inheritance. Our default ML Model is the [DenseClassifier](hbw/ml/dense_classifier.py), which uses the base model and some [mixins](hbw/ml/mixins.py) for additional functionality and overwrites their class parameters. From each ML Model, new models can be derived with different parameters using the `derive` method:
```
DenseClassifier.derive("dense_default", cls_dict={"folds": 5})
```
All ML Models can be used as part of law tasks, e.g. with
```
law run cf.MLTraining --version v1 --ml-model dense_default --branches 0
```
NOTE: the `DenseClasifier` already defines, which config, selector and producers are required, so you don't need to add them on the command line.


### Inference
Modules to prepare datacards are located in [this](hbw/inference) folder.
At the moment, there is only the 'default' inference model ([here](hbw/inference/default.py)).
Datacard are produced via calling
```
law run cf.CreateDatacards --version v1 --inference-model default
```
Similar to the ML Models, we can also derive additional inference models using the `derive` method.

NOTE: our inference model already defines, which ml_model is required and based on that,  producer requirements are resolved automatically per default. It is therefore better to not add them to the task parameters since the dependencies are already a bit complicated.


### Tasks
Analysis-specific tasks are defined in [this](hbw/tasks) folder.


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
