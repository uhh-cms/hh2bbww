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
law run cf.PlotVariables \
    --version v1 \
    --datasets st_tchannel_t_powheg \
    --producers features \
    --variables jet1_pt,jet2_pt \
    --categories 1e

# create a (test) datacard (CMS-style)
law run cf.CreateDatacards \
    --version v1 \
    --producers features \
    --inference-model test \
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
At the moment, there is only one class to define simple feed-forward NN's (located [here](hbw/ml/simple.py)). This class can be used to derive ML models of this type with different sets of parameters (which takes place [here](hbw/ml/derived.py)). These derived models can be used as part of law tasks, e.g. with
```
law run cf.MLTraining --version v1 --ml-model default
```

### Inference
Modules to prepare datacards are located in [this](hbw/inference) folder.
At the moment, there is only the 'default' inference model ([here](hbw/inference/default.py)).
Datacard are produced via calling
```
law run cf.CreateDatacards --version v1 --inference-model default --ml-models default
```

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
