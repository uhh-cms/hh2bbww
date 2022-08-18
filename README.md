# Analysis playground

Testing area for full-Python HEP analyses based on [columnflow](https://github.com/uhh-cms/columnflow), [law](https://github.com/riga/law) and [order](https://github.com/riga/order).


### Quickstart

A couple test tasks are listed below.
They might require a **valid voms proxy** for accessing input data.

```shell
# clone the project
git clone --recursive git@github.com:uhh-cms/analysis_playground.git
cd analysis_playground

# source the setup and store decisions in .setups/dev.sh (arbitrary name)
source setup.sh dev

# index existing tasks once to enable auto-completion for "law run"
law index --verbose

# run your first task
# (they are all shipped with columnflow and thus have the "cf." prefix)
law run cf.ReduceEvents \
    --version v1 \
    --dataset st_tchannel_t \
    --branch 0

# create a plot
law run cf.PlotVariables \
    --version v1 \
    --datasets st_tchannel_t \
    --producers features \
    --variables jet1_pt \
    --categories 1e \
    --branch 0

# create a (test) datacard (CMS-style)
law run cf.CreateDatacards \
    --version v1 \
    --producers features \
    --inference-model test \
    --workers 3
```


### Development

- Source hosted at [GitHub](https://github.com/uhh-cms/analysis_playground)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/uhh-cms/analysis_playground/issues)
