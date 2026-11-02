# Inference models and datacards

This folder defines the *inference models* of the analysis: which categories, processes and systematics go into the CMS combine datacards.
Datacards are written by the columnflow task `cf.cms.CreateDatacards`; the hbw tasks in [hbw/tasks/inference.py](../tasks/inference.py) rebin them, generate the statistical-inference commands and make plots.


## Concepts

### columnflow's `InferenceModel`
An inference model ([columnflow/inference/__init__.py](../../modules/columnflow/columnflow/inference/__init__.py)) is a python description of a statistical model that is independent of the combine syntax. It holds:

- **categories**: one per datacard (combine "bin"). Each one points, per config, to a config category and a variable whose histogram is used as the fit shape, and states whether `data_obs` comes from real data or is faked from MC.
- **processes**: per category, with the config process (and thereby the datasets) behind it and an `is_signal` flag.
- **parameters**: the nuisances. The types used here are `rate_gauss` (written as `lnN`), `rate_unconstrained` (written as a `rateParam`) and `shape` (needs shifted histograms).

`cf.cms.CreateDatacards` takes this model, finds out which histograms it needs (configs, datasets, variables, categories, shift sources), requires the corresponding `cf.MergeShiftedHistograms` outputs and writes one `datacard__<category>.txt` plus `shapes__<category>.root` per category ([columnflow/tasks/cms/inference.py](../../modules/columnflow/columnflow/tasks/cms/inference.py)).
The task is multi-config, so a single datacard category can combine histograms from several configs (e.g. `--configs c22prev14,c22postv14`).

Note: the task family is `cf.cms.CreateDatacards` in the pinned columnflow version. Older notes or scripts may still use `cf.CreateDatacards`, which no longer resolves.

### `HBWInferenceModelBase`: configuration by class attributes
All our models derive from `HBWInferenceModelBase` in [base.py](base.py). Its `__init__` builds the model from class attributes, so a new model only needs different attribute values, no new code:

| attribute | meaning |
| --- | --- |
| `ml_model_name` | ML model name or list of names. Not used inside the model itself, but read by the config defaults to resolve `--ml-models` and `--producers` (see below). |
| `processes` | config process names. Each one must exist in all configs and have datasets; names are translated to combine names by `inf_proc` (see below). |
| `config_categories` | config category names; each becomes one datacard category with the same name. |
| `config_variable` | function `(self, config_cat_inst) -> variable name` choosing the fit variable per category. |
| `systematics` | list of nuisance names, may contain `{year}`, `{campaign}` and `{bjet_cat}` placeholders (see below). |
| `skip_datasets` | `{process: datasets}` to exclude specific datasets from a process. |
| `mc_stats` | add the `autoMCStats` line. |
| `skip_data` | if `True` (default), `data_obs` is faked from MC; if `False`, real data is used. |
| `unblind` | only matters with `skip_data=False`: `False` means "partially unblinded", which makes the hbw tasks mask observation/rates and blind sensitive bins. |
| `flow_strategy` | under/overflow handling passed to columnflow (default `"move"`; the single-variable models use `"remove"`). |
| `scale_signal` | optional factor applied to all signal processes. |
| `multi_variables` | `config_variable` returns a list of variables; only meant for `hbw.CreateMultipleDatacards`. |
| `bjet_cats`, `campaign_tags` | values used to expand placeholders and to restrict per-campaign nuisances. |
| `version`, `skip_ratify_shapes` | currently bookkeeping only: they don't enter output paths (`ratify_shape_parameters` is not called). |

The overridable hooks are `config_variable`, `cat_name` and `customize_category`.

### How systematics are interpreted
`format_systematics` expands the placeholders (`{year}`: campaign year, `{campaign}`: the config's `cpn_tag`, e.g. `2024` for `c24v15`, `{bjet_cat}`: `1b`, `2b`, `boosted`). Then each name is matched against the tables in [constants.py](constants.py):

- `lumi_*`: must be one of the uncertainties in the config's `luminosity` (e.g. `lumi_13p6TeV_2024`).
- `QCDscale_<key>` / `pdf_<key>`: rate uncertainties taken from the cmsdb cross-section uncertainties of the processes in `processes_per_QCDscale` / `processes_per_pdf_rate`.
- `BR_<decay>`: Higgs branching ratios from `higgs_br_uncertainties`.
- `rate_<key>`: free floating normalisations (`[0,2]`) for the processes in `processes_per_rate_unconstrained`; these processes get no QCDscale/pdf rate nuisance.
- `eff_hbb_*`: 10% lnN in the boosted categories (`hbb_efficiency_params`).
- everything else must be a key of `processes_per_shape` and becomes a shape nuisance. The shift source is the name without the year/campaign suffix (`pdf_shape_*` -> `pdf`, `murf_envelope_*` -> `murf_envelope`). It is only added for configs that know the shift `<source>_up`.

Names that match nothing only produce a warning ("The following systematics were not considered ..."). Read the log when you change systematics.
Shape nuisances are renamed to the CMS conventions via `rename_systematics`, or get a `CMS_` prefix. Processes are renamed by `inf_proc`: e.g. `tt` -> `ttbar`, `h_ggf` -> `ggH`, `hh_ggf_hbb_hww2l2nu_kl1_kt1` -> `ggHH_kl_1_kt_1_hbbhww2l2nu`, `hh_vbf_..._kv1_k2v1_kl1` -> `qqHH_CV_1_C2V_1_kl_1_...`. The HH signal names follow the convention of the HH inference tools.

With `skip_data=True` the fake data is built from the sum of all background processes. Signal processes are identified by `is_signal` on the config process name (containing `hh_`, or `hhh_` for HHH configs), the same check that sets the `is_signal` flag of the inference processes.


## Inference model -> ML model -> producers
Never pass `--producers` or `--ml-models` to `cf.cms.CreateDatacards` (or the hbw inference tasks). They are resolved from the inference model by `default_ml_model` and `default_producers` in [defaults_and_groups.py](../config/defaults_and_groups.py) (DL) and [sl_defaults_and_groups.py](../config/sl_defaults_and_groups.py) (SL):

1. If `--inference-model` is not given, the config's `default_inference_model` is used.
2. `--ml-models` defaults to the model's `ml_model_name`.
3. `--producers` defaults to `event_weights`, the ML-input producer (`dl_ml_inputs` for DL-HH) and `cats_ml_<first ML model>`. The latter categorizes events into the `ml_*` categories using the ML scores.

Each ML model is trained on its own `training_configs` (see [ml_dl_dih.py](../ml/derived/ml_dl_dih.py)). These can differ from the configs you request datacards for, and the training pulls in the full processing chain of those configs.


## Available models
Only the modules listed under `inference_modules` in [law.cfg](../../law.cfg) are registered: `sl`, `inf_dl_dih` and `inf_dl_trih`. All model names share one registry, so a derived name must be unique among them.

| file | content |
| --- | --- |
| [base.py](base.py) | `HBWInferenceModelBase`, the attribute-driven base class. |
| [constants.py](constants.py) | process renaming, rate-uncertainty tables, shape-systematic tables and systematic renaming. |
| [inf_dl_dih.py](inf_dl_dih.py) | DL HH models (`dl` and all variants), with `config_categories`, `systematics` and `processes_dict` collections. |
| [inf_dl_trih.py](inf_dl_trih.py) | DL HHH models (`dl_trih` and the `Cat_*_Bin_V5_shape*` variants). |
| [sl.py](sl.py) | SL models `sl`, `sl_syst`, `sl_syst_ggf`, `sl_test`. |
| [legacy_dl.py](legacy_dl.py) | legacy, not registered (older `dl`, `dl_apr2025*`, `dl_ml_study_*`). |
| [legacy_sl.py](legacy_sl.py) | legacy, not registered (`default`, `rates_only`, `sl_22`, `sl_22_shapes`, ...). |
| [old_model.py](old_model.py) | legacy, not registered: the first model, written with the `@inference_model` decorator instead of attributes. |

The most relevant models per channel:

- **DL HH** (`hbw_dl`, [inf_dl_dih.py](inf_dl_dih.py)): everything derives from `dl` (ML models `multiclassv3`, `ggfv3`, `vbfv3`; binary-score variables `logit_mlscore.sig_{ggf,vbf}_binary`). `default_unblind` is the config default (`systematics.default`, `config_categories.default_boosted`, real data). Further variants:
  - `default` (pseudo-data), `rate_only` (rate systematics only), `default_data` and its VBF-tagger variants such as `vbfmqq_data`.
  - single-variable models for control distributions: `mbb`, `mll`, `ll_pt_dycr`, `n_jet_dycr`, ...
  - 2024/2025 models with the year-specific systematic sets `rate_24_25` and `full_24_25`: `rate24_251`, `rate24_0`, `full_2425`, `rate_boostedbkg`, `rate_boostedbkg32`. Their ML models are trained on `c24v15` and `c25v15`. `rate24`, `full24`, `rate24_closure`, `full24_closure` and `rate24_v3_lep2pt15` refer to ML models that are commented out in [ml_dl_dih.py](../ml/derived/ml_dl_dih.py) and currently cannot be resolved.

  `systematics.default` only contains the 2022/2023 luminosity nuisances, so for 2024/2025 use the `rate_24_25`/`full_24_25` based models (e.g. `full_2425`).
- **DL HHH** (`hbw_hhh_dl`, [inf_dl_trih.py](inf_dl_trih.py)): `Cat_eq2b_Bin_V5_shape`, `Cat_eq3b_Bin_V5_shape`, `Cat_geq4b_Bin_V5_shape` (one per b-jet multiplicity) and their `*_unblind` twins (real data, but partially unblinded: `unblind=False`). The HHH configs inherit `default_unblind` as their default, so always pass `--inference-model`. Note that the ML models they name (`Bin_V1`, and `multiclass_hhh_v0`/`hhh_v0` for `dl_trih`) are currently commented out.
- **SL** (`hbw_sl`, [sl.py](sl.py)): `sl` (ML model `sl_22post`, rate systematics), `sl_syst` (+ shape systematics), `sl_syst_ggf`. The SL config default `sl_22` lives in the unregistered `legacy_sl.py`, so pass `--inference-model sl` explicitly. The SL rate-systematic names (`QCDScale_*`, `lumi_13TeV_*`) don't match what the base class expects (`QCDscale_*`, `lumi_13p6TeV_<year>`), so they are mostly skipped with a warning.


## Recipes

### Create datacards
```shell
law run cf.cms.CreateDatacards \
    --version v1 \
    --configs c24v15,c25v15 \
    --inference-model full_2425 \
    --workers 4
```
`full_2425` uses the 2024/2025 ML models, so the command also trains them if needed (see the [ML README](../ml/README.md)). Omitting `--inference-model` uses the config default `default_unblind`, whose ML models are trained on the 2022/2023 configs and whose systematics only contain the 2022/2023 luminosity nuisances. The output path contains `inf__<model>`, so each model gets its own folder.

### Derive a variant
Add a `.derive` call to one of the registered files, e.g. at the end of [inf_dl_dih.py](inf_dl_dih.py):
```python
my_rate_test = default_unblind.derive("my_rate_test", cls_dict={
    "systematics": systematics.rate_24,                # fewer systematics
    "config_categories": config_categories.sr_resolved,  # only resolved signal regions
    "ml_model_name": ["multiclass_2425_v3l2pt15", "ggf_2425_v3l2pt15", "vbf_2425_v3l2pt15"],
})
```
Everything not listed in `cls_dict` is inherited. Replacing the variable works with `"config_variable": lambda self, config_cat_inst: "mli_mbb_rebinned3"`, as in the `mbb` model. A model in a new file only becomes visible after adding the module to `inference_modules` in [law.cfg](../../law.cfg). `law index` is not needed, since inference models are not tasks.

### Flat rebinning, blinding and the combine/dhi commands
`cf.cms.CreateDatacards` keeps the binning of the config variable. `hbw.ModifyDatacardsFlatRebin` rebins every category so that the summed histogram of selected processes is flat, with at least 12 effective background MC entries per bin. The selected processes come from the config's `inference_category_rebin_processes` (SM HH for signal categories, backgrounds for background categories). The number of bins comes from `default_bins_per_category`, or from the CLI:
```shell
law run hbw.ModifyDatacardsFlatRebin \
    --version v1 \
    --configs c24v15 \
    --inference-model default_unblind \
    --bins-per-category sr__resolved__1b__ml_sig_ggf=10,sr__boosted__ml_bkg=1
```
For partially unblinded models (`skip_data=False`, `unblind=False`), rates and observation are set to `-1`, and the most signal-like range (above a significance threshold) is moved out of the fitted range into the overflow.

`hbw.PrepareInferenceTaskCalls` builds on these cards. It copies them into its output folder and writes a `Run.sh` with one bash function per step: `prepare_cards` and `combine_cards`, limits (`pointlimits`, `limits_kl`, `limits_c2v`), likelihood scans, `impacts`, `gof` and `postfitshapes`/`prefitshapes`. The steps call law tasks of the HH inference tools (`CombineDatacards`, `PlotUpperLimitsAtPoint`, `PlotPullsAndImpacts`, `MergePreAndPostFitShapes`, ...). `--config-groups` defaults to the four 2022/2023 configs, so set it explicitly; groups are separated by `:`.
```shell
law run hbw.PrepareInferenceTaskCalls \
    --version v1 \
    --config-groups c24v15 \
    --inference-model default_unblind \
    --requested-keys prepare_cards,pointlimits,impacts
```
`Run.sh` has to be executed in a separate inference setup. It expects `$DHI_DATA` and the `prepare_cards.py` helper, neither of which is part of this repo. Run it as `dry_run=true bash Run.sh` to only print the commands, or source it and call single functions. `hbw.PlotShiftedInferencePlots` plots nominal and shifted templates of the rebinned cards per category.

### Datacards for several models at once
```shell
law run hbw.MultiDatacards \
    --version v1 \
    --configs c24v15 \
    --inference-models default_unblind,mbb,mll
```
This runs `cf.cms.CreateDatacards` for each model. It also writes `MultiDatacards_Calls.txt` with a `MergePreAndPostFitShapes` call that applies the fit of the main cards (`$CARDS`) to the cards of all listed models. This is how pre/post-fit distributions of non-fit variables such as `mbb` are produced.

`hbw.CreateMultipleDatacards` ([multiple_datacards.py](../tasks/multiple_datacards.py)) is different: it writes one datacard per variable for models with `multi_variables=True`. No registered model currently sets this.

### Pre- and post-fit plots
`hbw.PlotPostfitShapes` ([postfit_plots.py](../tasks/postfit_plots.py), marked work in progress) reads a `FitDiagnostics` output or a merged pre/post-fit shapes file from the inference side. The `--inference-model` must be the one used for the cards:
```shell
law run hbw.PlotPostfitShapes \
    --version v1 \
    --config c24v15 \
    --inference-model default_unblind \
    --fit-diagnostics-file /path/to/fitdiagnostics_or_merged_shapes.root \
    --processes ddl4 \
    --merged-only False
```
`ddl4` is the process group "data + `dl4`" from [defaults_and_groups_hh.py](../config/defaults_and_groups_hh.py). Add `--prefit` for pre-fit shapes. By default only the merged category is plotted (`--merged-only`); `--skip-merged` disables it.
