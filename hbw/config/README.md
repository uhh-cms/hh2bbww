# Analysis and config

This folder, together with [hbw/analysis](../analysis), defines *what* the analysis runs on: campaigns, datasets, processes, shifts, categories, variables, corrections and the defaults that tasks use when you leave out a parameter.
Most of the objects involved come from [order](https://github.com/riga/order), and columnflow tasks look them up by name.


## Analysis vs. config

An `od.Analysis` is the top-level object and holds analysis-wide settings: sandboxes, histogram hooks, output path rules and the list of configs.
It is selected with `--analysis` (the default comes from `default_analysis` in the active law config file, see the root [README](../../README.md#law-config)).
All instances are built by `create_hbw_analysis` in [create_analysis.py](../analysis/create_analysis.py) and differ only in their tags:

| analysis | file | tags |
| --- | --- | --- |
| `hbw_dl` | [hbw_dl.py](../analysis/hbw_dl.py) | `is_dl`, `is_hh`, `is_nonresonant` |
| `hbw_hhh_dl` | [hbw_hhh_dl.py](../analysis/hbw_hhh_dl.py) | `is_dl`, `is_hhh`, `is_nonresonant` |
| `hbw_sl` | [hbw_sl.py](../analysis/hbw_sl.py) | `is_sl`, `is_nonresonant` |
| `hbw_sl_res` | [hbw_sl_res.py](../analysis/hbw_sl_res.py) | `is_sl`, `is_resonant` |
| `hbw_merged` | [hbw_merged.py](../analysis/hbw_merged.py) | `is_sl`, `is_dl`, `is_resonant`, `is_nonresonant` |

An `od.Config` is one campaign (one data-taking period plus nano version) inside an analysis, selected with `--config`.
It holds the datasets, processes, shifts, categories, variables, luminosity, corrections and everything else stored as auxiliary data (`cfg.x.<key>`).
The config copies the tags of its analysis, so all tag checks below work on the config as well.

On this branch only `hbw_dl` and `hbw_hhh_dl` can build configs.
`add_config` raises if a config has neither `is_hh` nor `is_hhh`, so the SL, resonant and merged analyses fail at config creation.
`hbw_merged` still matters as a *path name*, because shared outputs are written under it (see [Where outputs land](#where-outputs-land)).


## Lazy configs and `hbw.BuildCampaignSummary`

Configs are not built when the analysis is imported.
`add_lazy_config` in [create_analysis.py](../analysis/create_analysis.py) registers a factory, and the config is only built (via `add_config`) when a task first asks for it.

The factory does not read cmsdb directly.
It loads a pickled campaign written by `hbw.BuildCampaignSummary` ([campaigns.py](../tasks/campaigns.py)).
That task merges one or more cmsdb campaigns per config (`campaign_map`, e.g. `c24v15` combines `run3_2024_nano_v15` with the custom HHH samples of `run3_2024_nano_custom_v15`), applies the process modifications from [processes.py](../analysis/processes.py) and pickles the result.
If the pickle is missing, every task fails with a message telling you to run:

```shell
law run hbw.BuildCampaignSummary --config c24v15 --remove-output 0,a,y
```

Rerun it whenever you change cmsdb datasets or [hbw/analysis/processes.py](../analysis/processes.py), because the pickle is a snapshot.
The task also keeps a `backup_dataset_summary.yaml` and warns if the dataset content changed since then (`--recreate-backup-summary` updates it).

Each `add_lazy_config` call registers two configs:

- the full config `cXX...` with the given id,
- a limited twin `lXX...` (id + 1) that reads at most 2 files per dataset. It reuses the campaign pickle of the `c` config, so one `BuildCampaignSummary` covers both.

The limited config also skips LFN-count validation (`validate_dataset_lfns`) and enables the cutflow features (`do_cutflow_features`).
Registered configs are `c17v9`, `c22prev14`, `c22postv14`, `c23prev14`, `c23postv14`, `c24v15`, `c25v15` and `c26v15`.
Support status of the configs:

| configs | status |
| --- | --- |
| `c24v15`, `c25v15` | supported: defaults, ML models, inference models and corrections are set up for these |
| `c22prev14`, `c22postv14`, `c23prev14`, `c23postv14` | not maintained: they may still build, but nothing is tested or updated for them |
| `c26v15` | not yet fully configured: `add_config` has no 2026 pileup/lumi files (`No lumi and pu files provided for year 2026`) and no 2026 trigger scale factors yet |
| `c17v9` | legacy Run 2 config: the Run 2 external files are not implemented (`External files for Run 2 not yet implemented/checked`) |

The custom tt+b processes (`ttbb_custom`, `ttb_custom`, `tt2b_custom`, `tt_bb_custom`) only exist in campaigns with `ttbb_*_powheg` datasets (2024 and later). `configure_hbw_processes` skips them for older campaigns, so process groups that use them (e.g. `dl9`) only work for 2024+.

The pinned `modules/cmsdb` commit contains all of these campaigns, so the limitations are on the analysis side.
All law config files default to `c24v15`; pass `--config l24v15` for quick tests on 2 files per dataset.


## How `add_config` builds a config

[config_run2.py](config_run2.py) holds `add_config`; despite the name, it covers Run 2 and Run 3.
In order, it:

1. Creates the `od.Config` with the analysis tags. It sets `cfg.x.lepton_tag` (`sl`/`dl`) and `cfg.x.signal_tag` (`hh`/`hhh`), and chooses which mll and b-jet category functions to use.
2. Adds datasets and processes (`add_hbw_processes_and_datasets` and `configure_hbw_datasets` in [datasets.py](datasets.py), `configure_hbw_processes` in [processes.py](processes.py)) and sets colors and labels (`stylize_processes` in [styling.py](styling.py)).
3. Sets luminosity, JEC/JER, b-tagging (working points, `b_tagger`, `btag_column`, plus the WP-based SF config from [corrections.py](corrections.py) for 2024+), MET, lepton SF/calibration configs and top-pt/V+jets/DY reweighting settings.
4. Registers shifts with `cfg.add_shift` and maps them to columns with `add_shift_aliases`. JEC source shift ids come from [jec_sources.yaml](jec_sources.yaml), whose order must not change.
5. Registers external files (`cfg.x.external_files`, mostly correctionlib JSONs located through `CATInfo`) and triggers (`add_triggers` in [trigger.py](trigger.py)).
6. Defines `cfg.x.keep_columns`, i.e. what survives `cf.ReduceEvents`, and pins `cf.CalibrateEvents` to `default_common_version` via `cfg.x.versions`.
7. Adds the selection-level categories and the base variables, then sets the `default_*` aux and groups (next section).
8. Runs the channel-specific hooks: `configure_dl` (DL + non-resonant), `configure_sl` (SL + non-resonant) or `configure_sl_res` (SL + resonant). On this branch only `configure_dl` is reached; it adds `m_ll` and `channel_id` to the kept columns.

To make something era-dependent, use `cfg.x.if_era(run=..., year=..., tag=..., cfg_tag=..., values=[...])`, as [datasets.py](datasets.py) does.


## Defaults and groups

When you omit `--selector`, `--producers`, `--categories` etc., columnflow looks up `default_<name>` in the config aux.
Values can be plain strings or callables that are evaluated per task.
A group name given on the command line (e.g. `--categories dl_sr`) is expanded through the matching `<name>_groups` aux.
Both are set by one function, chosen by tag:

| tags | file | function |
| --- | --- | --- |
| `is_dl` + `is_hh` | [defaults_and_groups_hh.py](defaults_and_groups_hh.py) | `set_dl_hh_config_defaults_and_groups` |
| `is_dl` + `is_hhh` | [defaults_and_groups_hhh.py](defaults_and_groups_hhh.py) | `set_dl_hhh_config_defaults_and_groups` |
| `is_sl` | [sl_defaults_and_groups.py](sl_defaults_and_groups.py) | `set_sl_config_defaults_and_groups` (not reachable on this branch) |

Both DL variants first call `set_dl_config_defaults_and_groups` from [defaults_and_groups.py](defaults_and_groups.py) and then override parts of it.
For a DL HH config (`hbw_dl`) the defaults resolve to:

| parameter | default |
| --- | --- |
| calibrators | `ak4`, `ak8`, `ele`, `muo` (override with `default_calibrators` in the law config) |
| selector | `dl1` (`sl1` for SL) |
| reducer | `default` |
| producers | `event_weights`, `pre_ml_cats`, `dl_ml_inputs` (`hhh_dl_ml_inputs` for HHH) |
| hist producer | `with_dy_corr` (`hhh_default` for HHH) |
| inference model | `default_unblind` |
| ML model | taken from the inference model's `ml_model_name`, otherwise `default_ml_models` from the law config; ML training tasks fall back to `multiclass_2425_v32l2pt15` |
| categories / variables | `incl`, `sr`, `dycr`, `ttcr` / `jet0_pt`, `mll`, `n_jet`, `ptll`, `lepton0_pt`, `lepton1_pt` |

The producer default (`default_producers`) depends on the ML model.
If an ML model is set or resolved (for example through the inference model in `cf.cms.CreateDatacards`), `pre_ml_cats` is replaced by `cats_ml_<ml_model>`, which is appended last so that its `category_ids` win.
Tasks of the ML training pipeline drop `pre_ml_cats` entirely.
For these tasks the default ML model is the first entry of `default_ml_models` in the law config, or otherwise `multiclass_2425_v32l2pt15` (`sl_22post` for SL).
The analysis also stores `default_ml_model` as aux ([create_analysis.py](../analysis/create_analysis.py)), so that `cf.MLEvaluation` resolves the same default.

Groups defined in these files include `process_groups`, `dataset_groups`, `category_groups`, `variable_groups`, `shift_groups`, `selector_step_groups`, `producer_groups`, `general_settings_groups`, `process_settings_groups`, `variable_settings_groups` and `custom_style_config_groups`.
There are also inference helpers: `default_bins_per_category` and `inference_category_rebin_processes`.

To see what a task resolves to, ask law for the status instead of running it:

```shell
law run cf.ProduceColumns --version v1 --config l24v15 --dataset tt_dl_powheg --print-status 0
```

The resolved selector and producers then appear in the printed task and its output path.
Alternatively, inspect the config interactively (after `source setup.sh`):

```python
from hbw.analysis.hbw_dl import hbw_dl
cfg = hbw_dl.get_config("l24v15")  # triggers the lazy factory
cfg.x.default_selector, cfg.x.default_hist_producer, cfg.x.category_groups["dl_sr"]
```


## Where outputs land

Outputs go to the location set for each task family in the `[outputs]` section of the active law config.
For `local`, that is `$CF_STORE_LOCAL` (default `$CF_DATA/cf_store`).
An entry can name a *store-parts modifier* as its last field (format documented in `$CF_BASE/law.cfg`), for example in [law.dl.cfg](../../law.dl.cfg):

```
task_cf.ReduceEvents: local, %(shared_dust_location_balduin)s, hbw_parts
```

The modifiers live in `analysis_inst.x.store_parts_modifiers` in [create_analysis.py](../analysis/create_analysis.py).
`hbw_parts` does the following:

- Writes `cf.GetDatasetLFNs`, `cf.CalibrateEvents`, `cf.SelectEvents`, `cf.ReduceEvents` and their merge/provide tasks under the analysis name `hbw_merged`, so all sub-analyses share them.
- Maps `l` configs to their `c` name for `cf.GetDatasetLFNs`, `cf.CalibrateEvents`, `cf.SelectEvents` and `cf.ReduceEvents`, so limited and full configs share those outputs.
- Reorders the path to `analysis/calibrators/selector/reducer/config/producers/ml.../hist_producer/inf_model/task_family/.../shift/dataset/version`.
- Drops `--version` from the path for the calibration, selection, reduction, `cf.ProduceColumns` and ML tasks. Their path already contains each module's own `version` attribute, so you bump that attribute (e.g. in a selector's `cls_dict`) to get a new output.
- Appends `V<n>` to the dataset or shift part if the dataset or shift has a `version` aux (e.g. `w_lnu_amcatnlo` with `use_old_version`, all `hh_vbf_hbb_htt*` datasets).
- For histogram tasks of non-DY datasets, reuses the path of the hist producer's `nondy_hist_producer`, so variants that only change DY weights share the other histograms.

The second modifier, `pre_reducer`, is legacy: it references the old `weight_producer` part, and no law config uses it.
[law.shared.cfg](../../law.shared.cfg) also refers to modifiers (`merged_analysis`, `limited_config_shared`) that no longer exist.


## Recipes

### Add a variable

Add a `config.add_variable(...)` call to `add_variables` in [variables.py](variables.py).
The `expression` is either a column string (`"Jet.pt[:,0]"`) or a callable.
A callable needs its input columns declared in `aux["inputs"]`:

```python
config.add_variable(
    name="ptll",
    expression=lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
    binning=(240, 0., 1200.),
    unit="GeV",
    x_title=r"$p_{T}^{ll}$",
    aux={"inputs": {"{Electron,Muon}.{pt,eta,phi,mass,charge}"}},
)
```

Variables of columns created by a producer are usually registered in that producer's `init`, so they only exist when the producer is used:

- [dl/variables.py](dl/variables.py) is used by `dl_ml_inputs` and `hhh_dl_ml_inputs`.
- [ml_variables.py](ml_variables.py) is used by `sl_ml_inputs`.
- [cutflow_variables.py](cutflow_variables.py) is used by the cutflow and gen selectors/producers.

These helpers are wrapped in `call_once_on_config` from [util.py](../util.py), so calling them repeatedly is safe.
Shared binning, units and titles are in [styling.py](styling.py) (`default_var_binning`, `default_var_unit`).
To plot the new variable as part of a set, add it to a `variable_groups` entry in the defaults file.

### Add a category

1. Write a categorizer in [categorization/categories.py](../categorization/categories.py) with `@categorizer(uses={...})`. It returns `events, mask`.
2. Register the category in [categories.py](categories.py) with `config.add_category(name=..., id=..., selection="<categorizer name>")`. Pick an id in the right power-of-ten block (see the module docstring), because combined categories get the *sum* of their parents' ids.

Categories added in `add_categories_selection` (`incl`, `sr`/`dycr`/`ttcr`, lepton channels) are assigned during `cf.SelectEvents`.
Jet and b-jet categories, and all combinations such as `sr__2mu__resolved__2b`, are built in `add_categories_production`.
That function is called by the `pre_ml_cats` producer ([production/categories.py](../production/categories.py)) during `cf.ProduceColumns`.
ML output categories are added by `cats_ml_<model>` via `add_categories_ml`.
After changing categories, the corresponding `cf.SelectEvents` or `cf.ProduceColumns` outputs have to be recreated.

### Add a dataset

1. Make sure the dataset exists in the cmsdb campaign(s) that `campaign_map` in [campaigns.py](../tasks/campaigns.py) lists for your config.
2. Add its name to the list of its process in `hbw_dataset_names` in [datasets.py](datasets.py). Wrap it in `config.x.if_era(...)` if it only exists for some eras. The dict key must be a process name known to the campaign, since the process is added with `procs.n(<key>)`.
3. Rerun `hbw.BuildCampaignSummary` for the config if cmsdb changed.

A dataset name that is not in the campaign is only logged as a warning and skipped, so check the log when a dataset silently does not show up.
For 2024+ every MC dataset must match exactly one pattern in `btag_wp_eff_groups` (in `configure_hbw_datasets`), otherwise config creation raises.
Dataset tags such as `is_ttbar`, `is_dy` or `has_lhe_weights` are also assigned in `configure_hbw_datasets`, based on name patterns.

### Add a process

Processes come from cmsdb.
A process becomes part of the config when it is a key in `hbw_dataset_names` with at least one dataset.
For a custom grouping of existing processes, follow these steps:

1. Create it in `modify_cmsdb_processes` in [hbw/analysis/processes.py](../analysis/processes.py), using `create_parent_process(children, name=..., id=..., label=...)`. The id must be unique, which `unique_process_sanity_check` enforces.
2. Add it to the config in `configure_hbw_processes` ([processes.py](processes.py)) via `config.add_process(config.x.procs.n.<name>)`.
3. Optionally set its color and label in `default_process_colors` / `default_labels` ([styling.py](styling.py)) and add it to a `process_groups` entry.
4. Rerun `hbw.BuildCampaignSummary`, because process modifications are baked into the campaign pickle.


## Files

| file | content |
| --- | --- |
| [config_run2.py](config_run2.py) | `add_config`: the main config builder for all campaigns |
| [datasets.py](datasets.py) | dataset lists per process and era, dataset tags, limited-file handling, UHH-campaign LFN lookup |
| [processes.py](processes.py) | `configure_hbw_processes`, process helpers (`create_parent_process`, dummy xsecs, ML process preparation) |
| [categories.py](categories.py) | category definitions for selection, production and ML stages |
| [variables.py](variables.py) | base variables available after `cf.ReduceEvents` |
| [ml_variables.py](ml_variables.py) | SL ML input variables (added by `sl_ml_inputs`) |
| [cutflow_variables.py](cutflow_variables.py) | `cf_*` cutflow variables and gen-level variables |
| [styling.py](styling.py) | color palettes, process colors/labels, default variable binnings/units, `quick_addvar` |
| [trigger.py](trigger.py) | `Trigger` / `TriggerLeg` classes and `add_triggers` |
| [corrections.py](corrections.py) | `btag_sf_cfg`: WP-based b-tag SF configuration (2024+) |
| [scale_factors.py](scale_factors.py) | `configure_for_scale_factors`; not called anywhere, superseded by the `triggersf` reducer |
| [jec_sources.yaml](jec_sources.yaml) | ordered JEC source list used to assign stable shift ids |
| [defaults_and_groups.py](defaults_and_groups.py) | DL base defaults/groups and the resolver functions (`default_producers`, `default_ml_model`, ...) |
| [defaults_and_groups_hh.py](defaults_and_groups_hh.py) | DL HH overrides: signal, process/category/variable groups, plot settings |
| [defaults_and_groups_hhh.py](defaults_and_groups_hhh.py) | DL HHH overrides |
| [sl_defaults_and_groups.py](sl_defaults_and_groups.py) | SL defaults/groups (currently unreachable) |
| [dl/](dl) | `configure_dl` and DL/HHH variables |
| [sl/](sl) | `configure_sl` (adds the QCD process) |
| [sl_res/](sl_res) | `configure_sl_res` and resonant variables (their producer module is not loaded in `law.cfg`) |
| [../analysis/create_analysis.py](../analysis/create_analysis.py) | `create_hbw_analysis`: lazy configs, store-parts modifiers, sandboxes |
| [../analysis/hist_hooks.py](../analysis/hist_hooks.py) | hooks for `--hist-hooks`: `cumsum`, `cumsum_reverse`, `rebin`, `blind_bins_above_score` |
| [../analysis/processes.py](../analysis/processes.py) | `modify_cmsdb_processes`: DY hf/lf splitting and custom parent processes, applied in `hbw.BuildCampaignSummary` |
