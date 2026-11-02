# Analysis tasks and tooling

This folder holds the analysis-specific [law](https://github.com/riga/law) tasks (namespace `hbw.`).
The main processing chain is made of columnflow tasks (namespace `cf.`): calibration, selection, reduction, column production, ML, histograms, plots and datacards.
The `hbw.` tasks add what columnflow does not provide, such as campaign building, inspection, derived corrections, custom tables and plots, and ML/inference helpers.
This README also covers the helper scripts, plot functions, examples and tests that you use alongside these tasks.

The configs are designed for 2024 and 2025 (`c24v15`, `c25v15`); for the status of the other configs see the [config README](../config/README.md).
The commands below use `c24v15` / `l24v15`.


## Running tasks with law

### Setup and law config

```shell
source setup.sh dev            # once per shell
law index --verbose            # after adding/renaming tasks, enables autocompletion
echo $LAW_CONFIG_FILE          # the law config that is active
law config analysis.default_config
```

`setup.sh` sets `LAW_CONFIG_FILE` to `$HBW_BASE/$HBW_LAW_CONFIG`.
`HBW_LAW_CONFIG` is asked for during the interactive setup, defaults to `law.dl.nocert.cfg` and is stored in `.setups/<name>.sh`.
The variant files inherit from each other and mostly change the defaults used when `--analysis`/`--config` are omitted:

| file | inherits from | `default_analysis` | `default_config` |
| --- | --- | --- | --- |
| [law.cfg](../../law.cfg) | `$CF_BASE/law.cfg` | `hbw_merged` | `c24v15` |
| [law.nocert.cfg](../../law.nocert.cfg) | `law.cfg` | (inherited) | (inherited) |
| [law.dl.nocert.cfg](../../law.dl.nocert.cfg) | `law.nocert.cfg` | `hbw_dl` | `c24v15` |
| [law.dl.cfg](../../law.dl.cfg) | `law.cfg` | `hbw_dl` | `c24v15` |
| [law.hhh.dl.cfg](../../law.hhh.dl.cfg) | `law.dl.cfg` | `hbw_hhh_dl` | `c24v15` |
| [law.sl.cfg](../../law.sl.cfg), [law.sl.nocert.cfg](../../law.sl.nocert.cfg) | `law.cfg` / `law.nocert.cfg` | `hbw_sl` | `c24v15` |
| [law.resonant.nocert.cfg](../../law.resonant.nocert.cfg) | `law.nocert.cfg` | `hbw_sl_res` | `c24v15` |

The `*.nocert.cfg` files set `skip_ensure_proxy: True` and read input files from the local DESY dCache (`lfn_sources: local_desy_dcache`), so no grid proxy is needed on the NAF.
All of them default to `c24v15`. Note that only the DL analyses (`hbw_dl`, `hbw_hhh_dl`) can build configs, see the [config README](../config/README.md).

[columnflow_patches.py](../columnflow_patches.py) changes the default of `--version` to `analysis.default_version` from `law.cfg` (currently `prod3`).
If you leave out `--version`, your outputs therefore end up in the shared `prod3` production.
Pass your own version while you are testing.

### Everyday parameters

| parameter | what it does |
| --- | --- |
| `law run <task> --help` | lists all parameters of a task, including the defaults resolved from the config |
| `--version v1` | output version; part of every output path |
| `--print-status 0` | shows which outputs exist, without running anything. The value is the recursion depth: `0` means only this task, `-1` means all dependencies |
| `--print-output 0` | prints the output paths (same depth syntax) |
| `--remove-output 0,a,y` | deletes outputs. The values are depth, mode (`i` interactive, `a` all, `d` dry run) and a flag to run the task afterwards |
| `--fetch-output 0,a` | copies the outputs into the current directory |
| `--branch 0` / `--branches 0,3,7` | runs one branch / a subset of branches of a workflow (e.g. one input file per branch). `:` gives ranges, e.g. `--branches 0:10` |
| `--workflow htcondor` | submits the branches as HTCondor jobs instead of running them locally (`local` is the default) |
| `--workers 4` | number of tasks law runs in parallel in your shell |

You can set parameters of upstream tasks with `--<task family>-<param>`.
Shell brace expansion makes this shorter:

```shell
law run cf.MergeReducedEventsWrapper --configs c24v15 --datasets "*" --shifts nominal --workers 50 \
    --cf.MergeReducedEvents-{retries=2,workflow=htcondor} \
    --cf.ReduceEvents-pilot
```

Useful remote-workflow parameters are `--pilot`, `--no-poll`, `--parallel-jobs`, `--tasks-per-job` and `--htcondor-memory 4GB`.
`--pilot` drops the workflow's requirements, so the jobs are submitted right away and each branch resolves its own inputs.
`--no-poll` submits the jobs and returns instead of waiting for them.
`--retries` defaults to 3 (also set by the patches).
`claw` ([modules/columnflow/bin/claw](../../modules/columnflow/bin/claw)) is a drop-in for `law` that runs the command inside the columnar dev sandbox.
The helper scripts use it for tasks that need the sandbox already on the submitting side.

### Pinning versions and resources in law.cfg

The `[versions]` and `[resources]` sections of [law.cfg](../../law.cfg) map lookup keys to values.
A key has the form `cfg_<config>__task_<family>__dataset_<dataset>__shift_<shift>`, and each part may be a pattern. The order of the keys sets the resolution order, so put specific keys above general ones.
`[versions]` pins `--version` per task.
`cfg.x.versions` in [config_run2.py](../config/config_run2.py) also pins one: `cf.CalibrateEvents` always uses `analysis.default_common_version` (`common3`).
`[resources]` sets parameters such as `htcondor_memory=3GB` for remote jobs, e.g. `task_cf.ReduceEvents__dataset_dy*`.

On the NAF, a patch makes every HTCondor job claim one `naf_<username>` resource.
The per-user limits are listed in `[luigi_resources]`, so add a line for your user there.


## Base classes ([base.py](base.py))

- `HBWTask` subclasses columnflow's `BaseTask` and only sets `task_namespace = "hbw"`. Every task that inherits it shows up as `hbw.<ClassName>`.
- `ColumnsBaseTask` is a local workflow over the merged reduced files of one dataset. It requires `cf.MergeReducedEvents` plus `cf.ProduceColumns` for each `--producers` entry and `cf.MLEvaluation` for each `--ml-models` entry. Use it as the base for tasks that need to read the final event columns.

Tasks are built by combining columnflow mixins (`CalibratorClassesMixin`, `SelectorClassMixin`, `ProducerClassesMixin`, `MLModelsMixin`, `HistProducerClassMixin`, `CategoriesMixin`, ...).
Each mixin adds its CLI parameters and resolves the config defaults described in [CLAUDE.md](../../CLAUDE.md).
Read the class bases to see which parameters a task accepts.


## Which modules are loaded

law only knows the tasks of modules listed under `[modules]` in [law.cfg](../../law.cfg):

```
hbw.tasks.{inspection,campaigns,ml,inference,postfit_plots,plotting,wrapper,union,optimization,corrections,dy_corrections,trigger_sf,yields,multiple_datacards,merge_nanos}
```

Two tracked files are **not** listed:
- [create_clib_file.py](create_clib_file.py) has no tasks. It provides `create_dy_weight_correction`, which `hbw.ExportDYWeights` imports to build the correctionlib payload.
- [resolve_dummy.py](resolve_dummy.py) defines `ResolveDummy`. To use it, add the module to `[modules]` and rerun `law index`.


## Task overview

| module | tasks | purpose |
| --- | --- | --- |
| [campaigns.py](campaigns.py) | `hbw.BuildCampaignSummary` | merges the cmsdb campaigns of a config into the pickled campaign that the lazy config factory loads. **Needed before any other task** |
| [inspection.py](inspection.py) | `hbw.SelectionSummary`, `hbw.DatasetSummary`, `hbw.DumpAnalysisSummary`, `hbw.CheckConfig`, `hbw.CheckMixins`, `hbw.CheckColumns`, `hbw.DummyWorkflow` | tables of selection efficiencies and datasets; debugging sessions for configs, parameters and columns |
| [plotting.py](plotting.py) | `hbw.PlotVariablesMultiHistProducer` | overlays the same distribution made with several hist producers |
| [yields.py](yields.py) | `hbw.CustomCreateYieldTable` | yield tables (txt/csv/json + summary plot) with ratios and normalisation options |
| [wrapper.py](wrapper.py) | `hbw.ControlPlots`, `hbw.MLInputPlots`, `hbw.InferencePlots`, `hbw.ShiftedInferencePlots` | wrappers that request predefined sets of `cf.PlotVariables1D` / `cf.PlotShiftedVariables1D` / `cf.CreateYieldTable` |
| [union.py](union.py) | `hbw.CustomUniteColumns` | writes reduced + produced + ML columns into one flat csv/parquet/root file per branch |
| [corrections.py](corrections.py) | `hbw.GetBtagNormalizationSF` | b-tag shape-weight renormalisation SFs (correctionlib json) |
| [dy_corrections.py](dy_corrections.py) | `hbw.ComputeDYWeights`, `hbw.ExportDYWeights` | DY n_jet/p_T(ll) reweighting fitted in `dycr`, exported to correctionlib |
| [trigger_sf.py](trigger_sf.py) | `hbw.ComputeTriggerSF` | dilepton trigger SFs and plots from MET-triggered data vs MC |
| [merge_nanos.py](merge_nanos.py) | `MergeNanos` | `hadd`s privately produced NanoAOD files into ~2.4 GB chunks |
| [optimization.py](optimization.py) | (not described here) | listed in `[modules]`, so its tasks are loaded |
| [ml.py](ml.py) | `hbw.MLPreTraining`, `hbw.MLEvaluationSingleFold`, `hbw.PlotMLResultsSingleFold`, `hbw.PlotMLResultsSingleFoldTest`, `hbw.MLOptimizer`, `SimpleMergeMLEvents` | ML helpers, see [hbw/ml/README.md](../ml/README.md) |
| [inference.py](inference.py) | `hbw.ModifyDatacardsFlatRebin`, `hbw.PlotShiftedInferencePlots`, `hbw.PrepareInferenceTaskCalls`, `hbw.MultiDatacards` | datacard post-processing, see [hbw/inference/README.md](../inference/README.md) |
| [multiple_datacards.py](multiple_datacards.py) | `hbw.CreateMultipleDatacards` | see [hbw/inference/README.md](../inference/README.md) |
| [postfit_plots.py](postfit_plots.py) | `hbw.PlotPostfitShapes` | pre-/post-fit plots from fit results, see [hbw/inference/README.md](../inference/README.md) |

`MergeNanos`, `SimpleMergeMLEvents` and `ResolveDummy` do not inherit from `HBWTask`, so they do not get the `hbw.` prefix.
Check their family names with `law index --verbose`.


## Task modules

### Campaigns ([campaigns.py](campaigns.py))

`campaign_map` lists the cmsdb campaigns that are merged for each config.
For example, `c24v15` combines `run3_2024_nano_v15` with `run3_2024_nano_custom_v15`, and the latter wins for `hhh_*` datasets.
`broken_files` lists input files that are skipped per dataset.
The task writes `dataset_summary.yaml`, `campaign_summary.yaml` and `hbw_campaign_inst.pickle`, independent of `--version`.
It only accepts the full config names (the keys of `campaign_map`), and one run serves both the full and the limited config.

```shell
law run hbw.BuildCampaignSummary --config c24v15 --remove-output 0,a,y
```

Rerun it after updating `modules/cmsdb` or editing `broken_files`.
It compares the result to a stored `backup_dataset_summary.yaml` and logs the differences.
Add `--recreate-backup-summary` to accept them.

### Inspection ([inspection.py](inspection.py))

- `hbw.SelectionSummary` reads `cf.MergeSelectionStats` (and runs the selection if needed). It writes per-dataset efficiency, expected yield, number of selected events and cross section, plus a step-by-step table, as csv and text. Choose columns with `--keys-of-interest`.
- `hbw.DatasetSummary` dumps `n_events`/`n_files` per dataset and source campaign of a config to yaml.
- `hbw.DumpAnalysisSummary` writes a LaTeX dataset table and a csv of DAS keys, processes and cross sections, for the analysis note.
- `hbw.CheckConfig` runs the inits of the selected calibrators/selector/producers/ML models, prints the resulting categories and variables and opens IPython with the config. `hbw.CheckMixins` does the same for the full plotting-parameter stack. Both have an always-missing dummy output, so they run every time. Pass `--skip-debugger` to only print.
- `hbw.CheckColumns` (a `ColumnsBaseTask`) lists the fields of the reduced, produced and ML-evaluated columns of one branch and then opens IPython with all columns merged (`--skip-debugger` to only print). Like the `Check*` tasks, it never writes its output, so it runs every time.
- `hbw.DummyWorkflow` is the dummy base of the `Check*` tasks.

```shell
law run hbw.SelectionSummary --version v1 --config c24v15 --datasets tt_dl_powheg,hh_ggf_hbb_hvv_kl1_kt1_powheg
law run hbw.CheckConfig --version v1 --config l24v15 --dataset tt_dl_powheg --producers event_weights --skip-debugger
```

### Plots and tables ([plotting.py](plotting.py), [yields.py](yields.py), [wrapper.py](wrapper.py))

`hbw.PlotVariablesMultiHistProducer` takes `--hist-producers a,b` and plots the summed processes once per hist producer, with a ratio to the first one.
Its legend labels are hard-coded to "Before/After DY NLO reweighting".

`hbw.CustomCreateYieldTable` is a customised `cf.CreateYieldTable`.
It adds the options `--normalize-yields {per_process,per_category,all}`, `--ratio data,background`, `--ratio-modes`, `--transpose`, `--table-format` (any `tabulate` format, e.g. `latex_raw`) and `--output-suffix`:

```shell
law run hbw.CustomCreateYieldTable --version v1 --configs c24v15 --processes tt,dy,st --categories sr,dycr,ttcr --table-format latex_raw
```

The wrapper tasks in [wrapper.py](wrapper.py) only require columnflow plotting tasks.
`hbw.InferencePlots` plots the datacard variable of every category of `--inference-model`, and `--add-variables` also plots `--variables`.
`hbw.ShiftedInferencePlots` plots the systematic shifts that the inference model assigns to each process.
The module docstring warns that these tasks have not been tested since the TaskArrayFunction init changes.

### Column export ([union.py](union.py))

`hbw.CustomUniteColumns` runs `--union-producer` (default `sync_exercise`) on the merged columns.
It writes one file per branch in the format given by `--file-type` (`parquet`, `csv` or `root`).
Use it for synchronisation exercises or to hand events to external code.

### Derived corrections ([corrections.py](corrections.py), [dy_corrections.py](dy_corrections.py), [trigger_sf.py](trigger_sf.py))

These tasks produce correctionlib files that producers later consume.
Producers require them automatically, so you rarely call them by hand except to inspect plots:

- `hbw.GetBtagNormalizationSF` derives SFs that restore the event yield after applying b-tag shape weights, binned in HT, n_jet and n_hf. It is required by the `normalized_btag_weights` producer ([normalized_btag.py](../production/normalized_btag.py)). Options: `--rescale-mode {nevents,xs}`, `--njet-overflow`, `--nhf-overflow`.
- `hbw.ComputeDYWeights` fits the data/MC ratio in `dycr__2mu` as a function of `n_jet-ptll_for_dy_corr_V3` per config. `hbw.ExportDYWeights` combines the configs into `dy_correction_weight.json.gz`. The `dy_correction_weight` producer ([dy_correction_weight.py](../production/dy_correction_weight.py)) requires `hbw.ExportDYWeights`.
- `hbw.ComputeTriggerSF` computes trigger SFs per dilepton channel from MET-triggered data (`data_met`) against background MC. It uses fine 2D lepton-p_T histograms and optimises the binning; expect about 9 GB of memory.

```shell
law run hbw.GetBtagNormalizationSF --version v1 --config c24v15 --shift nominal
law run hbw.ExportDYWeights --version v1 --configs c24v15 --workflow htcondor --workers 50
law run hbw.ComputeTriggerSF --version v1 --configs c24v15
```

### Other modules

- [merge_nanos.py](merge_nanos.py): `MergeNanos --sample <name>` merges custom HHH NanoAOD files. The input and output paths are hard-coded to personal `/data/dust` areas.
- [resolve_dummy.py](resolve_dummy.py): `ResolveDummy` stacks all CSP/plotting mixins with a no-op `run`. It times `resolve_param_values`, which helps when profiling slow parameter resolution. It is not loaded by default (see above).
- [optimization.py](optimization.py): listed in `[modules]`; not described here.


## Helper scripts ([hbw/scripts](../scripts))

`setup.sh` adds `hbw/scripts` to `PATH`.
Run Python scripts that need columnflow inside a sandbox, e.g. `cf_sandbox venv_columnar_dev 'python <script>'`.

- [workflow.sh](../scripts/workflow.sh) is the reference for running the full DL chain. It contains one shell function per step: `recreate_campaign_summary`, `run_merge_reduced_events`, `run_merge_selection_stats`, `prepare_dy_corr`, `run_ml_training`, `prepare_mlcolumns`, `run_ml_evaluation`, `run_merge_shifted_histograms_htcondor`, `run_datacards`, and many `run_and_fetch_*` plotting functions. Its built-in defaults are `c24v15`, the ML models `multiclass_2425_v3l2pt15`/`ggf_2425_v3l2pt15`/`vbf_2425_v3l2pt15` and the inference model `full_2425`. Sourcing the file only defines the functions:

  ```shell
  source hbw/scripts/workflow.sh
  dry_run=true                         # print commands instead of running them
  recreate_campaign_summary c24v15
  ```
- [hbwtasks.sh](../scripts/hbwtasks.sh) is meant to be sourced. It defines `hbw_*` shortcuts that submit columnflow wrappers to HTCondor: `hbw_calibration`, `hbw_reduction`, `hbw_merge_reduction`, `hbw_reduction_status`, `hbw_ml_training`, `hbw_datacards`, and some plotting functions. Version and config come from `law config analysis.default_version` / `default_config`, and extra arguments are appended (e.g. `hbw_reduction --configs c24v15`). It also defines the alias `hbw_synchronization`. The plotting functions are marked as untested.
- [test_config.py](../scripts/test_config.py) loads `analysis.default_analysis` / `default_config` and prints processes (with event counts), datasets, DAS keys, categories, variables, shifts and aux keys. It uses the default config, so run it with a law config that defaults to `c24v15` (e.g. `law.dl.cfg`).
- [check_correction_json.py](../scripts/check_correction_json.py) `<file.json.gz>` lists the corrections in a correctionlib file and opens IPython.
- [find_empty_files.zsh](../scripts/find_empty_files.zsh) `<dir> [--delete] [--gfal]` finds (and optionally removes) zero-size outputs, e.g. after failed jobs.
- [cmsdb/](../scripts/cmsdb) holds the tools for adding datasets to cmsdb. [parse_datasets.py](../scripts/cmsdb/parse_datasets.py) collects DAS strings per campaign. [build_campaign.py](../scripts/cmsdb/build_campaign.py) turns them into cmsdb dataset files. Both call `my_get_das_info.py`, so first copy [get_das_info_nano_leg_v14.py](../scripts/cmsdb/get_das_info_nano_leg_v14.py) or [get_das_info_nano_leg_v15.py](../scripts/cmsdb/get_das_info_nano_leg_v15.py) under that name. Check the generated names by hand.
- [example.py](../scripts/example.py), [run_genplots.py](../scripts/run_genplots.py), [ml_grid_search.py](../scripts/ml_grid_search.py) and [synchronization.py](../scripts/synchronization.py) show how to build tasks in Python and call `task.law_run()`. They are older and may need updating. [check_hhh_2024_datasets.sh](../scripts/check_hhh_2024_datasets.sh) is a personal DAS query script.


## Plot functions ([hbw/plotting](../plotting))

Any columnflow plot task accepts `--plot-function <module.function>`, and `--general-settings key=value` passes extra arguments to that function:

| function | use |
| --- | --- |
| [example.py](../plotting/example.py) `my_plot1d_func` | minimal template for writing your own plot function |
| [s_over_b.py](../plotting/s_over_b.py) `plot_s_over_b`, `cutflow_s_over_b` | S/B or S/sqrt(B) per bin (optionally cumulative) or per selection step (with `cf.PlotCutflow`) |
| [plot_fits.py](../plotting/plot_fits.py) `plot_fit` | fits `gauss`/`scalable_norm`/`scalable_exponnorm` to a distribution |
| [plot_ratio.py](../plotting/plot_ratio.py) `plot_ratio`, `plot_variable_efficiency_slice` | ratio plots; cut efficiency vs. cut value in slices of a 2D histogram |
| [plot_2d_ratios.py](../plotting/plot_2d_ratios.py) `plot_2d` | 2D plots with configurable z range and out-of-range colours |

```shell
law run cf.PlotVariables1D --version v1 --config l24v15 --processes tt,hh_ggf_hbb_hvv_kl1_kt1 --variables jet1_pt \
    --plot-function hbw.plotting.s_over_b.plot_s_over_b --general-settings sqrt_b
```


## Python helpers

- [columnflow_patches.py](../columnflow_patches.py): `patch_all()` runs on every `import hbw` (see [hbw/\_\_init\_\_.py](../__init__.py)). It sets the `--version` and `--retries` defaults and adds the CSP / inference-model `version` attribute to output paths (`<name>V<version>`). It also trims the `cf.MLTraining` requirements, adds the NAF HTCondor resource and makes `--fetch-output` hashes stable. If columnflow behaves differently here than in its docs, look here first.
- [util.py](../util.py): shared helpers. Among them are `debugger()` (IPython with the caller's namespace; `q()` exits), `call_once_on_config`, `four_vec`, `build_param_product`, `timeit_multiple` and `round_sig`. It also defines the deferred-column helpers used in `uses`/`produces` (`IF_NANO_V14`, `IF_2024`, `IF_DL`, `IF_MC`, `BTAG_COLUMN`, `MET_COLUMN`, ...).
- [hist_util.py](../hist_util.py): `apply_rebinning_edges` rebins one axis of a `hist` histogram to a subset of its edges.


## Examples and notebooks

[examples/000_test](../../examples/000_test) contains small shell scripts for one step each (`01_selection.sh` ... `07_plot_shift.sh`).
The shared settings are in `common.sh`.
They run on the limited config `l24v15` with a few DL datasets; change `common.sh` to test other configs, datasets or categories.
Extra arguments are passed through to `law run`.

[notebooks/](../../notebooks) contains [build_jupyter_kernel.sh](../../notebooks/build_jupyter_kernel.sh), which creates a Jupyter kernel `cf_ml` from your `.setups/dev.sh` (edit `CF_SETUP_NAME` if your setup has a different name).
[setup.py](../../notebooks/setup.py) adds hbw, columnflow, cmsdb, law and order to `sys.path` and sets law-config defaults (`hbw_dl`/`c24v15`).
[example.ipynb](../../notebooks/example.ipynb) shows the import check.


## Linting and tests

CI ([lint_and_test.yaml](../../.github/workflows/lint_and_test.yaml)) runs on Python 3.9 after `source setup.sh ""`.
Run the same checks locally before pushing:

```shell
./tests/run_linting                 # flake8 hbw tests (config: .flake8, max line length 120, double quotes)
./tests/run_tests [pytest args]     # pytest in the venv_columnar_dev sandbox, skips NAF-only tests
bash tests/run_test unit/test_util.py
./tests/run_naf                     # NAF-only tests (external files, law CLI)
./tests/run_all [force]             # linting + tests; "force" continues after a failure
```

The tests cover helpers, categorizers and selectors on synthetic events, config building and defaults, the ML and inference model registries, and the resolution of the main tasks (without running them); see [tests/README.md](../../tests/README.md).
They do not process real events, so still test a task change on the limited config (`l24v15`) with `--branch 0` before submitting jobs.
