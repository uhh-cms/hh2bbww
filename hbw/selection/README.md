# Calibration, selection and reduction

This folder, together with [calibration/](../calibration) and [reduction/](../reduction), defines the first
event-processing steps of the analysis. They turn NanoAOD into the reduced event files that every later
task (`cf.ProduceColumns`, ML, histograms, datacards) reads:

```
cf.CalibrateEvents  ->  cf.SelectEvents  ->  cf.ReduceEvents  ->  cf.MergeReductionStats / cf.MergeReducedEvents
  (calibrators)          (selector)  \         (reducer)
                                      ->  cf.MergeSelectionStats  (event counts and weight sums for normalization)
```

All three tasks run per dataset and per input file. Branch `i` of a task processes file `i` of the dataset.


## Concepts in columnflow terms

The physics code is written as *task array functions* (TAFs): Python functions that take an awkward array
`events` and are wrapped by a decorator. There are three kinds in this chain:

| Kind | Decorator | Returns | Used by |
| --- | --- | --- | --- |
| Calibrator | `@calibrator` | `events` with corrected columns (e.g. `Jet.pt` after JEC) | `cf.CalibrateEvents`, `--calibrators a,b,c` |
| Selector | `@selector` | `(events, SelectionResult)` | `cf.SelectEvents`, `--selector name` |
| Reducer | `@reducer` | `events`, already filtered | `cf.ReduceEvents`, `--reducer name` |

- **`uses` / `produces`** declare which columns a TAF reads and writes, e.g. `"Jet.{pt,eta}"`. Columns that are
  not in `uses` are never loaded, and columns that are not in `produces` are not saved. If you put another TAF
  in `uses`, it becomes a dependency that you call with `self[other](events, **kwargs)`. Always forward
  `**kwargs`.
- **Hooks.** `@my_selector.init` runs once per config and dataset and may add columns, dependencies or shifts.
  `post_init`, `requires` and `setup` hooks also exist (see [common.py](common.py) and
  [../calibration/default.py](../calibration/default.py) for examples).
- **`exposed=True`** marks a selector as a top-level selector that you can pass to `--selector`. The building
  blocks (`pre_selection`, `jet_selection`, ...) are not exposed.
- **`SelectionResult`** collects three things:
  - `steps`: named boolean event masks. You can plot them in a cutflow or apply a subset of them later.
  - `objects`: index arrays per collection. For example, `objects["Jet"]["Bjet"]` makes the reducer create a
    new `Bjet` collection out of these `Jet` entries.
  - `event`: the final event mask.

  Results combine with `+`. A fourth field, `aux`, holds arrays passed between sub-selectors and is not
  saved to disk.
- **Registration.** A module is only imported if it is listed in [law.cfg](../../law.cfg), under
  `calibration_modules`, `selection_modules` and `reduction_modules` in `[analysis]`. Here these are
  `hbw.calibration.default`, `hbw.selection.{jet,common,sl_remastered,dl_remastered}` and
  `hbw.reduction.default`. Other files in this folder are only active if one of these modules imports them.


## Calibration ([../calibration/default.py](../calibration/default.py))

The calibrators are kept separate so that each one writes its own output, and `cf.SelectEvents` reads all
of them. The default list is `ak4,ak8,ele,muo` (`default_calibrator` in
[../config/defaults_and_groups.py](../config/defaults_and_groups.py); it can be overridden with
`default_calibrators` in the law config).

- `deterministic_seeds_calibrator` produces reproducible random seeds per event and per Jet, FatJet, Electron
  and Muon. Every calibrator derived from `seeds_user_base` requires it automatically as a separate
  `cf.CalibrateEvents` run.
- `ak4` (derived from `jet_base`) applies, in order:
  - JEC with the `Total` uncertainty for MC (nominal only for data);
  - JER smearing for MC, with horn handling;
  - the MET phi correction (Run 2 only).

  The corrections are propagated to MET, and non-finite values are replaced afterwards. For 2024 data, the
  L2L3Residual correction is evaluated with jet pt clamped to 30 GeV for 2.0 < |eta| < 2.5
  (`jec_clamp_2024_data_l2l3residual`).
- `ak8` applies JEC and JER to `FatJet` and recomputes the softdrop mass. It raises an error for any
  non-nominal shift.
- `ele` applies electron scale and smearing; `muo` applies muon scale and resolution.
- Other `jet_base` variants: `jec_only`, `skip_jer`, `no_breg`, `with_b_reg` and `with_b_reg_test` toggle
  b-jet regression and JER. `ak4uncs` and `ak8uncs` set `jec_variations_only=True`, which currently raises
  `NotImplementedError`. `combined` runs seeds, `ak4`, `ak8` and `ele` in one task, but is not used by
  default.


## Selection

### Building blocks

`sl1` and `dl1` call the same sequence of building blocks:

1. `pre_selection` ([common.py](common.py)) prepares the events:
   - computes the jet ID, `mc_weight` and the `large_weights_killer`;
   - marks the hard-process gen particles (MC only, `hard_gen_particles` in [gen.py](gen.py));
   - sets `process_id`.

   It also defines the steps `good_vertex`, `met_filter` (patched with the ECAL bad-calibration filter),
   `json` (golden JSON, data only) and `jet_veto_map`, and combines them into `cleanup`.
2. The lepton selection uses `lepton_definition` ([lepton.py](lepton.py)), which defines loose, fakeable and
   tight electrons and muons, and veto taus. It sets `Muon.is_tight` / `Electron.is_tight` and adds the steps
   `VetoTau`, `ll_lowmass_veto` and `ll_zmass_veto`. The channel-specific wrappers `sl_lepton_selection` and
   `dl_lepton_selection` then add the channel steps (`Lep_e`, `Lep_mm`, `SR`, `Fake`, ...).
3. The trigger selection differs between the channels:
   - DL uses `hbw_trigger_selection` ([trigger.py](trigger.py)). It matches HLT paths to trigger objects
     and L1 seeds and stores `trigger_ids`. It then checks that the fired trigger matches the lepton channel
     (`Trigger`, `TriggerAndLep`) and removes events that appear in more than one data stream
     (`data_double_counting`, using the stream order in `config.x.data_streams`).
   - SL builds its trigger steps directly in `sl_lepton_selection` from the `config.x.trigger` dict.
4. `jet_selection` ([jet.py](jet.py)) selects AK4 jets with pt >= `jet_pt` (default 25 GeV), tight ID and
   ΔR > 0.4 from the selected leptons. It defines the steps `nJet1`-`nJet4` and `nBjet1`/`nBjet2` (b-tag column
   and working point come from the config), and the objects `Jet`, `ForwardJet`, `Bjet` and `Lightjet`.
   `sl_boosted_jet_selection` / `dl_boosted_jet_selection` add the `HbbJet` steps and objects.
   `vbf_jet_selection` adds `VBFJet` and the `VBFJetPair` step.
5. `get_weights_and_no_sel_mask` ([common.py](common.py), [bad_events.py](bad_events.py)) computes the event
   weights that need normalization. It also defines `no_sel_mask`, which flags events with broken LHE or PDF
   weights; these events get `mc_weight = 0` and are removed from `cleanup`.
6. `post_selection` ([common.py](common.py)) runs the bookkeeping:
   - computes `category_ids`;
   - fills the stats via `hbw_selection_step_stats` and `hbw_increment_stats` ([stats.py](stats.py)): event
     counts and weight sums per step and per process;
   - fills the normalization histograms via `hbw_selection_hists` ([hists.py](hists.py)).

   `cf.MergeSelectionStats` merges these outputs, and the normalization weights are later computed from
   them.

If the config is limited to at most 10 files per dataset (the `l*` configs), `do_cutflow_features` is set.
The selectors then additionally write `cutflow.*` columns (object multiplicities) for cutflow variables.

### The `sl1` and `dl1` selectors

`dl1` ([dl_remastered.py](dl_remastered.py)) is the default for `is_dl` configs. It requires:

- `cleanup`;
- at least `n_jet` jets (default 1);
- exactly two tight, opposite-charge leptons above the pt thresholds (`Dilepton`, `SR`, `Charge`);
- a veto on a third loose lepton (`TripleLooseLeptonVeto`);
- m_ll >= 12 GeV (`DiLeptonMass12`);
- the trigger steps.

It combines these into the steps `all_but_trigger_and_bjet`, `all_but_bjet` and `all_but_trigger`. The final
step `all` (= `results.event`) adds at least `n_btag` b-jets (default 1). Several steps (`all`,
`all_but_bjet`, `Trigger`, ...) are also written as `steps.*` columns, so you can change the selection after
the reduction. For MC, `dl1` also fills b-tag efficiency histograms (`fill_btag_wp_count_hists`). Its triggers
come from `add_triggers` in [../config/trigger.py](../config/trigger.py).

`sl1` ([sl_remastered.py](sl_remastered.py)) is the default for `is_sl` configs. It requires exactly one lepton
(`Lepton`, `DileptonVeto`) and `VetoTau`, plus either the resolved selection (at least 3 jets and at least
1 b-jet) or a boosted `HbbJet`. Its trigger and lepton pt thresholds are only defined up to 2023: for other
years, `sl_lepton_selection_init` raises an error. `sl1` therefore does not run on `c24v15` as is.

Thresholds are class attributes (`mu_pt`, `ele_pt`, `mu2_pt`, `ele2_pt`, `jet_pt`, `n_jet`, `n_btag`, `trigger`,
`trigger_config_func`). `configure_selector` copies them into the config aux (`config.x.selector_config` and
`config.x.<name>`) the first time a selector is initialized for that config.

### Derived variants

| Selector | Differences from its parent |
| --- | --- |
| `dl1_no_btag` | `n_btag=0` (no b-jet requirement in `all`) |
| `dl1_sl_triggers` | keeps only `HLT_IsoMu24` and `HLT_Ele30_WPTight_Gsf`; lepton pt 25/31/15/15 GeV |
| `dl1_mu18` | lepton pt thresholds: mu 18, e 25, subleading 15/15 GeV |
| `dl1_low_lep` | all lepton pt thresholds 15 GeV, `version=1` |
| `dl1_no_trigger` | only `version=1`; the selection logic is the same as `dl1` |
| `dl1_test`, `test_dl` | copies of `dl1` for testing (`dl1_test` pins `version=3`) |
| `sl1_no_btag` | `n_btag=0`, `b_tagger="deepjet"` |
| `sl1_no_trig` | no triggers, all lepton pt thresholds 15 GeV |
| `sl1_test` | copy of `sl1` with `version=0` |

### Versions and output paths

The analysis patches columnflow ([../columnflow_patches.py](../columnflow_patches.py), `patch_csp_versioning`)
so that a calibrator, selector or reducer appears in the output path as `<name>V<version>`. For example:

- `sl1` has `version=1`, so it appears as `sl1V1`;
- `ak4` has `version=1`, so it appears as `ak4V1`;
- `dl1` reads its version from `dl1_version` in the `[analysis]` section of the active law config (fallback
  `3`).

Changing the `version` in a `cls_dict`, or in the law config, therefore makes the task write new outputs
without touching the old ones. For `cf.CalibrateEvents`, `cf.SelectEvents`, `cf.ReduceEvents` and their merge
tasks, the `--version` CLI parameter is removed from the path (`reorganize_parts` in
[../analysis/create_analysis.py](../analysis/create_analysis.py)). It has no effect on these outputs.

Two other rules in `hbw_parts` (same file) affect where outputs are stored:

- **Shared between analyses.** The outputs of `cf.CalibrateEvents`, `cf.SelectEvents`, `cf.ReduceEvents`,
  `cf.MergeSelectionStats`, `cf.MergeReducedEvents`, `cf.MergeReductionStats`, `cf.ProvideReducedEvents` and
  `cf.GetDatasetLFNs` are always stored under the analysis name `hbw_merged`. `hbw_dl`, `hbw_sl`,
  `hbw_hhh_dl`, ... therefore share them.
- **Shared between limited and full configs.** For `cf.CalibrateEvents`, `cf.SelectEvents`, `cf.ReduceEvents`
  and `cf.GetDatasetLFNs`, the limited config name is mapped to the full one (`l24v15` -> `c24v15`). Branches
  you run with `l24v15` are therefore reused by `c24v15`.

A dataset or shift with a `version` aux gets `V<n>` appended to its path part.


## Reduction ([../reduction/default.py](../reduction/default.py))

The `default` reducer runs columnflow's `cf_default`. It applies `results.event`, or the AND of
`--selector-steps` if you pass any, builds the object collections from `objects`, and keeps only the columns
listed in `cfg.x.keep_columns["cf.ReduceEvents"]` ([../config/config_run2.py](../config/config_run2.py)).

It then checks that no jet below the selector's `jet_pt` survived, and adds gen-level information on the
selected events:

- `gen_parton_top` for top samples;
- `gen_v_boson` for V+jets samples;
- `gen_dilepton` and the recoil-corrected MET for DY samples;
- `gen_hbv_decay` for all samples.

Variants:

- `triggersf` reduces with `all_but_trigger` and keeps the MET-trigger HLT/L1 columns needed for trigger
  scale factors.
- `noselection` keeps all events that pass `no_sel_mask`.
- `test_default`, `test_triggersf` and `test_noselection` are plain copies.

`cf.MergeReductionStats` (set to `n-inputs: -1` in [law.cfg](../../law.cfg), i.e. it uses all files)
computes how many reduced files are merged by `cf.MergeReducedEvents`.


## Recipes

The commands assume the DL law config (`hbw_dl` as default analysis, `c24v15` as default config). See the
[config README](../config/README.md#lazy-configs-and-hbwbuildcampaignsummary) for the support status of the configs.

```shell
# select one file of a dataset on the limited config (also runs the needed CalibrateEvents tasks)
law run cf.SelectEvents \
    --config l24v15 \
    --dataset tt_dl_powheg \
    --calibrators ak4,ak8,ele,muo \
    --selector dl1 \
    --branch 0

# reduce the same file; --selector-steps would apply a subset of steps instead of "all"
law run cf.ReduceEvents --config l24v15 --dataset tt_dl_powheg --branch 0

# show where the outputs are stored, then open them in IPython
law run cf.ReduceEvents --config l24v15 --dataset tt_dl_powheg --branch 0 --print-output 0
cf_inspect /path/to/events_0.parquet
```

`cf.SelectEvents` writes `results_<i>.parquet` (steps, objects, event mask), `columns_<i>.parquet` (new
columns), `stats_<i>.json` and `hists_<i>.pickle`. To check selection efficiencies and expected yields per
dataset from the merged stats:

```shell
law run hbw.SelectionSummary --version v1 --config l24v15 --datasets tt_dl_powheg,hh_ggf_hbb_hvv_kl1_kt1_powheg
```

To plot a cutflow, list the steps in the order they should be applied, because they are applied
cumulatively. The labels come from `config.x.selector_step_labels`, which the selector inits fill:

```shell
law run cf.PlotCutflow \
    --version v1 \
    --config l24v15 \
    --datasets tt_dl_powheg \
    --selector dl1 \
    --categories incl \
    --selector-steps cleanup,Dilepton,SR,Charge,TripleLooseLeptonVeto,DiLeptonMass12,Trigger,TriggerAndLep,nJet1,nBjet1
```

For a full dataset, drop `--branch` and add `--workflow htcondor`.

**Adding a selection step.** Inside a selector, add a mask such as
`results.steps["MyCut"] = <bool array per event>`. Every step is stored automatically and can be used in
cutflows and with `--selector-steps`. A step only affects the final selection if you also add it to the
`all*` combinations in `dl1` / `sl1`. Add any new input columns to `uses`, and give the step a label via
`self.config_inst.x.selector_step_labels` in the init.

**Deriving a selector variant.** Add a line at the end of [dl_remastered.py](dl_remastered.py):

```python
dl1_jet30 = dl1.derive("dl1_jet30", cls_dict={"jet_pt": 30., "version": 1})
```

Then run it with `--selector dl1_jet30`. The new name, and the `version` you set, give the variant its own
output path. Note that `configure_selector` writes the thresholds into the config aux only for the first
selector initialized for a config, so don't mix selectors with different thresholds in one law process.


## Files

| File | Content |
| --- | --- |
| [common.py](common.py) | `pre_selection`, `post_selection`, `get_weights_and_no_sel_mask`, `configure_selector`, `hbw_met_filters` |
| [sl_remastered.py](sl_remastered.py) | `sl1`, `sl_lepton_selection` and SL variants |
| [dl_remastered.py](dl_remastered.py) | `dl1`, `dl_lepton_selection` and DL variants |
| [lepton.py](lepton.py) | `lepton_definition`: loose, fakeable and tight leptons, tau veto, m_ll vetoes |
| [jet.py](jet.py) | `jet_selection`, `sl_boosted_jet_selection` / `dl_boosted_jet_selection`, `vbf_jet_selection` |
| [trigger.py](trigger.py) | `hbw_trigger_selection`: trigger matching, trigger-lepton consistency, data double counting |
| [bad_events.py](bad_events.py) | `no_sel_mask` for events with broken LHE or PDF weights |
| [gen.py](gen.py) | `hard_gen_particles` (MC only) |
| [stats.py](stats.py) | event counts and weight sums per step and process for `MergeSelectionStats` |
| [hists.py](hists.py) | normalization histograms (per process, and in ht/njet/nhf bins) |
| [cutflow_features.py](cutflow_features.py) | legacy `cutflow_features` producer, not imported anywhere |
| [gen_hbw_features.py](gen_hbw_features.py) | legacy gen-matching producers, not imported anywhere |
| [../calibration/default.py](../calibration/default.py) | all calibrators (`ak4`, `ak8`, `ele`, `muo`, seeds, variants) |
| [../reduction/default.py](../reduction/default.py) | `default`, `triggersf` and `noselection` reducers |
