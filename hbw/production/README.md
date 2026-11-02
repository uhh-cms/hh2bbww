# Production, categories and histogram weights

This README covers everything between `cf.ReduceEvents` and the histograms: the producers in this
folder and in [hbw/trigger](../trigger), the categorizers in [hbw/categorization](../categorization)
and the hist producers in [hbw/weight](../weight).
The relevant task chain is

```
cf.ReduceEvents -> cf.MergeReducedEvents -> cf.ProduceColumns (one per producer) -> cf.CreateHistograms -> cf.MergeHistograms -> cf.PlotVariables1D / cf.cms.CreateDatacards
```

The examples below use `c24v15` and its limited twin `l24v15`; see the [config README](../config/README.md#lazy-configs-and-hbwbuildcampaignsummary) for the support status of the configs.
The default law config selects the `hbw_dl` analysis and `c24v15`; the examples pass `--config` explicitly anyway.


## Producers

A producer is a function decorated with `@producer` (from `columnflow.production`). It receives an awkward
array `events` with the columns it declared in `uses`, adds new columns via `set_ak_column` and declares
them in `produces`. Other producers can be listed in `uses`/`produces`; they are then called as
`events = self[other_producer](events, **kwargs)` and their columns are included automatically.
Optional hooks are attached with decorators on the producer:

- `@<producer>.init`: runs once per config/dataset. Used to add `uses`/`produces` depending on dataset tags
  or year, and to add variables to the config (e.g. [ml_dih_inputs.py](ml_dih_inputs.py) calls `add_dl_ml_variables`).
- `@<producer>.post_init`: like `init`, but also knows the task (used e.g. to drop `_up`/`_down` columns for shifted tasks).
- `@<producer>.requires` / `@<producer>.setup`: declare extra upstream tasks and load their outputs
  (correction files via `BundleExternalFiles`, selection stats via `MergeSelectionStats`, ML scores via `MLEvaluation`).
- `@<producer>.skip`: skip the producer for some datasets (most weight producers also set `mc_only=True`).

Variants are created with `.derive("name", cls_dict={...})`. A `version` entry in the decorator or `cls_dict`
is part of the output path. Some versions are read from the law config: `event_weights_version`,
`dl_ml_inputs_version` and `cats_ml_version` in the `[analysis]` section.

### `cf.ProduceColumns` and combining producers

`cf.ProduceColumns` runs exactly **one** producer (`--producer`) on the merged reduced events of one dataset
and writes only the produced columns to a parquet file. A producer only sees the reduced events. It cannot
read columns written by another `ProduceColumns` output unless it calls that producer itself as a dependency.

Downstream tasks (`cf.CreateHistograms`, `cf.MLTraining`, plotting, datacards) take a list `--producers a,b,c`.
They require one `ProduceColumns` task per producer and merge all columns into the reduced events in that order.
If two producers write the same column, the later one wins. `pre_ml_cats` and `cats_ml_*` rely on this to
overwrite the `category_ids` written during selection.

### Default producers

If `--producers` is not given, the `default_producer` aux of the config is used. It is set by
[defaults_and_groups.py](../config/defaults_and_groups.py) (DL; HH and HHH call it from
[defaults_and_groups_hh.py](../config/defaults_and_groups_hh.py) / [defaults_and_groups_hhh.py](../config/defaults_and_groups_hhh.py))
and [sl_defaults_and_groups.py](../config/sl_defaults_and_groups.py) (SL). The function returns
`["event_weights", "pre_ml_cats", <ml_inputs>]`, but replaces `pre_ml_cats` by `cats_ml_<model>`, appended at
the end, when an ML model is in play (`--ml-models`, or the model of the inference model in `cf.cms.CreateDatacards`).
In the ML training pipeline (`cf.MLTraining` & co.) it drops the category producer entirely.

| analysis | ML-input producer | default hist producer |
| --- | --- | --- |
| SL (`hbw_sl`) | `sl_ml_inputs` | `default` |
| DL HH (`hbw_dl`) | `dl_ml_inputs` | `with_dy_corr` |
| DL HHH (`hbw_hhh_dl`) | `hhh_dl_ml_inputs` | `hhh_default` |

When you pass `--producers` explicitly, you replace this whole list. Include `event_weights` and
`pre_ml_cats` yourself if you need weights and categories.

### ML inputs

[ml_inputs.py](ml_inputs.py) defines `common_ml_inputs` (features shared by SL and DL) and `sl_ml_inputs`,
[ml_dih_inputs.py](ml_dih_inputs.py) defines `dl_ml_inputs` (plus `hh_bjets`), and
[ml_trih_inputs.py](ml_trih_inputs.py) defines `hhh_dl_ml_inputs`. All ML inputs are prefixed `mli_`. In `init`,
each producer bookkeeps its columns in `config_inst.x.ml_input_columns`, adds a config variable for each of them
and raises if a column has no variable (`check_variable_existence`). Object collections such as `Lepton`,
`VBFCandidateJet`, `FatBjet` and a 4-vector MET are built on the fly by `prepare_objects` in
[prepare_objects.py](prepare_objects.py). That producer is used almost everywhere and stores nothing itself.


## Event weights

Weights are built in two stages.

**In `cf.SelectEvents`** ([selection/common.py](../selection/common.py)): `mc_weight`, `large_weights_killer` and
`event_weights_to_normalize` ([weights.py](weights.py)) compute the pileup, parton-shower, scale (mur/muf), PDF
and, before 2024, b-tag weights. The selection stats store their per-process sums before the event selection
(only the base mask `no_sel_mask` is applied, see [selection/stats.py](../selection/stats.py)).

**In `cf.ProduceColumns`**, the `event_weights` producer ([weights.py](weights.py)) produces:

- `stitched_normalization_weight` via `combined_normalization_weights`, which picks one normalization per dataset:
  `stitched_normalization_weights_brs_from_processes` for `is_hbv` signal datasets, `stitched_normalization_weights`
  for `dy_*_m50*` and `w_lnu*_amcatnlo`, and plain `normalization_weights` otherwise. It also produces
  `dataset_normalization_weight` ([dataset_normalization.py](dataset_normalization.py)): lumi * xsec / sum of weights, without stitching.
- `top_pt_theory_weight` (`is_ttbar` datasets) and `vjets_weight` (`is_v_jets`).
- b-tag: `btag_wp_weights` for 2024 and later, `normalized_btag_weights` ([normalized_btag.py](normalized_btag.py)) for 2022/2023.
- lepton SFs: `electron_weight`, `electron_reco_weight`, `muon_id_weight`, `muon_iso_weight` (2024 additionally `muon_low_pt_id_weight`).
- `trigger_weight`: `dl_trigger_weights` ([trigger.py](trigger.py)) evaluates custom trigger-SF JSONs per channel
  (ee, mumu, emu). SL configs use `sl_trigger_weights` instead, which only produces `muon_trigger_weight`.
- `normalized_{pu,isr,fsr,mur,muf,murmuf_envelope,pdf}_weight`: built by `normalized_weight_factory`
  ([normalized_weights.py](normalized_weights.py)). Each is multiplied by sum(mc_weight) / sum(mc_weight * w),
  taken per process id from the selection stats, so the weight does not change the yield before selection.

Dataset or config tags switch parts off: `skip_btag_weights`, `skip_electron_weights`, `skip_muon_weights`,
`skip_trigger_weights`, `skip_scale`, `skip_pdf`, `no_ps_weights`.

Two weights live outside `event_weights`. `dy_correction_weight` ([dy_correction_weight.py](dy_correction_weight.py), DY only) reads a
correction fitted by `hbw.ExportDYWeights` ([tasks/dy_corrections.py](../tasks/dy_corrections.py)) from
`with_trigger_weight` histograms in `dycr__2mu`. `nlo_dy_reweight_weight` ([nlo_reweighting.py](nlo_reweighting.py), DY only, ONNX sandbox) is the second one.

### Hist producers: which weights are multiplied

A hist producer decides the per-event weight used to fill histograms. Almost all of ours derive from `base` in
[weight/default.py](../weight/default.py), which multiplies every column in its `weight_columns` dict. Data always gets weight 1.
Columns that do not apply to a dataset are dropped automatically (e.g. `top_pt_theory_weight` for non-ttbar,
`dy_correction_weight` for non-DY, scale/PDF/PS weights for datasets with the skip tags). Further `cls_dict` options:

- `categorizer_cls`: an extra event mask applied before filling. This is how the subleading-lepton cut
  `mask_fn_lep2_pt15` (pT > 15 GeV) enters all DL-HH hist producers.
- `dy_correction_weight_producer`: if `dy_correction_weight` is a weight column, the hist producer itself requires
  `ProduceColumns` with this producer. You do not add it to `--producers`.
- `nondy_hist_producer`: non-DY datasets reuse the histograms of this hist producer. `with_dy_corr` sets it to
  `with_trigger_weight`, and `hbw_parts` in [create_analysis.py](../analysis/create_analysis.py) rewrites the output path accordingly.
- `pre_label`: text prepended to all category labels in plots.

The main ones:

| hist producer | file | weights |
| --- | --- | --- |
| `default` | [hp_dih.py](../weight/hp_dih.py) | stitched norm, DY corr., trigger, b-tag, PU, lepton SFs, scale, PDF, ISR/FSR, top pT |
| `with_trigger_weight` | [hp_dih.py](../weight/hp_dih.py) | as `default` without DY corr., plus `mask_fn_lep2_pt15` |
| `with_dy_corr` | [hp_dih.py](../weight/hp_dih.py) | `with_trigger_weight` + `dy_correction_weight` (DL-HH default) |
| `no_trig_sf`, `dl_orth2_with_l1_seeds` | [hp_dih.py](../weight/hp_dih.py) | without trigger SF, used by `hbw.ComputeTriggerSF` |
| `hhh_default`, `hhh_shape` | [hp_trih.py](../weight/hp_trih.py) | HHH set with `mask_fn_ar`; `hhh_shape` adds b-tag shifts |
| `mc_weight`, `stitched_norm`, `unstitched_norm`, `no_weights`, ... | [default.py](../weight/default.py) | single columns, for checks |

[hp_dih.py](../weight/hp_dih.py) contains many more derived variants for studies, e.g. with other categorizers or weight sets.
`unstitched` is derived in both `hp_dih.py` and `hp_trih.py`. Lookup by name returns the later-loaded `hp_trih.py` version.

**Shifts.** Each value in `weight_columns` lists the shift sources of that weight, e.g.
`"muon_id_weight": ["mu_id_sf"]` or `"trigger_weight": ["trigger_sf"]` (`{year}`/`{cpn_tag}` are formatted in).
The hist producer registers the corresponding `*_up`/`*_down` shifts. For such a shift,
`cf.CreateHistograms` reads the column alias (e.g. `muon_id_weight_up`) instead of the nominal column. No upstream
task is re-run. The aliases are defined next to the shifts in [config_run2.py](../config/config_run2.py) via
`add_shift_aliases`. The hist producer raises if a listed shift has no alias for its column.


## Categories

There are two levels:

- **Categorizers** (`@categorizer`, [categorization/categories.py](../categorization/categories.py)) return
  `(events, mask)`. Categories in the config reference them by name, e.g. `selection="catid_2mu"`.
- **Categories** are `order` objects created in [config/categories.py](../config/categories.py). The columnflow
  producer `category_ids` evaluates the categorizers of every leaf category and stores the matching ids in the
  `category_ids` column. Histograms have a category axis, and a parent category is the sum of its leaves.

During `cf.SelectEvents`, only the basic categories exist (`add_categories_selection`): `incl`, `sr`/`dycr`/`ttcr`
for DL (`sr`/`fake` for SL) and the lepton channels (`2e`, `2mu`, `emu`, `ge3lep` for DL). In `cf.ProduceColumns`:

- `pre_ml_cats` ([categories.py](categories.py)) calls `add_categories_production`. That adds `resolved`/`boosted` and the
  b-jet categories (`1b`, `2b` for HH; `2b`, `3b`, `4b` for HHH), then builds all combinations of the blocks
  main x lepton x jet x b. Names are joined with `__` (e.g. `sr__2mu__resolved__2b`), and the id is the sum of the root ids.
- `cats_ml_<model>` is derived automatically for every `MLClassifierBase` subclass. It requires `MLEvaluation`, adds
  one `ml_<node>` category per training node (categorizer `catid_ml_<node>`) and combines `sr` x jet x b x DNN node.
  It also produces `mlscore.max_score`. `pre_ml_cats` and `cats_ml_*` cannot be used in the same task.

[masks_dih.py](../categorization/masks_dih.py), [masks_trih.py](../categorization/masks_trih.py) and
[trigger/trigger_cats.py](../trigger/trigger_cats.py) contain `mask_fn_*` categorizers. These are not used as
categories but as `categorizer_cls` of hist producers.


## Recipes

```shell
# run a single producer on one file of one dataset
law run cf.ProduceColumns --version v1 --config l24v15 --dataset tt_dl_powheg --producer event_weights --branch 0

# histograms with the default producers (event_weights, pre_ml_cats, dl_ml_inputs) and default hist producer
law run cf.CreateHistograms --version v1 --config l24v15 --dataset tt_dl_powheg --variables mli_mll,jet0_pt --branch 0

# plots: process group "dl1", category group "dl_sr", explicit producers and hist producer
law run cf.PlotVariables1D --version v1 --config c24v15 \
    --processes dl1 --categories dl_sr --variables mli_mll,jet0_pt \
    --producers event_weights,pre_ml_cats,dl_ml_inputs \
    --hist-producer with_trigger_weight

# same plot with a weight shift (no reprocessing, only aliased weight columns)
law run cf.PlotShiftedVariables1D --version v1 --config c24v15 \
    --processes tt --datasets tt_dl_powheg --categories sr --variables mli_mll \
    --shift-sources mu_id_sf
```

The DL-HH default `with_dy_corr` needs the DY correction chain for DY datasets. The first plot that includes DY
therefore runs `hbw.ExportDYWeights` and the `with_trigger_weight` histograms of all major processes. Use
`--hist-producer with_trigger_weight` if you don't need the correction.

**Add a new column/variable.**
1. Write a producer in a module listed in `production_modules` in [law.cfg](../../law.cfg), or add your module there. Unlisted modules are silently not loaded.
2. Declare all input columns in `uses` (or call `prepare_objects` for `Lepton` & co.) and all outputs in `produces`.
3. Add a config variable, either in [config/variables.py](../config/variables.py) / [config/dl/variables.py](../config/dl/variables.py),
   or in the producer's `init` guarded by `config_inst.has_variable(...)` (see `transformer_init` in [transformer.py](transformer.py)).
4. Run with `--producers event_weights,pre_ml_cats,dl_ml_inputs,<your_producer>`.

If a variable only needs columns that are already there, an expression is enough and you need no producer:
`expression="Jet.pt[:,0]"`, or a lambda with `aux={"inputs": {...}}`.

**Add a weight.**
1. Produce `my_weight` plus `my_weight_up`/`my_weight_down`, either inside `event_weights` (then bump
   `event_weights_version`) or as a separate producer that you add to `--producers`.
2. In [config_run2.py](../config/config_run2.py): `cfg.add_shift(name="my_sf_up", id=<free id>, type="shape")` (same for `down`)
   and `add_shift_aliases(cfg, "my_sf", {"my_weight": "my_weight_{direction}"})`.
3. Derive a hist producer, e.g. in [hp_dih.py](../weight/hp_dih.py):
   `default_hist_producer.derive("with_my_weight", cls_dict={"weight_columns": {..., "my_weight": ["my_sf"]}})`.
4. Use `--hist-producer with_my_weight`.


## Files

Modules not listed in [law.cfg](../../law.cfg) (`production_modules`, `categorization_modules`,
`hist_production_modules`) are marked *not loaded by default*. Some of them are still imported by other modules,
which is noted below. `hbw.ml.stats` (the `prepml*` ML-preprocessing producers) is also registered as a production module.

### hbw/production

| file | content |
| --- | --- |
| [weights.py](weights.py) | `event_weights`, `event_weights_to_normalize`, `large_weights_killer`, normalization and lepton-SF wrappers |
| [normalized_weights.py](normalized_weights.py) | `normalized_weight_factory` (normalization via selection stats); imported by `weights.py` |
| [normalized_btag.py](normalized_btag.py) | `normalized_btag_weights` (pre-2024); imported by `weights.py` |
| [dataset_normalization.py](dataset_normalization.py) | `dataset_normalization_weight`; imported by `weights.py` |
| [trigger.py](trigger.py) | `dl_trigger_weights`, `sl_trigger_weights`; imported by `weights.py` |
| [top_pt_theory.py](top_pt_theory.py) | `top_pt_theory_weight`, `gen_parton_top`; imported by `weights.py` and the reducer |
| [gen_v.py](gen_v.py) | `gen_v_boson`, `vjets_weight`; imported by `weights.py` and the reducer |
| [dy_correction_weight.py](dy_correction_weight.py) | `dy_correction_weight` (+ `dy_incl_corr_weight`, `dy_weight_test`) |
| [nlo_reweighting.py](nlo_reweighting.py) | `nlo_reweighting`: DY NLO reweighting with an ONNX model |
| [categories.py](categories.py) | `pre_ml_cats`, `cats_ml` and the derived `cats_ml_<model>` |
| [ml_inputs.py](ml_inputs.py) | `common_ml_inputs`, `sl_ml_inputs`, `prepare_bjets`, `vbf_jets`, `METCorr` |
| [ml_dih_inputs.py](ml_dih_inputs.py) | `dl_ml_inputs`, `hh_bjets` |
| [ml_trih_inputs.py](ml_trih_inputs.py) | `hhh_dl_ml_inputs` |
| [prepare_objects.py](prepare_objects.py) | `prepare_objects`, `prepare_hhh_bjets`; imported everywhere |
| [met.py](met.py) | `met_phi_prod`: MET-phi correction and `RecoilCorrMET` |
| [neutrino.py](neutrino.py) | `neutrino_reconstruction`, `top_reconstruction` (SL, W-mass constraint) |
| [gen_hbv_decay.py](gen_hbv_decay.py) | `gen_hbv_decay`, `gen_hbw_decay_features`: gen-level HH decay products |
| [transformer.py](transformer.py) | `transformer`: evaluates an external ONNX transformer, produces `transformerScore*` |
| [synchronization.py](synchronization.py) | `sync_exercise`: column dump for synchronization |
| [process_ids.py](process_ids.py) | `hbw_process_ids` and sub-process producers; not loaded by default, imported by the selection |
| [jets.py](jets.py) | `jetId_v12`, `fatjetId_v12`, `vbf_candidates`, `njet_for_recoil`; not loaded by default, imported by selection/reducer |
| [electrons.py](electrons.py) | `electron_ee_veto`; not loaded by default, imported by the selection |
| [filter.py](filter.py) | `ECALBadCalibrationFilter`; not loaded by default, imported by the selection |
| [resonant_features.py](resonant_features.py) | `resonant_features` (SL resonant); not loaded by default |
| [studies.py](studies.py) | `U_vectorV3`; not loaded by default |

### hbw/trigger

| file | content |
| --- | --- |
| [trigger_prod.py](../trigger/trigger_prod.py) | `trigger_prod_{sl,sl_ids,dl,dls,db}`: `trig_ids` column for trigger studies; `trigger_check` (debugging) |
| [trigger_cats.py](../trigger/trigger_cats.py) | `mask_fn_*` for orthogonal-trigger selections; not loaded by default, imported by `hp_dih.py` |
| [trigger_util.py](../trigger/trigger_util.py) | efficiency/SF helpers and binning optimisation used by `hbw.ComputeTriggerSF` ([trigger_sf.py](../tasks/trigger_sf.py)) |
| [plot_efficiencies.py](../trigger/plot_efficiencies.py), [plot_efficiencies2d.py](../trigger/plot_efficiencies2d.py) | trigger efficiency plot functions |

### hbw/categorization and hbw/weight

| file | content |
| --- | --- |
| [categorization/categories.py](../categorization/categories.py) | `catid_*` categorizers (leptons, mll regions, b-jets, resolved/boosted, ML nodes, triggers) |
| [categorization/masks_dih.py](../categorization/masks_dih.py) | `mask_fn_*` cuts for DL-HH hist producers (`mask_fn_lep2_pt15`, `mask_fn_met70`, ...) |
| [categorization/masks_trih.py](../categorization/masks_trih.py) | `mask_fn_sr`, `mask_fn_ar`, `mask_fn_lep2_pt15` for HHH |
| [weight/default.py](../weight/default.py) | `base` hist producer and single-column ones |
| [weight/hp_dih.py](../weight/hp_dih.py) | `default`, `with_trigger_weight`, `with_dy_corr` and many variants |
| [weight/hp_trih.py](../weight/hp_trih.py) | `hhh_default`, `hhh_shape`, `unstitched` |
