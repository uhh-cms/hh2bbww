# coding: utf-8
"""
A script to plot the data yields for different years and categories.

Loads per-dataset "muon_pt" histograms (columnflow cf.MergeHistograms output,
pickled hist.Hist objects), sums them per run period across both the
data_mu and data_egamma streams, and produces a yield-comparison plot
(2024 vs 2025) with a ratio panel, plus a printed summary table of yields
(combined and per-stream).
"""

import json
import os
import pickle

import hist
import matplotlib.pyplot as plt
import mplhep as hep  # HEP (CMS) extensions/styling on top of mpl
import numpy as np
import pandas as pd
import yaml

hep.style.use("CMS")

DATASET_YAML = "/data/dust/user/markusla/analysis/dilepton/hh2bbww/hbw/scripts/c24v15.yaml"
MUON_PT_PICKLE_BASE = (
    "/data/dust/user/matthiej/mttbar/data/mtt_store/new_analysis_mtt/"
    "cf.MergeHistograms/run3_mtt_{year}_nano_v15_new/{dataset}/nominal/"
    "calib__default_nominal/sel__ttbar_res_sel/red__default/"
    "prod__add_prod_cats__ttbar__features__61b14d821e/"
    "hist__no_trigger_weights/ml__v4_2425__9f70fb602c/260804_v8/"
    "hist__var_muon_pt.pickle"
)
# /data/dust/user/markusla/analysis/dilepton/hh2bbww/data/cf_store/hbw_hhh_dl/cf.MergeSelectionStats/c24v15

# years = [
#     "l24v15",
#     "l25v15",
# ]
years = [
    "c24v15",
    "c25v15",
]

# Both data streams get combined into the total "Data" yield.
DATA_STREAMS = ["data_mu", "data_egamma", "data_muoneg"]
# DATA_STREAMS = ["data_muoneg", "data_egamma"]

# Category to select on the "category" axis (StrCategory, from
# add_prod_cats). "incl" is the inclusive bin.
CATEGORY = "incl"
SHIFT = "nominal"

# SELSTATS_BASE = (
#     "/data/dust/user/matthiej/mttbar/data/mtt_store/new_analysis_mtt/"
#     "cf.MergeSelectionStats/run3_mtt_{year}_nano_v15_new/{dataset}/nominal/"
#     "calib__default_nominal/sel__ttbar_res_sel/260804_v8/stats.json"
# )
# SELSTATS_BASE = (
# "/data/dust/user/letzerba/public/hh2bbww/data/common_store/hbw_merged/calib__ak4V1__ak8V0__eleV0__c3633df749/sel__dl1V0/{year}/cf.MergeSelectionStats/nominal/{dataset}/stats.json"
# )
# SELSTATS_BASE = (
#     "/data/dust/user/markusla/public/hh2bbww/data/common_store/hbw_merged/calib__ak4V1__ak8V0__eleV0__c3633df749/sel__dl1V2/{year}/cf.MergeSelectionStats/nominal/{dataset}/stats.json"
# )
SELSTATS_BASE = (
    "/data/dust/user/markusla/public/hh2bbww/data/common_store/hbw_merged/calib__ak4V1__ak8V0__eleV0__c3633df749/sel__dl1V2/{year}/cf.MergeSelectionStats/nominal/{dataset}/stats.json"  # noqa: E501
)

# If True, the printed cutflow table (and its year-over-year ratios) merge
# data_mu + data_egamma into a single "data" column per year. If False, each
# stream gets its own column so you can see whether a difference between
# years is stream-specific (e.g. a trigger/JSON change hitting only one
# stream) or common to both.
MERGE_STREAMS_FOR_STATS = False

PLOT_DIR = "workflow_scripts/plots/muon_pt_yields"
os.makedirs(PLOT_DIR, exist_ok=True)


def check_availability(dataset_cfg):
    """
    Print a found/missing summary for both the muon_pt pickles and the
    selection-stats jsons, per year and stream, before doing any real work.
    Nothing here is fatal -- it's purely so incomplete inputs (jobs still
    running, etc.) are obvious up front instead of showing up as silent NaNs
    or a crash later on.
    """
    print("Input availability:")
    for label, base_path in (("muon_pt hist", MUON_PT_PICKLE_BASE), ("stats.json", SELSTATS_BASE)):
        for year in years:
            for stream in DATA_STREAMS:
                datasets = get_datasets_for_year(dataset_cfg, stream, year)
                missing = [d for d in datasets if not os.path.exists(
                    base_path.format(year=year, dataset=d))]
                found = len(datasets) - len(missing)
                status = "OK" if not missing else f"missing {missing}"
                print(
                    f"  [{label}] {year} {stream}: {found}/{len(datasets)} found ({status})")
    print()


def load_dataset_config():
    with open(DATASET_YAML, "r") as f:
        return yaml.safe_load(f)


def get_datasets_for_year(dataset_cfg, group_name, year):
    """
    Look up the list of dataset names for `group_name` (e.g. "data_mu")
    and `year` (e.g. "2024") from the parsed datasets.yaml, using the
    "{year}full" era key.
    """
    group = dataset_cfg[group_name]
    era_key = f"{year}"
    eras = group.get("eras", {})
    if era_key not in eras:
        raise KeyError(
            f"No era '{era_key}' found for group '{group_name}'. "
            f"Available eras: {list(eras.keys())}",
        )
    return eras[era_key]


def load_histogram(year, dataset):
    """
    Load a single dataset's pickled hist.Hist for the muon_pt variable.
    Axes: category (StrCategory), process (StrCategory, single value e.g.
    'data_mu'), shift (StrCategory), muon_pt (Variable), storage=Weight().
    """
    path = MUON_PT_PICKLE_BASE.format(year=year, dataset=dataset)
    if not os.path.exists(path):
        print(
            f"  [warn] missing pickle for dataset '{dataset}' ({year}): {path}")
        return None

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        if "muon_pt" in obj:
            h = obj["muon_pt"]
        elif len(obj) == 1:
            h = next(iter(obj.values()))
        else:
            raise KeyError(
                f"Cannot determine which histogram to use in {path}: "
                f"available keys are {list(obj.keys())}",
            )
    else:
        h = obj

    return h


def select_1d(h):
    """
    Reduce the 4D (category, process, shift, muon_pt) histogram down to a
    1D muon_pt histogram by selecting CATEGORY and SHIFT, and summing over
    the process axis. Each per-dataset pickle only ever carries a single
    process value (e.g. 'data_mu' or 'data_e' -- the exact label varies and
    isn't worth depending on), so summing over it is equivalent to
    selecting it but doesn't require knowing the label in advance.
    """
    return h[{
        "category": hist.loc(CATEGORY),
        "process": sum,
        "shift": hist.loc(SHIFT),
    }]


def sum_stream_histogram(dataset_cfg, group_name, year):
    """
    Sum the 1D muon_pt histogram over all datasets in one data stream/year.
    Returns None (with a warning) if no pickle could be loaded at all, e.g.
    because that stream/year's jobs haven't produced output yet.
    """
    datasets = get_datasets_for_year(dataset_cfg, group_name, year)
    print(f"[{year}] {group_name}: summing {len(datasets)} datasets: {datasets}")

    total = None
    for dataset in datasets:
        h = load_histogram(year, dataset)
        if h is None:
            continue
        h1d = select_1d(h)
        total = h1d.copy() if total is None else total + h1d

    if total is None:
        print(
            f"  [warn] no muon_pt histograms available yet for {group_name} / {year} -- skipping")

    return total


def sum_year_histogram(dataset_cfg, year):
    """
    Combine all DATA_STREAMS into one total-data muon_pt histogram for
    `year`. Streams with no available data are dropped; returns
    (None, {}) if nothing was available for this year at all.
    """
    stream_hists = {}
    for stream in DATA_STREAMS:
        h = sum_stream_histogram(dataset_cfg, stream, year)
        if h is not None:
            stream_hists[stream] = h

    total = None
    for h in stream_hists.values():
        total = h.copy() if total is None else total + h
    return total, stream_hists


def load_stats(year, dataset):
    """Load one dataset's cf.MergeSelectionStats stats.json."""
    path = SELSTATS_BASE.format(year=year, dataset=dataset)
    if not os.path.exists(path):
        print(
            f"  [warn] missing stats for dataset '{dataset}' ({year}): {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def sum_stats_dicts(dicts):
    """
    Recursively sum a list of stats dicts, key by key. Handles the flat
    "num_events..." counters as well as the nested "*_per_process" dicts
    (e.g. {"10": 315211957}) by summing per sub-key.
    """
    dicts = [d for d in dicts if d is not None]
    if not dicts:
        return {}

    total = {}
    for key in dicts[0]:
        values = [d[key] for d in dicts if key in d]
        if isinstance(values[0], dict):
            total[key] = sum_stats_dicts(values)
        else:
            total[key] = float(sum(values))
    return total


def sum_stream_stats(dataset_cfg, group_name, year):
    """Sum stats.json over all datasets in one data stream/year."""
    datasets = get_datasets_for_year(dataset_cfg, group_name, year)
    stats_list = [load_stats(year, dataset) for dataset in datasets]
    return sum_stats_dicts(stats_list)


def get_step_order(stats_by_column):
    """
    Preserve the order in which "num_events_step_*" keys first appear
    across all loaded stats dicts (dict insertion order from json.load
    reflects the order columnflow's selector wrote them in).
    """
    step_keys = []
    seen = set()
    for stats in stats_by_column.values():
        for key in stats:
            if key.startswith("num_events_step_") and key not in seen:
                step_keys.append(key)
                seen.add(key)
    return step_keys


def build_cutflow_tables(year_stream_stats):
    """
    Build (counts, efficiency, year-over-year efficiency ratio) DataFrames
    from {year: {stream: stats_dict}}. Columns are combined into a single
    "data" stream per year if MERGE_STREAMS_FOR_STATS is True, otherwise
    kept per-stream so differences can be traced to one trigger stream.

    "efficiency" here is each step's count divided by that column's total
    "num_events" -- i.e. relative to the full processed sample, not a
    sequential/step-over-step efficiency, since the selection steps aren't
    all part of one linear chain (e.g. "num_events_step_all_but_bjet" is an
    alternate branch, not a continuation of "num_events_step_BJet").
    """
    columns = {}
    for year in years:
        if MERGE_STREAMS_FOR_STATS:
            merged = sum_stats_dicts(list(year_stream_stats[year].values()))
            if merged:
                columns[(year, "data")] = merged
        else:
            for stream in DATA_STREAMS:
                stats = year_stream_stats[year][stream]
                if stats:
                    columns[(year, stream)] = stats

    step_keys = get_step_order(columns)
    row_keys = ["num_events"] + step_keys + ["num_events_selected"]

    counts = pd.DataFrame(
        {col: [stats.get(k, np.nan) for k in row_keys]
         for col, stats in columns.items()},
        index=row_keys,
    )
    counts.columns = pd.MultiIndex.from_tuples(
        counts.columns, names=["year", "stream"])

    efficiency = counts.div(counts.loc["num_events"], axis=1)

    ref_year = years[0]
    ratio_cols = {}
    streams_present = counts.columns.get_level_values("stream").unique()
    for year in years[1:]:
        for stream in streams_present:
            if (year, stream) not in efficiency.columns or (ref_year, stream) not in efficiency.columns:
                continue
            ratio_cols[(year, stream)] = (
                efficiency[(year, stream)] / efficiency[(ref_year, stream)]
            )
    ratio = pd.DataFrame(ratio_cols)
    if not ratio.empty:
        ratio.columns = pd.MultiIndex.from_tuples(
            ratio.columns, names=["year_vs_" + ref_year, "stream"],
        )

    return counts, efficiency, ratio


def plot_cutflow_comparison(year_stream_stats):
    """
    Grouped bar chart of per-step efficiency (relative to num_events) for
    each year, always combining streams for readability (the printed table
    is where the per-stream breakdown lives).
    """
    combined = {year: sum_stats_dicts(
        list(year_stream_stats[year].values())) for year in years}
    plot_years = [year for year in years if combined[year].get("num_events")]
    skipped = [year for year in years if year not in plot_years]
    if skipped:
        print(
            f"  [warn] no stats available yet for {skipped} -- omitting from cutflow plot")
    if not plot_years:
        print("  [warn] no stats available for any year -- skipping cutflow plot")
        return

    step_keys = get_step_order({y: combined[y] for y in plot_years})
    row_keys = ["num_events"] + step_keys + ["num_events_selected"]

    fig, ax = plt.subplots(figsize=(10, 0.35 * len(row_keys) + 1.5))
    n_years = len(plot_years)
    bar_height = 0.8 / n_years
    y_pos = np.arange(len(row_keys))
    colors = {"2024": "black", "2025": "crimson"}

    for i, year in enumerate(plot_years):
        num_events_total = combined[year]["num_events"]
        eff = [combined[year].get(k, np.nan) /
               num_events_total for k in row_keys]
        offset = (i - (n_years - 1) / 2) * bar_height
        ax.barh(
            y_pos + offset,
            eff,
            height=bar_height,
            label=year,
            color=colors.get(year),
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_keys, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of num_events")
    ax.set_title(f"Cutflow comparison ({'+'.join(DATA_STREAMS)})")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(
        PLOT_DIR, f"cutflow_comparison_{'_'.join(years)}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved cutflow comparison plot to {out_path}")


def main():
    dataset_cfg = load_dataset_config()
    check_availability(dataset_cfg)

    # year_hists = {}
    # year_stream_hists = {}
    # for year in years:
    #     total, stream_hists = sum_year_histogram(dataset_cfg, year)
    #     year_stream_hists[year] = stream_hists
    #     if total is not None:
    #         year_hists[year] = total

    # plot_years = [year for year in years if year in year_hists]

    # --- summary table: combined + per-stream yields ----------------------
    # rows = []
    # for year in years:
    #     row = {"year": year}
    #     row["total_data"] = float(year_hists[year].sum().value) if year in year_hists else np.nan
    #     for stream in DATA_STREAMS:
    #         row[stream] = (
    #             float(year_stream_hists[year][stream].sum().value)
    #             if stream in year_stream_hists[year]
    #             else np.nan
    #         )
    #     rows.append(row)
    # summary = pd.DataFrame(rows)
    # print("\nMuon_pt yields per year (category='%s'):" % CATEGORY)
    # print(summary.to_string(index=False))

    # # --- comparison plot with ratio panel ----------------------------------
    # if not plot_years:
    #     print("\n[warn] no muon_pt histograms available for any year -- skipping yield plot")
    # else:
    #     fig, (ax_main, ax_ratio) = plt.subplots(
    #         nrows=2,
    #         ncols=1,
    #         figsize=(8, 8),
    #         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    #         sharex=True,
    #     )

    #     colors = {"2024": "black", "2025": "crimson"}
    #     for year in plot_years:
    #         hep.histplot(
    #             year_hists[year],
    #             ax=ax_main,
    #             label=f"{year} (Data)",
    #             histtype="errorbar",
    #             color=colors.get(year),
    #         )

    #     ax_main.set_ylabel("Events")
    #     ax_main.set_yscale("log")
    #     ax_main.legend(title=f"{'+'.join(DATA_STREAMS)}, category={CATEGORY}")

    #     if len(plot_years) >= 2:
    #         ref_year, other_year = plot_years[0], plot_years[1]
    #         ref_vals = year_hists[ref_year].values()
    #         other_vals = year_hists[other_year].values()
    #         centers = year_hists[ref_year].axes[0].centers

    #         with np.errstate(divide="ignore", invalid="ignore"):
    #             ratio = np.where(ref_vals > 0, other_vals / ref_vals, np.nan)

    #         ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    #         ax_ratio.plot(centers, ratio, "o", color=colors.get(other_year), markersize=4)
    #         ax_ratio.set_ylabel(f"{other_year} / {ref_year}")
    #         ax_ratio.set_ylim(0, 2)
    #     else:
    #         ax_ratio.text(
    #             0.5, 0.5, f"ratio needs \u2265 2 years (have {plot_years})",
    #             ha="center", va="center", transform=ax_ratio.transAxes,
    #         )
    #     ax_ratio.set_xlabel(r"Muon $p_{T}$ [GeV]")

    #     hep.cms.label(ax=ax_main, data=True, label="Preliminary")

    #     out_path = os.path.join(PLOT_DIR, f"muon_pt_data_{'_'.join(plot_years)}.pdf")
    #     fig.savefig(out_path, bbox_inches="tight")
    #     print(f"\nSaved plot to {out_path}")

    # --- selection-stats cutflow: counts, efficiency, year-over-year ratio -
    year_stream_stats = {
        year: {
            stream: sum_stream_stats(dataset_cfg, stream, year) for stream in DATA_STREAMS
        }
        for year in years
    }

    counts, efficiency, ratio = build_cutflow_tables(year_stream_stats)

    print("\nCutflow counts:")
    print(counts.to_string(float_format=lambda v: f"{v:,.0f}"))

    print("\nCutflow efficiency (fraction of num_events):")
    print((efficiency * 100).to_string(float_format=lambda v: f"{v:.2f}%"))

    if not ratio.empty:
        print(
            f"\nEfficiency ratio vs {years[0]} (values far from 1.0 flag a year-to-year change):")
        print(ratio.to_string(float_format=lambda v: f"{v:.3f}"))

    plot_cutflow_comparison(year_stream_stats)


if __name__ == "__main__":
    main()
