#!/usr/bin/env bash
# Extract dataset names for a given era (optionally restricted to one
# dataset key) from a datasets YAML file.
# Pure bash/awk — no yq or python-yaml dependency required.
#
# Usage: ./get_datasets.sh <yaml_file> [era] [dataset_key]
#   era defaults to "2024full"
#   dataset_key restricts to a single top-level key (e.g. "data_egamma");
#   omit to include all dataset keys.
#
# Assumes the YAML follows the structure:
#   <dataset>:
#       eras:
#           <era_name>:
#               - <item>
#               - <item>
 
set -euo pipefail
 
YAML_FILE="${1:?Usage: $0 <yaml_file> [era] [dataset_key]}"
ERA="${2:-2024full}"
DATASET_KEY="${3:-}"
 
[[ -f "$YAML_FILE" ]] || { echo "Error: file not found: $YAML_FILE" >&2; exit 1; }
 
mapfile -t DATASETS < <(
    awk -v era="$ERA" -v key="$DATASET_KEY" '
        function indent(line,    s) {
            s = line
            sub(/[^ ].*$/, "", s)
            return length(s)
        }
        function trim(line,    s) {
            s = line
            sub(/^[ \t]+/, "", s)
            sub(/[ \t]+$/, "", s)
            return s
        }
        {
            line = $0
            t = trim(line)
 
            # Track which top-level dataset key (indent 0) we are under
            if (indent(line) == 0 && t ~ /:$/) {
                top_key = t
                sub(/:$/, "", top_key)
                in_era = 0
            }
 
            key_ok = (key == "" || key == top_key)
 
            if (in_era) {
                # blank or comment lines do not break the block
                if (t == "" || t ~ /^#/) next
 
                cur_indent = indent(line)
                if (cur_indent <= era_indent) {
                    in_era = 0
                } else if (t ~ /^- /) {
                    val = t
                    sub(/^- /, "", val)
                    print val
                    next
                }
            }
 
            if (!in_era && key_ok && t == era ":") {
                in_era = 1
                era_indent = indent(line)
            }
        }
    ' "$YAML_FILE"
)
 
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    if [[ -n "$DATASET_KEY" ]]; then
        echo "No datasets found for era '${ERA}' under key '${DATASET_KEY}' in ${YAML_FILE}" >&2
    else
        echo "No datasets found for era '${ERA}' in ${YAML_FILE}" >&2
    fi
    exit 1
fi

echo "================================"

case "$ERA" in
    2024full)
        echo "Running find_empty_files.zsh for era 2024full"
        config_name="c24v15"
        ;;
    2025full)
        echo "Running find_empty_files.zsh for era 2025full"
        config_name="run3_mtt_2025_nano_v15_new"
        ;;
    2026full)
        echo "Running find_empty_files.zsh for era 2026full"
        config_name="run3_mtt_2026_nano_v15_new"
        ;;
    *)
        echo "Warning: unknown era '$ERA', proceeding anyway" >&2
        ;;
esac

# Loop over the datasets
FAILED=()
for dataset in "${DATASETS[@]}"; do
    echo "$dataset"
    # do something with "$dataset" here
    if ! ./find_empty_files.zsh "/pnfs/desy.de/cms/tier2/store/user/bletzer/hbw_store_v1/hbw_merged/c24v15/cf.CalibrateEvents/calib__ak4V1/nominal/${dataset}/" --delete --gfal ; then
        echo "Warning: find_empty_files.zsh failed for ${dataset}" >&2
        FAILED+=("$dataset")
    fi
    echo "--------------------------------"
    if ! ./find_empty_files.zsh "/pnfs/desy.de/cms/tier2/store/user/bletzer/hbw_store_v1/hbw_merged/c24v15/cf.CalibrateEvents/calib__ak8V0/nominal/${dataset}/" --delete --gfal ; then
        echo "Warning: find_empty_files.zsh failed for ${dataset}" >&2
        FAILED+=("$dataset")
    fi
    echo "--------------------------------"
    if ! ./find_empty_files.zsh "/pnfs/desy.de/cms/tier2/store/user/bletzer/hbw_store_v1/hbw_merged/c24v15/cf.CalibrateEvents/calib__deterministic_seeds_calibrator/nominal/${dataset}/" --delete --gfal ; then
        echo "Warning: find_empty_files.zsh failed for ${dataset}" >&2
        FAILED+=("$dataset")
    fi
    echo "--------------------------------"
    if ! ./find_empty_files.zsh "/pnfs/desy.de/cms/tier2/store/user/bletzer/hbw_store_v1/hbw_merged/c24v15/cf.CalibrateEvents/calib__eleV0/nominal/${dataset}/" --delete --gfal ; then
        echo "Warning: find_empty_files.zsh failed for ${dataset}" >&2
        FAILED+=("$dataset")
    fi
    echo "--------------------------------"
    if ! ./find_empty_files.zsh "/pnfs/desy.de/cms/tier2/store/user/bletzer/hbw_store_v1/hbw_merged/c24v15/cf.CalibrateEvents/calib__muoV0/nominal/${dataset}/" --delete --gfal ; then
        echo "Warning: find_empty_files.zsh failed for ${dataset}" >&2
        FAILED+=("$dataset")
    fi
    echo "--------------------------------"
    if ! ./find_empty_files.zsh "/pnfs/desy.de/cms/tier2/store/user/bletzer/hbw_store_v1/hbw_merged/calib__ak4V1__ak8V0__eleV0__c3633df749/sel__dl1V0/c24v15/cf.SelectEvents/nominal/${dataset}/" --delete --gfal ; then
        echo "Warning: find_empty_files.zsh failed for ${dataset}" >&2
        FAILED+=("$dataset")
    fi
    echo "--------------------------------"
    if ! ./find_empty_files.zsh "/pnfs/desy.de/cms/tier2/store/user/bletzer/hbw_store_v1/hbw_merged/calib__ak4V1__ak8V0__eleV0__c3633df749/sel__dl1V0/red__default/c24v15/cf.ReduceEvents/nominal/${dataset}/" --delete --gfal ; then
        echo "Warning: find_empty_files.zsh failed for ${dataset}" >&2
        FAILED+=("$dataset")
    fi
    echo "--------------------------------"
done

echo "==============================="

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed datasets: ${FAILED[*]}" >&2
fi
