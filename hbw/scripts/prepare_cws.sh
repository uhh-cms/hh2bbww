#!/usr/bin/env bash
# Prepares the cws inference for several inference models. Every inference model is resolved in its own law call,
# because its ML categories (added once per config, ml_ids differ between Cat_3b and Cat_4b) would clash with those
# of the other model in a shared process. Afterwards the manifests are combined into inference.yaml + run.sh.
#
# usage:
#   bash prepare_cws.sh --config-groups c24v15,c25v15 --models 3b=gatja_3b_shape,4b=gatja_4b_shape \
#       [--version <version>] [--card-mode standard|dummy] [-- <extra args for PrepareCWSDatacards, e.g. --workers 4>]
#
#   --config-groups   as for the law tasks (groups separated by ':', configs within a group by ',')
#   --models          <label>=<inference model>, comma separated; the label names the datacard group (3b, 4b)
#   --version         version of both tasks (default: the analysis default)
#   --card-mode       default card mode of the generated run.sh (default: standard; run.sh can switch with card_mode=)
set -euo pipefail

config_groups=""
models=""
version_args=()
card_mode_args=()
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-groups) config_groups="$2"; shift 2 ;;
        --models) models="$2"; shift 2 ;;
        --version) version_args=(--version "$2"); shift 2 ;;
        --card-mode) card_mode_args=(--card-mode "$2"); shift 2 ;;
        --) shift; extra=("$@"); break ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument '$1' (see --help)"; exit 2 ;;
    esac
done
if [[ -z "$config_groups" || -z "$models" ]]; then
    echo "[ERROR] --config-groups and --models are required (see --help)"
    exit 2
fi

labels=()
manifests=()
IFS=',' read -ra pairs <<< "$models"
for pair in "${pairs[@]}"; do
    label="${pair%%=*}"
    model="${pair#*=}"
    if [[ "$label" == "$pair" || -z "$label" || -z "$model" ]]; then
        echo "[ERROR] --models entries must look like <label>=<inference model>, got '$pair'"
        exit 2
    fi
    args=(
        hbw.PrepareCWSDatacards --inference-model "$model" --config-groups "$config_groups"
        ${version_args[@]+"${version_args[@]}"} ${extra[@]+"${extra[@]}"}
    )

    echo ""
    echo "========== [$label] $model =========="
    law run "${args[@]}"
    manifest=$(law run "${args[@]}" --print-output 0 | grep -o '/[^ ,)]*manifest\.json' | head -n 1 || true)
    if [[ -z "$manifest" || ! -f "$manifest" ]]; then
        echo "[ERROR] could not find the manifest.json of $model (law run ${args[*]} --print-output 0)"
        exit 1
    fi
    echo "[OK] $manifest"
    labels+=("$label")
    manifests+=("$manifest")
done

join() { local IFS=","; echo "$*"; }

echo ""
echo "========== combine: ${labels[*]} =========="
# always regenerate: this step only writes text files
law run hbw.PrepareCWSTaskCalls \
    --manifests "$(join "${manifests[@]}")" \
    --model-labels "$(join "${labels[@]}")" \
    ${version_args[@]+"${version_args[@]}"} \
    ${card_mode_args[@]+"${card_mode_args[@]}"} \
    --remove-output 0,a,y