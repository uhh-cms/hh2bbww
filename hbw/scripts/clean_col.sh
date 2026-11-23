#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

BASE=/data/dust/user/markusla/analysis/dilepton/hh2bbww/data/cf_store/hbw_hhh_dl/cf.ProduceColumns

TO_DELETE=(
  prod__hhh_ml_inputsV1
)

LIST=~/to_delete.txt
: > "$LIST"

# layout: <campaign>/<dataset>/<shift>/<calib>/<sel>/<red>/<prod>
for p in "${TO_DELETE[@]}"; do
  for d in "$BASE"/c24v15/*/*/*/*/*/"$p" "$BASE"/c25v15/*/*/*/*/*/"$p"; do
    [[ -d "$d" ]] && echo "$d" >> "$LIST"
  done
done

echo "Found $(wc -l < "$LIST") folders -> $LIST"

if [[ "${1:-}" == "--delete" ]]; then
  while IFS= read -r d; do
    rm -rf -- "$d"
  done < "$LIST"
  echo "Deleted."
else
  echo "Dry run only. Check $LIST, then run: bash $0 --delete"
fi