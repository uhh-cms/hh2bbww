#!/bin/zsh

# Usage:
#   ./find_empty_files.zsh /path/to/folder               # just list empty files
#   ./find_empty_files.zsh /path/to/folder --delete      # delete with rm
#   ./find_empty_files.zsh /path/to/folder --delete --gfal # delete with gfal-rm

# --- Parse arguments ---
target_dir="$1"
delete_mode=false
use_gfal=false

for arg in "$@"; do
  case $arg in
    --delete)
      delete_mode=true
      ;;
    --gfal)
      use_gfal=true
      ;;
  esac
done

# --- Validate input ---
if [[ -z "$target_dir" ]]; then
  echo "Usage: $0 /path/to/folder [--delete] [--gfal]"
  exit 1
fi

if [[ ! -d "$target_dir" ]]; then
  echo "Error: '$target_dir' is not a directory."
  exit 1
fi

# --- Find empty files ---
empty_files=($(find "$target_dir" -type f -empty))

if [[ ${#empty_files[@]} -eq 0 ]]; then
  echo "No empty files found in '$target_dir'."
  exit 0
fi

echo "Found ${#empty_files[@]} empty file(s):"
for file in $empty_files; do
  echo "  $file"
done

# --- Handle deletion ---
webdav_prefix="davs://dcache-cms-webdav-wan.desy.de:2880"


if $delete_mode; then
  echo
  read "confirm? Are you sure you want to delete these files? (y/n) "
  if [[ "$confirm" == [Yy] ]]; then
    for file in $empty_files; do
      if $use_gfal; then
        # Ensure file is an absolute path (you can adapt this if needed)
        gfal_path="${webdav_prefix}${file}"
        gfal-rm "$gfal_path" && echo "gfal-rm: $gfal_path"
      else
        rm "$file" && echo "rm: $file"
      fi
    done
    echo "All empty files deleted."
  else
    echo "Aborted deletion."
  fi
fi
# --- End of script ---
# Exit with success
exit 0
