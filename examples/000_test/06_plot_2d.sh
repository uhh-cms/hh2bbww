#!/bin/sh
action () {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    source ${this_dir}/common.sh

    args=(
        --version $test_version
        --config $test_config
        --datasets $test_dataset
        --categories $test_categories
        --variables $test_2d_variables
        --zscale log
        --shape-norm False
        # --cms-label simpw
        --view-cmd imgcat
        "$@"
    )

    law run cf.PlotVariables2D "${args[@]}"
}

action "$@"
