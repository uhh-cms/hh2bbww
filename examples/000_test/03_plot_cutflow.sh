#!/bin/sh
action () {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    source ${this_dir}/common.sh

    args=(
        --version $test_version
        --config $test_config
        --datasets $test_datasets
        --categories $test_cutflow_categories
        --selector-steps $test_steps
        --yscale log
        --shape-norm True
        --process-settings unstack_all
        --cms-label simpw
        --view-cmd imgcat
        "$@"
    )
    law run cf.PlotCutflow "${args[@]}"

    args=(
        ${args[@]}
        --variables $test_cutflow_variables
    )
    law run cf.PlotCutflowVariables1D "${args[@]}" --per-plot processes
    # TODO: there's an issue with the output definition in --per-plot steps, task tries
    #       to produce plots for processes for which no dataset was requested
    # law run cf.PlotCutflowVariables1D "${args[@]}" --per-plot steps
}

action "$@"
