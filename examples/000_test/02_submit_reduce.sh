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
        --workers 4
        --cf.ReduceEvents-workflow htcondor
        --cf.ReduceEvents-pilot True
        --cf.ReduceEvents-parallel-jobs 5000
        "$@"
    )

    law run cf.ReduceEventsWrapper "${args[@]}"
}

action "$@"
