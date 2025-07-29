#!/usr/bin/env bash

setup_hbw() {
    # Runs the project setup, leading to a collection of environment variables starting with either
    #   - "CF_", for controlling behavior implemented by columnflow, or
    #   - "HBW_", for features provided by the analysis repository itself.
    # Check the setup.sh in columnflow for documentation of the "CF_" variables. The purpose of all
    # "HBW_" variables is documented below.
    #
    # The setup also handles the installation of the software stack via virtual environments, and
    # optionally an interactive setup where the user can configure certain variables.
    #
    #
    # Arguments:
    #   1. The name of the setup. "default" (which is itself the default when no name is set)
    #      triggers a setup with good defaults, avoiding all queries to the user and the writing of
    #      a custom setup file. See "interactive_setup()" for more info.
    #
    #
    # Optinally preconfigured environment variables:
    #   None yet.
    #
    #
    # Variables defined by the setup and potentially required throughout the analysis:
    #   HBW_BASE
    #       The absolute analysis base directory. Used to infer file locations relative to it.
    #   HBW_SETUP
    #       A flag that is set to 1 after the setup was successful.

    #
    # load cf setup helpers
    #

    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local cf_base="${this_dir}/modules/columnflow"
    CF_SKIP_SETUP="true" source "${cf_base}/setup.sh" "" || return "$?"

    #
    # prevent repeated setups
    #

    cf_export_bool HBW_SETUP
    if ${HBW_SETUP} && ! ${CF_ON_SLURM}; then
        >&2 echo "the HH -> bbWW analysis was already succesfully setup"
        >&2 echo "re-running the setup requires a new shell"
        return "1"
    fi

    #
    # prepare local variables
    #

    local orig="${PWD}"
    local setup_name="${1:-default}"
    local setup_is_default="false"
    [ "${setup_name}" = "default" ] && setup_is_default="true"

    # zsh options
    if ${shell_is_zsh}; then
        emulate -L bash
        setopt globdots
    fi

    #
    # global variables
    # (HBW = hh2bbww, CF = columnflow)
    #

    # start exporting variables
    export HBW_BASE="${this_dir}"
    export CF_BASE="${cf_base}"
    export CF_REPO_BASE="${HBW_BASE}"
    export CF_REPO_BASE_ALIAS="HBW_BASE"
    export CF_SETUP_NAME="${setup_name}"
    export CF_SCHEDULER_HOST="${CF_SCHEDULER_HOST:-naf-cms16.desy.de}"
    export CF_SCHEDULER_PORT="${CF_SCHEDULER_PORT:-6028}"

    # interactive setup
    if ! ${CF_REMOTE_ENV}; then
        cf_setup_interactive_body() {
            # the flavor will be cms
            export CF_FLAVOR="cms"

            # query common variables
            cf_setup_interactive_common_variables

            # specific variables would go here
            query HBW_LAW_CONFIG "Name of the file to be used as law config (must be located in $HBW_BASE)" "law.dl.nocert.cfg"
        }
        cf_setup_interactive "${CF_SETUP_NAME}" "${HBW_BASE}/.setups/${CF_SETUP_NAME}.sh" || return "$?"
    fi

    # continue the fixed setup
    export CF_CONDA_BASE="${CF_CONDA_BASE:-${CF_SOFTWARE_BASE}/conda}"
    export CF_VENV_BASE="${CF_VENV_BASE:-${CF_SOFTWARE_BASE}/venvs}"
    export CF_CMSSW_BASE="${CF_CMSSW_BASE:-${CF_SOFTWARE_BASE}/cmssw}"
    export HBW_LAW_CONFIG="${HBW_LAW_CONFIG:-law.dl.nocert.cfg}"
    export LAW_CONFIG_FILE="${LAW_CONFIG_FILE:-${HBW_BASE}/${HBW_LAW_CONFIG}}"

    #
    # common variables
    #

    cf_setup_common_variables || return "$?"

    #
    # minimal local software setup
    #

    cf_setup_software_stack "${CF_SETUP_NAME}" || return "$?"

    # ammend paths that are not covered by the central cf setup
    export PATH="${HBW_BASE}/bin:${PATH}"
    export PATH="${HBW_BASE}/hbw/scripts:${PATH}"
    export PYTHONPATH="${HBW_BASE}:${HBW_BASE}/modules/cmsdb:${PYTHONPATH}"

    # initialze submodules
    if ! ${CF_REMOTE_ENV} && [ -e "${HBW_BASE}/.git" ]; then
        local m
        for m in $( ls -1q "${HBW_BASE}/modules" ); do
            cf_init_submodule "${HBW_BASE}" "modules/${m}"
        done
    fi

    #
    # additional common cf setup steps
    #

    cf_setup_post_install || return "$?"

    # update the law config file to switch from mirrored to bare wlcg targets
    # as local mounts are typically not available remotely
    if ${CF_REMOTE_ENV}; then
        sed -i -r 's/(.+\: ?)wlcg_mirrored, local_.+, ?(wlcg_[^\s]+)/\1wlcg, \2/g' "${LAW_CONFIG_FILE}"
    fi

    #
    # finalize
    #


    export HBW_SETUP="true"
}

main() {
    # Invokes the main action of this script, catches possible error codes and prints a message.

    # run the actual setup
    if setup_hbw "$@"; then
        cf_color green "HH -> bbWW analysis successfully setup"
        return "0"
    else
        local code="$?"
        cf_color red "setup failed with code ${code}"
        return "${code}"
    fi
}

# entry point
if [ "${HBW_SKIP_SETUP}" != "true" ]; then
    main "$@"
fi
