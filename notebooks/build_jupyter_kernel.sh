#!/bin/sh

# USAGE: run on a fresh shell to create jupyter kernel with name VENV_NAME.
# If necessary, customize some variables such as the CF_SETUP_NAME,
# VENV_NAME or which packages are supposed to be installed

action () {
    # customizable variables
    CF_SETUP_NAME="dev"
    VENV_NAME="cf_ml"
    DISPLAY_NAME=$VENV_NAME

    # determine path to this directory
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # set some other paths
    local HBW_BASE="$this_dir/.."
    local CF_BASE="$HBW_BASE/modules/columnflow"
    source $HBW_BASE/.setups/${CF_SETUP_NAME}.sh

    # setup path to venv location
    local JUPYTER_VENV_PATH="$CF_SOFTWARE_BASE/jupyter_venvs"
    mkdir -p $JUPYTER_VENV_PATH

    # create and activate python venv from conda
    $CF_SOFTWARE_BASE/conda/bin/python3.9 -m venv $JUPYTER_VENV_PATH/$VENV_NAME
    source $JUPYTER_VENV_PATH/$VENV_NAME/bin/activate

    pip install ipykernel

    # install packages of interest (cf is always required)
    pip install -r $CF_BASE/sandboxes/cf.txt
    # pip install -r $CF_BASE/sandboxes/columnar.txt
    pip install -r $HBW_BASE/sandboxes/ml_plotting.txt

    # create the kernel
    ipython kernel install --user --name=${DISPLAY_NAME}
}

action "$@"
