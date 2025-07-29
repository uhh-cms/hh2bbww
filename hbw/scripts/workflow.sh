#!/bin/bash

# How to run our analysis:
# - hbw.BuildCampaignSummary
# - cf.GetDatasetLFNsWrapper
# - cf.MergeReducedEventsWrapper
# - cf.MergeSelectionStatsWrapper
# - hbw.GetBtagNormalizationSF
# - cf.ProduceColumns --producers event_weights,dl_ml_inputs,pre_ml_cats
# - hbw.ExportDYWeights
# - cf.MLTraining --ml-model multiclassv1
# - cf.MLTraining --ml-model ggfv1
# - cf.MLTraining --ml-model vbfv1
# - cf.MLEvaluationWrapper for all three models
# - cf.CreateDatacards with nominal shift
# - add shape systematics and jerc shifts

# Types of plots that we typically produce:
# - cf.PlotShiftedVariables1D for all DNN input features
# - cf.PlotVariables1D for all DNN outputs
# - hbw.PlotMLResultsSingleFold for all ML models
# - hbw.CustomCreateYieldTable for different sets of categories and per campaign
# - hbw.PostfitPlots for pre & post-fit plots (TODO: not function for that yet)


# Default values
default_configs="c22postv14"
all_configs="c22prev14,c22postv14,c23prev14,c23postv14"
default_shifts="nominal"
dnn_multiclass="multiclassv1"
dnn_ggf="ggfv1"
dnn_vbf="vbfv1"
all_models="$dnn_multiclass,$dnn_ggf,$dnn_vbf"
inference_model="dl"
ml_inputs="ml_inputs"
ml_scores="mlscore.max_score,logit_mlscore.sig_ggf_binary,logit_mlscore.sig_vbf_binary"
dry_run="${dry_run:-false}"  # override with: dry_run=true ./workflow.sh ...

# === Helper to run or echo ===
run_cmd() {
    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] $*"
    else
        eval "$@"
    fi
}

run_and_fetch_cmd() {
    local folder="$1"
    mkdir -p $CF_DATA/fetched_plots/$folder && cd $CF_DATA/fetched_plots/$folder
    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] ${*:2}"
        echo "[DRY-RUN] ${*:2} --fetch-output 0,a"
    else
        eval "${@:2}"
        eval "${@:2} --fetch-output 0,a"
    fi
}

checksum() {
	# helper to include custom checksum based on time when task was called
	TEXT="time"
	TIMESTAMP=$(date +"%s")
   	echo "${TEXT}${TIMESTAMP}"
}

# === Task functions ===

run_merge_reduced_events() {
    local configs="${1:-$default_configs}"
    local shifts="${2:-$default_shifts}"
    local checksum=$(checksum)
    run_cmd law run cf.BundleRepo --custom-checksum "$checksum"

    run_cmd law run cf.MergeReducedEventsWrapper \
        --datasets \"*\" \
        --configs "$configs" \
        --shifts "$shifts" \
        --cf.MergeReducedEvents-{retries=1,workflow=htcondor} \
        --cf.ReduceEvents-pilot \
	    --cf.BundleRepo-custom-checksum $checksum \
        --workers 123
}

run_merge_selection_stats() {
    local configs="${1:-$default_configs}"
    local shifts="${2:-$default_shifts}"

    # call per config and shift for reduced resolving overhead
    for config in ${configs//,/ }; do
        for shift in ${shifts//,/ }; do
            run_cmd claw run cf.MergeSelectionStatsWrapper \
                --datasets \"*\" \
                --configs "$config" \
                --shifts "$shift" \
                --cf.SelectEvents-{retries=1,workflow=htcondor,pilot} \
                --workers 6

            run_cmd law run hbw.GetBtagNormalizationSF \
                --config "$config" \
                --shift "$shift"
        done
    done
}

run_plot_nominal() {
    local configs="${1:-$default_configs}"
    local variables="${2:-$ml_inputs}"
    local checksum=$(checksum)
    run_cmd law run cf.BundleRepo --custom-checksum "$checksum"

    for config in ${configs//,/ }; do
        for shift in ${shifts//,/ }; do
            echo "→ PlotNominal: config=$config, shift=$shift"
            run_cmd law run cf.PlotVariables1D \
                --configs "$config" \
                --shift "nominal" \
                --workflow htcondor \
                --variables "$variables" \
	            --cf.BundleRepo-custom-checksum $checksum \
                --workers 6
        done
    done
}

run_produce_columns() {
    local configs="${1:-$default_configs}"
    local shifts="${2:-$default_shifts}"
    local producers="${3:-event_weights,dl_ml_inputs,pre_ml_cats}"

    for config in ${configs//,/ }; do
        for shift in ${shifts//,/ }; do
            echo "→ ProduceColumns: config=$config, shift=$shift"
            # NOTE: this Wrapper does not work for some reason because it does not understand the
            # --datasets "*" argument and therefore runs on nothing
            run_cmd law run cf.ProduceColumnsWrapper \
                --producers "$producers" \
                --workers 8 \
                --datasets \"*\" \
                --configs "$config" \
                --shifts "$shift" \
                --cf.ProduceColumns-{retries=1,workflow=htcondor,no-poll}
        done
    done
}

run_ml_training() {
    for model in "$dnn_multiclass" "$dnn_ggf" "$dnn_vbf"; do
        echo "→ MLTraining: model=$model"
        run_cmd law run cf.MLTraining \
            --ml-model "$model" \
            --workers 123 \
            --workflow htcondor \
            --retries 1 \
            --cf.PrepareMLEvents-pilot \
            --no-poll
    done
}

prepare_dy_corr() {
    local configs="${1:-$all_configs}"
    local checksum=$(checksum)
    run_cmd law run cf.BundleRepo --custom-checksum "$checksum"

    run_cmd claw run hbw.ExportDYWeights \
        --configs "$configs" \
        --retries 1 \
        --workflow htcondor \
        --cf.CreateHistograms-pilot \
        --cf.BundleRepo-custom-checksum $checksum \
        --workers 123
}

prepare_dy_corr_weights() {
    local configs="${1:-$all_configs}"
    local shifts="${2:-$default_shifts}"
    local producer="dy_correction_weight"
    local checksum=$(checksum)
    run_cmd law run cf.BundleRepo --custom-checksum "$checksum"

    echo "→ ProduceColumnsWrapper: producer=$producer configs=$configs, shifts=$shifts"
    run_cmd law run cf.ProduceColumnsWrapper \
        --configs "$configs" \
        --shifts "$shifts" \
        --datasets \"dy_*\" \
        --producers "$producer" \
        --cf.ProduceColumns-{retries=1,workflow=htcondor} \
        --cf.CreateHistograms-pilot \
        --cf.BundleRepo-custom-checksum $checksum \
        --workers 123
}

prepare_mlcolumns() {
    local configs="${1:-$default_configs}"
    local shifts="${2:-$default_shifts}"
    local models="${3:-$all_models}"
    local variables="${4:-$ml_scores}"
    local checksum=$(checksum)
    run_cmd law run cf.BundleRepo --custom-checksum "$checksum"

    # workaround: trigger CreateHistograms with MLModels to MLEvaluations
    # should only be triggered after the DY weights have been produced
    valid_shifts="nominal,jec_Total_up,jec_Total_down,jer_up,jer_down"
    for config in ${configs//,/ }; do
        for shift in ${shifts//,/ }; do
            # only run for valid shifts
            echo "→ CreateHistograms (to trigger column production): config=$config, shift=$shift"
            if [[ ! ",$valid_shifts," =~ ",$shift," ]]; then
                echo "Skipping invalid shift: $shift for config: $config"
                continue
            fi
            run_cmd law run cf.CreateHistogramsWrapper \
                --configs "$config" \
                --shifts "$shift" \
                --datasets \"*\" \
                --cf.CreateHistograms-ml-models "$models" \
                --cf.CreateHistograms-variables "$variables" \
                --cf.CreateHistograms-{retries=1,workflow=htcondor,pilot,no-poll,htcondor-memory=3GB} \
                --cf.BundleRepo-custom-checksum $checksum \
                --workers 6
        done
    done
}

run_ml_evaluation() {
    local configs="${1:-$default_configs}"
    local shifts="${2:-$default_shifts}"
    local models="${3:-$all_models}"
    local checksum=$(checksum)
    run_cmd law run cf.BundleRepo --custom-checksum "$checksum"

    # NOTE: no-poll and pilot: therefore run only if all required Producers have been run prev.
    # the prepare_mlcolumns is probably a better approach (faster resolving for some reason
    # and less potential overhead because each submitted job resolves all Producers/MLModels)
    for config in ${configs//,/ }; do
        for shift in ${shifts//,/ }; do
            for model in ${models//,/ }; do
                echo "→ MLEvaluation: config=$config, shift=$shift, model=$model"
                run_cmd law run cf.MLEvaluationWrapper \
                    --shifts "$shift" \
                    --configs "$config" \
                    --cf.MLEvaluation-ml-model "$model" \
                    --datasets \"*\" \
                    --workers 8 \
                    --cf.BundleRepo-custom-checksum $checksum \
                    --cf.MLEvaluation-{retries=1,workflow=htcondor,pilot,no-poll}
            done
        done
    done
}

run_merge_histograms_local() {
    local configs="${1:-$default_configs}"
    local shifts="${2:-$default_shifts}"
    local models="${3:-$all_models}"
    local variables="${4:-$ml_scores}"

    for config in ${configs//,/ }; do
        for shift in ${shifts//,/ }; do
            echo "→ MergeHistograms: config=$config, shift=$shift, models=$models, variables=$variables"
            run_cmd claw run cf.MergeHistogramsWrapper \
                --configs "$config" \
                --datasets \"*\" \
                --shifts $shift \
                --cf.MergeHistograms-variables "$variables" \
                --cf.MergeHistograms-ml-models "$models" \
                --cf.MergeHistograms-pilot \
                --workers 6
        done
    done
}

run_merge_histograms_htcondor() {
    local configs="${1:-$default_configs}"
    local shifts="${2:-$default_shifts}"
    local models="${3:-$all_models}"
    local variables="${4:-$ml_scores}"
    local checksum=$(checksum)
    run_cmd law run cf.BundleRepo --custom-checksum "$checksum"

    for config in ${configs//,/ }; do
        for shift in ${shifts//,/ }; do
            echo "→ MergeHistograms: config=$config, shift=$shift, models=$models, variables=$variables"
            run_cmd claw run cf.MergeHistogramsWrapper \
                --configs "$config" \
                --shifts $shift \
                --datasets \"*\" \
                --cf.MergeHistograms-variables "$variables" \
                --cf.MergeHistograms-ml-models "$models" \
                --cf.MergeHistograms-{workflow=htcondor,pilot,no-poll} \
                --cf.BundleRepo-custom-checksum $checksum \
                --workers 6
        done
    done
}

run_create_datacards() {
    local configs="${1:-$default_configs}"
    local inference_model="${2:-$inference_model}"
    local models="${3:-$all_models}"
    local checksum=$(checksum)
    run_cmd law run cf.BundleRepo --custom-checksum "$checksum"

    for config in ${configs//,/ }; do
        echo "→ CreateDatacards: config=$config"
        run_cmd claw run cf.CreateDatacards \
            --configs "$config" \
            --inference-model "$inference_model" \
            --ml-models "$models" \
            --cf.MergeHistograms-{retries=1,workflow=local,pilot} \
            --workers 6 \
            --cf.BundleRepo-custom-checksum $checksum \
            --cf.MLEvaluation-workflow htcondor
    done
}

shape_shift_groups="btag_up,btag_down,theory_up,theory_down,experimental_up,experimental_down"
jerc_shifts="jec_Total_up,jec_Total_down,jer_up,jer_down"

vr_categories="incl,sr,dycr,ttcr"
lep_categories="sr__2e,sr__2mu,sr__emu"
jet_categories="sr__resolved__1b,sr__resolved__2b,sr__boosted__1b,sr__boosted__2b"
jet_categories_boosted_split="sr__resolved__1b,sr__resolved__2b,sr__boosted__1b,sr__boosted__2b"
preml_categories="incl,dycr,ttcr,sr,sr__2e,sr__2mu,sr__emu,sr__resolved__1b,sr__resolved__2b,sr__boosted"
inf_categories_sig_ggf="sr__resolved__1b__ml_sig_ggf,sr__resolved__2b__ml_sig_ggf,sr__boosted__ml_sig_ggf"
inf_categories_sig_vbf="sr__resolved__1b__ml_sig_vbf,sr__resolved__2b__ml_sig_vbf,sr__boosted__ml_sig_vbf"
inf_categories_sig="sr__resolved__1b__ml_sig_ggf,sr__resolved__1b__ml_sig_vbf,sr__resolved__2b__ml_sig_ggf,sr__resolved__2b__ml_sig_vbf,sr__boosted__ml_sig_ggf,sr__boosted__ml_sig_vbf"
inf_categories_bkg="sr__1b__ml_tt,sr__1b__ml_st,sr__1b__ml_dy_m10toinf,sr__1b__ml_h,sr__2b__ml_tt,sr__2b__ml_st,sr__2b__ml_dy_m10toinf,sr__2b__ml_h"
inf_categories="sr__resolved__1b__ml_sig_ggf,sr__resolved__1b__ml_sig_vbf,sr__resolved__2b__ml_sig_ggf,sr__resolved__2b__ml_sig_vbf,sr__boosted__ml_sig_ggf,sr__boosted__ml_sig_vbf,sr__1b__ml_tt,sr__1b__ml_st,sr__1b__ml_dy_m10toinf,sr__1b__ml_h,sr__2b__ml_tt,sr__2b__ml_st,sr__2b__ml_dy_m10toinf,sr__2b__ml_h"


run_and_fetch_mlplots() {
    for mlmodel in ${all_models//,/ }; do
        echo "→ MLPlots: model=$mlmodel"
        run_and_fetch_cmd mlplots/$mlmodel law run hbw.PlotMLResultsSingleFold \
            --ml-model $mlmodel \
            --cf.MLTraining-{retries=1,workflow=htcondor} \
            --hbw.MLEvaluationSingleFold-{retries=1,workflow=htcondor}
    done
}

run_and_fetch_mlscore_plots() {
    configs="${1:-"$all_configs"}"
    variables="${2:-"\"mlscore.*,rebinlogit_mlscore.sig*binary\""}"
    categories="${3:-"sr,sr__resolved__1b,sr__resolved__2b,sr__boosted"}"
    unstacked="${4:-false}"

    folder_name=${configs//,/_}

    if [[ "$unstacked" != "true" ]]; then
        process_groups_stack="ddl4"
        for processes in $process_groups_stack; do
            # NOTE: we could also use cf.PlotShiftedVariables1D here to include syst. unc.
            echo "→ PlotMLScores: configs=$configs, processes=$processes, variables=$variables, categories=$categories"
            run_and_fetch_cmd ml_scores/$folder_name claw run cf.PlotVariables1D \
                --configs $configs \
                --variables $variables \
                --categories $categories \
                --ml-models $all_models \
                --processes $processes \
                --workers 6 \
                --cf.MergeHistograms-pilot
        done
    else
        process_groups_unstack="bkgminor bkgmajor hbv_ggf_dl hbv_vbf_dl"
        for processes in $process_groups_unstack; do
            echo "→ PlotMLScores: configs=$configs, processes=$processes, variables=$variables, categories=$categories"
            run_and_fetch_cmd ml_scores/$folder_name claw run cf.PlotVariables1D \
                --configs $configs \
                --variables $variables \
                --categories $categories \
                --ml-models $all_models \
                --processes $processes \
                --process-settings unstack_all \
                --general-settings unstacked \
                --workers 6 \
                --cf.MergeHistograms-pilot
        done
    fi
}

run_and_fetch_all_mlscore_plots() {
    # TODO: DNN plots after categorization and rebinning
    for config in ${all_configs//,/ }; do
        # run_and_fetch_mlscore_plots "$config" "\"mlscore.*,rebinlogit_mlscore.sig*binary\"" "sr,sr__resolved__1b,sr__resolved__2b,sr__boosted" true
        run_and_fetch_mlscore_plots "$config" "\"mlscore.*,rebinlogit_mlscore.sig*binary\"" "sr,sr__resolved__1b,sr__resolved__2b,sr__boosted"
        run_and_fetch_mlscore_plots "$config" "mlscore.max_score" "$inf_categories_bkg"
        run_and_fetch_mlscore_plots "$config" "logit_mlscore.sig_vbf_binary" "$inf_categories_sig_vbf"
        run_and_fetch_mlscore_plots "$config" "logit_mlscore.sig_ggf_binary" "$inf_categories_sig_ggf"
    done

    # run_and_fetch_mlscore_plots "$all_configs" "\"mlscore.*,rebinlogit_mlscore.sig*binary\"" "sr,sr__resolved__1b,sr__resolved__2b,sr__boosted" true
    run_and_fetch_mlscore_plots "$all_configs" "\"mlscore.*,rebinlogit_mlscore.sig*binary\"" "sr,sr__resolved__1b,sr__resolved__2b,sr__boosted"
    run_and_fetch_mlscore_plots "$all_configs" "mlscore.max_score" "$inf_categories_bkg"
    run_and_fetch_mlscore_plots "$all_configs" "logit_mlscore.sig_vbf_binary" "$inf_categories_sig_vbf"
    run_and_fetch_mlscore_plots "$all_configs" "logit_mlscore.sig_ggf_binary" "$inf_categories_sig_ggf"
}

run_and_fetch_mcsyst_plots() {
    configs="${1:-$all_configs}"
    categories="${2:-"incl,dycr,ttcr,sr,sr__resolved__1b,sr__resolved__2b,sr__boosted"}"
    # categories="${2:-"ttcr,dycr,incl"}"
    # TODO {} resolving does not work with the run cmds
    # NOTE: fetching seems to result in a different file name each time, so this is kind of bad
    # ---> fixed by sorting shift_sources, variables, categories in c/f mixins
    # NOTE: fetching a different set of categories also changes the file name per plot (one hash per set of parameters)
    # so we either need to use the same set of categories when fetching of we need to fix the file name generation
    # see fetch_task_output in law/tasks/interative.py and live_task_id in law/tasks/base.py
    # there is also a patch in hbw/columnflow_patches.py that should fix this, but I am not sure if it breaks anything else
    for config in ${configs//,/ }; do
        echo "→ PlotShiftedVariables: config=$config"
        run_and_fetch_cmd with_syst_unc/$config claw run cf.PlotShiftedVariables1D \
            --configs "$config" \
            --shift-sources all \
            --ml-models $all_models \
            --variables ml_inputs \
            --processes ddl4 \
            --categories "$categories" \
            --workers 6 \
            --cf.MergeHistograms-pilot
    done
    run_and_fetch_cmd with_syst_unc/combined claw run cf.PlotShiftedVariables1D --configs $configs --ml-models $all_models --shift-sources all --variables ml_inputs --processes ddl4 --categories "$categories" --workers 6 --cf.MergeHistograms-pilot
}



run_and_fetch_all_yield_tables() {
    # for config in ${all_configs//,/ }; do
    #     echo "→ YieldTables: configs=$config"
    #     run_and_fetch_yield_tables "$config"
    # done
    echo "→ YieldTables: configs=$all_configs"
    run_and_fetch_yield_tables "$all_configs"

}
run_and_fetch_yield_tables() {
    configs="${1:-$all_configs}"
    # Replace commas with underscores for folder name
    folder_name=${configs//,/_}

    # vr categories with data and ratio
    run_and_fetch_cmd tables/$folder_name law run hbw.CustomCreateYieldTable --configs $configs --categories $vr_categories --processes table1 --ratio data,background --table-format latex_raw
    run_and_fetch_cmd tables/$folder_name law run hbw.CustomCreateYieldTable --configs $configs --categories $lep_categories --processes table1 --ratio data,background --table-format latex_raw
    run_and_fetch_cmd tables/$folder_name law run hbw.CustomCreateYieldTable --configs $configs --categories $jet_categories --processes table1 --ratio data,background --table-format latex_raw
    run_and_fetch_cmd tables/$folder_name law run hbw.CustomCreateYieldTable --configs $configs --categories $jet_categories_boosted_split --processes table1 --ratio data,background --table-format latex_raw

    # inference categories without data
    run_and_fetch_cmd tables/$folder_name law run hbw.CustomCreateYieldTable --configs $configs --categories $inf_categories_sig --processes table4 --table-format latex_raw
    run_and_fetch_cmd tables/$folder_name law run hbw.CustomCreateYieldTable --configs $configs --categories $inf_categories_bkg --processes table4 --table-format latex_raw
    # claw run hbw.CustomCreateYieldTable --config $config --categories $inf_categories --processes table4 --table-format latex_raw
}
calls() {
    # just a summary of calls that I often use
    claw run hbw.PrepareInferenceTaskCalls --inference-model dl_jerc_bjet_uncorr1 --configs "c22prev14,c22postv14,c23prev14,c23postv14" --remove-output 0,a,y --workers 6
    claw run hbw.CustomCreateYieldTable --processes data,background,tt,dy,dy_lf_m10to50,dy_lf_m50toinf,dy_hf_m10to50,dy_hf_m50toinf,st,w_lnu,vv,ttv,h,other --ratio data,background,tt,dy --remove-output 0,a,y --categories "sr__{1b,2b}__ml_{sig_ggf,sig_vbf}"
    claw run hbw.CustomCreateYieldTable --processes data,background,tt,dy,st,w_lnu,vv,ttv,h,other --ratio data,background,tt,dy --remove-output 0,a,y
}

# === Dispatcher ===
# TODO: this is still super messy (comment in & out whatever is currently requested),
# we might want to implement a proper dispatcher at some point
run_all() {
    # run_merge_reduced_events "$all_configs" "nominal"
    # run_merge_selection_stats "$all_configs" "nominal"

    # run_produce_columns "$all_configs" "nominal"
    # run_produce_columns "$all_configs" "$jerc_shifts"
    # prepare_dy_corr_weights "$all_configs" "nominal"
    # prepare_dy_corr_weights "$all_configs" "$jerc_shifts"
    # prepare_mlcolumns "$all_configs" "$nominal" "$dnn_multiclass,$dnn_ggf,$dnn_vbf" "$ml_inputs"
    # prepare_mlcolumns "$all_configs" "$jerc_shifts" "$dnn_multiclass,$dnn_ggf,$dnn_vbf" "$ml_inputs"

    # for config in ${all_configs//,/ }; do
    #     run_merge_histograms_htcondor "$config" "nominal"
    #     run_merge_histograms_htcondor "$config" "$shape_shift_groups"
    #     run_merge_histograms_local "$config" "$jerc_shifts"
    # done

    # run_and_fetch_all_yield_tables
    # run_and_fetch_all_mlscore_plots
    # run_and_fetch_mcsyst_plots

    # for config in ${all_configs//,/ }; do
    #     run_merge_histograms_local "$config" "nominal" "$dnn_multiclass,$dnn_ggf,$dnn_vbf" "$ml_inputs"
    #     run_merge_histograms_local "$config" "$shape_shift_groups" "$dnn_multiclass,$dnn_ggf,$dnn_vbf" "$ml_inputs"
    #     run_merge_histograms_local "$config" "$jerc_shifts" "$dnn_multiclass,$dnn_ggf,$dnn_vbf" "$ml_inputs"
    # done

    # run_and_fetch_cmd yield_tables/c22postv14 claw run hbw.CustomCreateYieldTable --config c22postv14 --processes background,hh_sm,tt,dy,st,w_lnu,vv,ttv,h,other --ratio data,background --categories "sr__\{1b,2b\}__ml_\{sig_ggf,sig_vbf\}" --remove-output 0,a,y

    # run_and_fetch_mcsyst_plots
    # run_and_fetch_mlplots

    # run_create_datacards "$all_configs" "dl_jerc" "$all_models"
}

# === Example usage ===
# dry_run=true ./workflow.sh
# ./workflow.sh

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && run_all "$@"
