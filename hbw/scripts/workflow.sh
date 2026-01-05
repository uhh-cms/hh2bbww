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
configs_sep="c22prev14 c22postv14 c23prev14 c23postv14"
default_shifts="nominal"
dnn_multiclass="multiclassv3"
dnn_ggf="ggfv3"
dnn_vbf="vbfv3_tag"
all_models="$dnn_multiclass,$dnn_ggf,$dnn_vbf"
inference_model="dl"
ml_inputs="ml_inputs"
ml_scores="logit_mlscore.sig_ggf_binary,logit_mlscore.sig_vbf_binary"
all_ml_scores="mlscore.*,rebinlogit_mlscore.sig*binary,logit_mlscore.sig*binary"

shape_shift_groups="btag_up,btag_down,theory_up,theory_down,experimental_up,experimental_down"
jerc_shifts="jec_Total_up,jec_Total_down,jer_up,jer_down"
all_shift_sources="all"

vr_categories="incl,sr,dycr,ttcr"
lep_categories="sr__2e,sr__2mu,sr__emu"
jet_categories="sr__resolved__1b,sr__resolved__2b,sr__boosted__1b,sr__boosted__2b"
jet_categories_boosted_split="sr__resolved__1b,sr__resolved__2b,sr__boosted__1b,sr__boosted__2b"
preml_categories="incl,dycr,ttcr,sr,sr__2e,sr__2mu,sr__emu,sr__resolved__1b,sr__resolved__2b,sr__boosted"
inf_categories_sig_ggf="sr__resolved__1b__ml_sig_ggf,sr__resolved__2b__ml_sig_ggf,sr__boosted__ml_sig_ggf"
inf_categories_sig_vbf="sr__resolved__1b__ml_sig_vbf,sr__resolved__2b__ml_sig_vbf,sr__boosted__ml_sig_vbf"
inf_categories_sig="sr__resolved__1b__ml_sig_ggf,sr__resolved__1b__ml_sig_vbf,sr__resolved__2b__ml_sig_ggf,sr__resolved__2b__ml_sig_vbf,sr__boosted__ml_sig_ggf,sr__boosted__ml_sig_vbf"
inf_categories_sig_resolved="sr__resolved__1b__ml_sig_ggf,sr__resolved__1b__ml_sig_vbf,sr__resolved__2b__ml_sig_ggf,sr__resolved__2b__ml_sig_vbf"
inf_categories_sig_boosted="sr__boosted__ml_sig_ggf,sr__boosted__ml_sig_vbf"
inf_categories_bkg="sr__1b__ml_tt,sr__1b__ml_st,sr__1b__ml_dy_m10toinf,sr__1b__ml_h,sr__2b__ml_tt,sr__2b__ml_st,sr__2b__ml_dy_m10toinf,sr__2b__ml_h"
inf_categories="sr__resolved__1b__ml_sig_ggf,sr__resolved__1b__ml_sig_vbf,sr__resolved__2b__ml_sig_ggf,sr__resolved__2b__ml_sig_vbf,sr__boosted__ml_sig_ggf,sr__boosted__ml_sig_vbf,sr__1b__ml_tt,sr__1b__ml_st,sr__1b__ml_dy_m10toinf,sr__1b__ml_h,sr__2b__ml_tt,sr__2b__ml_st,sr__2b__ml_dy_m10toinf,sr__2b__ml_h"




dry_run="${dry_run:-false}"  # override with: dry_run=true ./workflow.sh ...
global_checksum=""
# === Helper to run or echo ===
run_cmd() {
    if [[ "$dry_run" == "false" ]]; then
        echo "[RUNNING] $*"
        # NOTE: eval is needed to properly handle the quoting of arguments, but can mess up other things
        "$@"
    else
        echo "[DRY-RUN] $*"
    fi
}

run_and_fetch_cmd() {
    local folder="$1"
    mkdir -p $CF_DATA/fetched_plots/$folder && cd $CF_DATA/fetched_plots/$folder
    if [[ "$dry_run" == "false" ]]; then
        echo "[RUNNING] ${*:2}"
        "${@:2}"
        echo "[FETCHING] ${*:2} --fetch-output 0,a"
        # NOTE: it seems like the fetching only works with claw ???
        "${@:2} --fetch-output 0,a"
    else
        echo "[DRY-RUN] ${*:2}"
        echo "[DRY-RUN] ${*:2} --fetch-output 0,a"
    fi
}

checksum() {
    # If global checksum is already set, return it
    if [[ -n "$global_checksum" ]]; then
        echo "$global_checksum"
        return
    fi

	# helper to include custom checksum based on time when task was called
	TEXT="time"
	TIMESTAMP=$(date +"%s")
   	echo "${TEXT}${TIMESTAMP}"
}

# === Task functions ===
recreate_campaign_summary() {
    local configs="${1:-$all_configs}"

    for config in ${configs//,/ }; do
        echo "→ BuildCampaignSummary: config=$config"
        run_cmd law run hbw.BuildCampaignSummary \
            --config "$config" \
            --remove-output 0,a,y
    done
}

run_merge_reduced_events() {
    local configs="${1:-$default_configs}"
    local shifts="${2:-$default_shifts}"

    run_cmd law run cf.MergeReducedEventsWrapper \
        --datasets "*" \
        --configs "$configs" \
        --shifts "$shifts" \
        --cf.MergeReducedEvents-{retries=2,workflow=htcondor} \
        --cf.ReduceEvents-pilot \
	    --cf.BundleRepo-custom-checksum $(checksum) \
        --workers 123
}

run_merge_selection_stats() {
    local configs="${1:-$default_configs}"
    local shifts="${2:-$default_shifts}"

    # call per config and shift for reduced resolving overhead
    for config in ${configs//,/ }; do
        for shift in ${shifts//,/ }; do
            run_cmd claw run cf.MergeSelectionStatsWrapper \
                --datasets "*" \
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

    for config in ${configs//,/ }; do
        for shift in ${shifts//,/ }; do
            echo "→ PlotNominal: config=$config, shift=$shift"
            run_cmd claw run cf.PlotVariables1D \
                --configs "$config" \
                --shift "nominal" \
                --workflow htcondor \
                --variables "$variables" \
	            --cf.BundleRepo-custom-checksum $(checksum) \
                --workers 6
        done
    done
}

# run_produce_columns() {
#     local configs="${1:-$default_configs}"
#     local shifts="${2:-$default_shifts}"
#     local producers="${3:-event_weights,dl_ml_inputs,pre_ml_cats}"

#     for config in ${configs//,/ }; do
#         for shift in ${shifts//,/ }; do
#             echo "→ ProduceColumns: config=$config, shift=$shift"
#             # NOTE: this Wrapper does not work for some reason because it does not understand the
#             # --datasets "*" argument and therefore runs on nothing
#             run_cmd law run cf.ProduceColumnsWrapper \
#                 --producers "$producers" \
#                 --workers 8 \
#                 --datasets "*" \
#                 --configs "$config" \
#                 --shifts "$shift" \
#                 --cf.ProduceColumns-{retries=1,workflow=htcondor,no-poll}
#         done
#     done
# }


prepare_dy_corr() {
    local configs="${1:-$all_configs}"

    echo "→ ExportDYWeights: config=$configs"
    # start by running ExportDYWeights with all configs combined to better parallelize the jobs
    run_cmd law run hbw.ExportDYWeights \
        --configs "$configs" \
        --retries 1 \
        --workflow htcondor \
        --cf.CreateHistograms-pilot \
        --cf.BundleRepo-custom-checksum $(checksum) \
        --workers 123

    for config in ${configs//,/ }; do
        echo "→ ExportDYWeights: config=$config"
        # produce the DY weights for each config separately at the end
        run_cmd law run hbw.ExportDYWeights \
            --configs "$config"
    done
}

run_ml_training() {
    mlmodels="${1:-"$dnn_multiclass,$dnn_ggf,$dnn_vbf"}"
    for model in ${mlmodels//,/ }; do
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

fetch_ml_metrics() {
    for model in "$dnn_multiclass" "$dnn_ggf" "$dnn_vbf"; do
        echo "→ Fetch ML metrics: model=$model"
        run_and_fetch_cmd ml_metrics/$model claw run cf.MLTraining \
            --ml-model "$model" \
            --workers 0
    done
}

prepare_mlcolumns() {
    local configs="${1:-$default_configs}"
    local shifts="${2:-$default_shifts}"
    local models="${3:-$all_models}"
    local variables="${4:-$ml_scores}"
    local run_ml="false"

    if [[ "$run_ml" == "true" ]]; then
        # check that ML trainings are finished
        for model in ${models//,/ }; do
            echo "→ check MLTraining: model=$model"
            run_cmd law run cf.MLTraining \
                --ml-model "$model" \
                --workflow htcondor \
                --retries 1 \
                --cf.BundleRepo-custom-checksum $(checksum) \
                --cf.PrepareMLEvents-pilot
        done
    fi

    # workaround: trigger CreateHistograms with MLModels to MLEvaluations
    # should only be triggered after the DY weights have been produced
    # should only be triggered after the DNN training is done
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
                --datasets "*" \
                --cf.CreateHistograms-ml-models "$models" \
                --cf.CreateHistograms-variables "$variables" \
                --cf.CreateHistograms-{retries=1,workflow=htcondor,pilot,no-poll,htcondor-memory=3GB} \
                --cf.BundleRepo-custom-checksum $(checksum) \
                --workers 6
        done
    done
}

run_ml_evaluation() {
    local configs="${1:-$default_configs}"
    local shifts="${2:-$default_shifts}"
    local models="${3:-$all_models}"

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
                    --datasets "*" \
                    --workers 8 \
                    --cf.BundleRepo-custom-checksum $(checksum) \
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
                --datasets "*" \
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
    local ensure_nominal=true

    for config in ${configs//,/ }; do
        # run nominal first to ensure it is done before the syst. shifts
        if [[ "$ensure_nominal" != "false" ]]; then
            run_merge_histograms_local "$config" "nominal" "$models" "$variables"
        fi
        pids=()
        for shift in ${shifts//,/ }; do
            echo "→ MergeHistograms: config=$config, shift=$shift, models=$models, variables=$variables"
            (run_cmd claw run cf.MergeHistogramsWrapper \
                --configs "$config" \
                --shifts $shift \
                --datasets "*" \
                --cf.MergeHistograms-variables "$variables" \
                --cf.MergeHistograms-ml-models "$models" \
                --cf.MergeHistograms-{workflow=htcondor,pilot,no-poll,remote-claw-sandbox=venv_columnar} \
                --cf.BundleRepo-custom-checksum $(checksum) \
                --workers 6) &
                pids+=($!)
        done
        # Wait for all to finish
        for pid in "${pids[@]}"; do
        wait "$pid"
        done
        echo "Processes for config $config completed."
    done
}

run_merge_shifted_histograms_htcondor() {
    local configs="${1:-$default_configs}"
    local shifts="${2:-$all_shift_sources}"
    local models="${3:-$all_models}"
    local variables="${4:-$ml_scores}"
    local ensure_nominal=false

    for config in ${configs//,/ }; do
        # run nominal first to ensure it is done before the syst. shifts
        if [[ "$ensure_nominal" != "false" ]]; then
            run_merge_histograms_local "$config" "nominal" "$models" "$variables"
        fi
        echo "→ MergeShiftedHistograms: config=$config, shifts=$shifts, models=$models, variables=$variables"
        run_cmd claw run cf.MergeShiftedHistogramsWrapper \
            --configs "$config" \
            --cf.MergeShiftedHistograms-shift-sources $shifts \
            --datasets "*" \
            --cf.MergeShiftedHistograms-variables "$variables" \
            --cf.MergeShiftedHistograms-ml-models "$models" \
            --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,no-poll,remote-claw-sandbox=venv_columnar} \
            --cf.BundleRepo-custom-checksum $(checksum) \
            --workers 6
        echo "Processes for config $config completed."
    done
}

run_datacards() {
    local inference_model="${1:-$inference_model}"
    local config_groups="${2:-$all_configs}"
    local run_local="${3:-false}"

    local workflow="local"
    local workers=6
    local no_poll="False"
    if [[ "$run_local" == "false" ]]; then
        local workflow="htcondor"
        # local workers=123
        local workers=20
        local no_poll="True"
    fi

    echo "→ Datacards: config_groups=$config_groups, inference_model=$inference_model"
    # NOTE: this fails as long as any of the MergeShiftedHistograms outputs are missing
    # but the MergeShiftedHistograms jobs of the missing outputs will be submitted
    run_cmd claw run hbw.PrepareInferenceTaskCalls \
        --config-groups "$config_groups" \
        --inference-model "$inference_model" \
        --cf.MergeShiftedHistograms-{workflow=$workflow,pilot,no-poll=$no_poll,remote-claw-sandbox=venv_columnar} \
        --cf.BundleRepo-custom-checksum $(checksum) \
        --workers $workers
}

run_and_fetch_mlplots() {
    mlmodels="${1:-$all_models}"
    pids=()
    for mlmodel in ${mlmodels//,/ }; do
        echo "→ MLPlots: model=$mlmodel"
        run_and_fetch_cmd mlplots/$mlmodel claw run hbw.PlotMLResultsSingleFold \
            --ml-model $mlmodel \
            --cf.MLTraining-{retries=1,workflow=htcondor} \
            --hbw.MLEvaluationSingleFold-{retries=1,workflow=htcondor} \
            --workers 3 --workflow htcondor &
            pids+=($!)
    done
    # Wait for all to finish
    for pid in ${pids[@]}; do
        wait "$pid"
    done
    echo "All MLPlots processes completed."
}

run_and_fetch_mlscore_plots() {
    local configs="${1:-"$all_configs"}"
    local variables="${2:-"\"mlscore.*,rebinlogit_mlscore.sig*binary\""}"
    local category_groups="${3:-"sr,sr__resolved__1b,sr__resolved__2b sr__boosted"}"

    local folder_name=${configs//,/_}

    # NOTE: we could also use cf.PlotShiftedVariables1D here to include syst. unc.
    for categories in $category_groups; do
        # when category is boosted, use different settings
        if [[ $categories == *"boosted"* ]]; then
            local variable_settings="rebin_ml_scores100"
            local custom_style_config="default_rax75"
        else
            local variable_settings="none"
            local custom_style_config="default"
        fi
        echo "→ PlotMLScores: configs=$configs, categories=$categories, variables=$variables"
        run_and_fetch_cmd ml_scores_syst/$folder_name claw run cf.PlotShiftedVariables1D \
            --configs $configs \
            --shift-sources all \
            --variables $variables \
            --categories $categories \
            --ml-models $all_models \
            --processes ddl4 \
            --variable-settings $variable_settings \
            --custom-style-config $custom_style_config \
            --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,remote-claw-sandbox=venv_columnar} \
            --cf.BundleRepo-custom-checksum $(checksum) \
            --workers 123 --workflow htcondor
            # NOTE: params passed for partial blinding, currently not used anymore
            # --hist-hooks blind_bins_above_score \
            # --general-settings data_mc_plots_blind_conservative \
    done
}

run_and_fetch_mlscore_plots_unstacked() {
    local configs="${1:-"$all_configs"}"
    local variables="${2:-"\"mlscore.*,rebinlogit_mlscore.sig*binary\""}"
    local category_groups="${3:-"sr,sr__resolved__1b,sr__resolved__2b sr__boosted"}"
    local process_groups_unstack="bkgminor bkgmajor hbv_ggf_dl hbv_vbf_dl"

    for processes in $process_groups_unstack; do
        echo "→ PlotMLScores: configs=$configs, processes=$processes, variables=$variables, categories=$category_groups"
        run_and_fetch_cmd ml_scores/$folder_name claw run cf.PlotVariables1D \
            --configs $configs \
            --variables $variables \
            --categories $category_groups \
            --ml-models $all_models \
            --processes $processes \
            --process-settings unstack_all \
            --general-settings unstacked \
            --workers 6 \
            --cf.MergeHistograms-pilot
    done
}

run_and_fetch_all_mlscore_plots() {
    # run_and_fetch_mlscore_plots "$all_configs" "\"mlscore.*,rebinlogit_mlscore.sig*binary\"" "sr,sr__resolved__1b,sr__resolved__2b sr__boosted" true
    run_and_fetch_mlscore_plots "$all_configs" "\"mlscore.*,rebinlogit_mlscore.sig*binary\"" "sr,sr__resolved__1b,sr__resolved__2b sr__boosted"
    # run_and_fetch_mlscore_plots "$all_configs" "mlscore.max_score" "$inf_categories_bkg"
    # run_and_fetch_mlscore_plots "$all_configs" "logit_mlscore.sig_vbf_binary" "$inf_categories_sig_vbf"
    # run_and_fetch_mlscore_plots "$all_configs" "logit_mlscore.sig_ggf_binary" "$inf_categories_sig_ggf"

    for config in ${all_configs//,/ }; do
        # run_and_fetch_mlscore_plots "$config" "\"mlscore.*,rebinlogit_mlscore.sig*binary\"" "sr,sr__resolved__1b,sr__resolved__2b sr__boosted" true
        run_and_fetch_mlscore_plots "$config" "\"mlscore.*,rebinlogit_mlscore.sig*binary\"" "sr,sr__resolved__1b,sr__resolved__2b sr__boosted"
        # run_and_fetch_mlscore_plots "$config" "mlscore.max_score" "$inf_categories_bkg"
        # run_and_fetch_mlscore_plots "$config" "logit_mlscore.sig_vbf_binary" "$inf_categories_sig_vbf"
        # run_and_fetch_mlscore_plots "$config" "logit_mlscore.sig_ggf_binary" "$inf_categories_sig_ggf"
    done

}


run_and_fetch_mcstat_plots() {
    local configs="${1:-$all_configs}"
    local categories="${2:-"incl,dycr,ttcr,sr,sr__resolved__1b,sr__resolved__2b"}"
    local variables="${3:-$ml_inputs}"

    local folder_name=${configs//,/_}
    echo "→ PlotVariables: config=$configs, categories=$categories, variables=$variables"
    run_and_fetch_cmd no_syst_unc/$folder_name claw run cf.PlotVariables1D \
        --configs "$configs" \
        --ml-models $all_models \
        --variables $variables \
        --processes ddl4 \
        --categories "$categories" \
        --general-settings data_mc_plots_not_blinded \
        --workers 6 \
        --cf.CreateHistograms-pilot
        # --cf.MergeHistograms-pilot
}

run_and_fetch_mcsyst_plots() {
    local configs="${1:-$all_configs}"
    local categories="${2:-"incl,dycr,ttcr,sr,sr__resolved__1b,sr__resolved__2b"}"
    local variables="${3:-$ml_inputs}"

    local folder_name=${configs//,/_}
    echo "→ PlotShiftedVariables: config=$configs"
    run_and_fetch_cmd with_syst_unc/$folder_name claw run cf.PlotShiftedVariables1D \
        --configs "$configs" \
        --shift-sources all \
        --ml-models $all_models \
        --variables $variables \
        --processes ddl4 \
        --categories "$categories" \
        --general-settings data_mc_plots_not_blinded \
        --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,remote-claw-sandbox=venv_columnar} \
        --cf.BundleRepo-custom-checksum $(checksum) \
        --workers 123 --workflow htcondor
}

run_and_fetch_mcsyst_plots_boosted() {
    # separate command due to some tweaked plotting params in boosted categoriy
    local configs="${1:-$all_configs}"
    local categories="${2:-"sr__boosted"}"
    local variables="${3:-$ml_inputs}"

    local folder_name=${configs//,/_}
    echo "→ PlotShiftedVariables: config=$configs"
    run_and_fetch_cmd with_syst_unc/$folder_name claw run cf.PlotShiftedVariables1D \
        --configs "$configs" \
        --shift-sources all \
        --ml-models $all_models \
        --variables $variables \
        --processes ddl4 \
        --categories "$categories" \
        --general-settings data_mc_plots_not_blinded \
        --variable-settings boosted_rebin \
        --custom-style-config default_rax75 \
        --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,remote-claw-sandbox=venv_columnar} \
        --cf.BundleRepo-custom-checksum $(checksum) \
        --workers 123 --workflow htcondor
}

run_and_fetch_all_mcsyst_plots() {
    local configs="${1:-$all_configs}"
    # local categories="${2:-"incl,dycr,ttcr,sr,sr__resolved__1b,sr__resolved__2b"}"
    local categories="${2:-"sr,sr__resolved__1b,sr__resolved__2b"}"
    local variables="${3:-$ml_inputs}"
    local boosted_categories="sr__boosted"

    # split categories in boosted and non-boosted
    run_and_fetch_mcsyst_plots "$all_configs" "$categories" "$variables"
    run_and_fetch_mcsyst_plots_boosted "$all_configs" "$boosted_categories" "$variables"
    for config in ${configs//,/ }; do
        run_and_fetch_mcsyst_plots "$config" "$categories" "$variables"
        run_and_fetch_mcsyst_plots_boosted "$config" "$boosted_categories" "$variables"
    done
}

run_and_fetch_kinematics() {
    # kinematic features in different lepton channels
    for config in ${all_configs//,/ }; do
        run_and_fetch_mcstat_plots "$config" "sr,sr__2e,sr__2mu,sr__emu" "basic_kin"
    done
    run_and_fetch_mcstat_plots "$all_configs" "sr,sr__2e,sr__2mu,sr__emu" "basic_kin"

    # miscellaneous observables
    run_and_fetch_mcsyst_plots "$all_configs" "incl" "mli_mll"
    run_and_fetch_mcstat_plots "$all_configs" "sr" "mli_fj_particleNet_XbbVsQCD,mli_fj_particleNetWithMass_HbbvsQCD"
    run_and_fetch_mcstat_plots "$all_configs" "incl,sr,ttcr,dycr" "fatbjet0_pnet_hbb_pass_fail,fatbjet0_pnet_hbb"
}

run_and_fetch_btag_norm_sf() {
    local configs="${1:-$all_configs}"
    for config in ${configs//,/ }; do
        echo "→ BtagNormSF: config=$config"
        run_and_fetch_cmd btag_norm_sf/$config claw run hbw.GetBtagNormalizationSF --config $config
    done
}

run_and_fetch_all_yield_tables() {
    for config in ${all_configs//,/ }; do
        echo "→ YieldTables: configs=$config"
        run_and_fetch_yield_tables "$config"
    done
    echo "→ YieldTables: configs=$all_configs"
    run_and_fetch_yield_tables "$all_configs"
}

run_and_fetch_yield_tables() {
    # NOTE: in order to create consistent yield tables, we need to make categories order consistent
    local configs="${1:-$all_configs}"
    # Replace commas with underscores for folder name
    local folder_name=${configs//,/_}

    # vr categories with data and ratio
    run_and_fetch_cmd tables/$folder_name claw run hbw.CustomCreateYieldTable --remove-output 0,a,y --configs $configs --categories $vr_categories --processes table1 --ratio data,background --table-format latex_raw
    run_and_fetch_cmd tables/$folder_name claw run hbw.CustomCreateYieldTable --remove-output 0,a,y --configs $configs --categories $lep_categories --processes table1 --ratio data,background --table-format latex_raw
    run_and_fetch_cmd tables/$folder_name claw run hbw.CustomCreateYieldTable --remove-output 0,a,y --configs $configs --categories $jet_categories --processes table1 --ratio data,background --table-format latex_raw
    run_and_fetch_cmd tables/$folder_name claw run hbw.CustomCreateYieldTable --remove-output 0,a,y --configs $configs --categories $jet_categories_boosted_split --processes table1 --ratio data,background --table-format latex_raw

    # inference categories now also with data and ratio
    run_and_fetch_cmd tables/$folder_name claw run hbw.CustomCreateYieldTable --remove-output 0,a,y --configs $configs --categories $inf_categories_sig --processes table1 --table-format latex_raw
    run_and_fetch_cmd tables/$folder_name claw run hbw.CustomCreateYieldTable --remove-output 0,a,y --configs $configs --categories $inf_categories_bkg --processes table1 --table-format latex_raw
    # claw run hbw.CustomCreateYieldTable --config $config --categories $inf_categories --processes table4 --table-format latex_raw
}

run_gen_plots() {
    # pass arguments
    claw run cf.PlotVariables1D \
      --selector-steps no_sel_mask \
      --producers "combined_normalization_weights,gen_hbw_decay_features" \
      --hist-producer stitched_norm \
      --processes hbbhww_sm \
      --variables "gen_hbw*" \
      --categories incl \
      --shape-norm \
      --workers 6
}

run_and_fetch_gen_plots() {
    local base_cmd=(claw run cf.PlotVariables1D \
      --selector-steps no_sel_mask \
      --producers "combined_normalization_weights,gen_hbw_decay_features" \
      --hist-producer stitched_norm \
      --processes hbbhww_sm \
      --variables "gen_hbw.*" \
      --categories incl \
      --shape-norm \
      --workers 6
    )
    run_and_fetch_cmd gen_plots "${base_cmd[@]}"
    run_and_fetch_cmd gen_plots "${base_cmd[@]}" --plot-suffix linear --yscale linear
    run_and_fetch_cmd gen_plots_eff "${base_cmd[@]}" --plot-function columnflow.plotting.plot_functions_1d.plot_variable_efficiency --plot-suffix eff
    run_and_fetch_cmd gen_plots_eff "${base_cmd[@]}" --plot-function columnflow.plotting.plot_functions_1d.plot_variable_efficiency --plot-suffix eff_linear --yscale linear
}

calls() {
    # mbb, bb_pt, b1_pt, mbbllmet prefit shapes
    claw run cf.CreateDatacards --inference-model mbbllmet --configs $all_configs  --workers 6 --remove-output 0,a,y

    law run hbw.PlotPostfitShapes --version 2022_2023__mbbV15_noflow1 --fit-diagnostics-file /afs/desy.de/user/f/frahmmat/Projects/inference/data/store/MergePreAndPostFitShapes/hh_model_NNLOFix_13p6__model_default/datacards_aeaeb5cef4/m125.0/poi_r/2022_2023__mbbV15_noflow1/shapes_merged_23b6c2ab9a__unblinded__poi_r__params_r1.0_r_gghh1.0_r_qqhh1.0_kl1.0_kt1.0_CV1.0_C2V1.0__merged_prefit.root --inference-model mbb --processes ddl4 --general-settings dpostfit --custom-style-config dpostfit --prefit --merged-only False --remove-output 0,a,y
    law run hbw.PlotPostfitShapes --version 2022_2023__b1_ptV15_noflow1 --fit-diagnostics-file /afs/desy.de/user/f/frahmmat/Projects/inference/data/store/MergePreAndPostFitShapes/hh_model_NNLOFix_13p6__model_default/datacards_aeaeb5cef4/m125.0/poi_r/2022_2023__b1_ptV15_noflow1/shapes_merged_75e43eea1a__unblinded__poi_r__params_r1.0_r_gghh1.0_r_qqhh1.0_kl1.0_kt1.0_CV1.0_C2V1.0__merged_prefit.root --inference-model b1_pt --processes ddl4 --general-settings dpostfit --custom-style-config dpostfit --prefit --merged-only False --remove-output 0,a,y
    law run hbw.PlotPostfitShapes --version 2022_2023__bb_ptV15_noflow1 --fit-diagnostics-file /afs/desy.de/user/f/frahmmat/Projects/inference/data/store/MergePreAndPostFitShapes/hh_model_NNLOFix_13p6__model_default/datacards_aeaeb5cef4/m125.0/poi_r/2022_2023__bb_ptV15_noflow1/shapes_merged_7bd786756a__unblinded__poi_r__params_r1.0_r_gghh1.0_r_qqhh1.0_kl1.0_kt1.0_CV1.0_C2V1.0__merged_prefit.root --inference-model bb_pt --processes ddl4 --general-settings dpostfit --custom-style-config dpostfit --prefit --merged-only False --remove-output 0,a,y
    law run hbw.PlotPostfitShapes --version 2022_2023__mbbllmetV15_noflow2 --fit-diagnostics-file /afs/desy.de/user/f/frahmmat/Projects/inference/data/store/MergePreAndPostFitShapes/hh_model_NNLOFix_13p6__model_default/datacards_aeaeb5cef4/m125.0/poi_r/2022_2023__mbbllmetV15_noflow2/shapes_merged_b30a82c5cc__unblinded__poi_r__params_r1.0_r_gghh1.0_r_qqhh1.0_kl1.0_kt1.0_CV1.0_C2V1.0__merged_prefit.root --inference-model mbbllmet --processes ddl4 --general-settings dpostfit --custom-style-config dpostfit --prefit --merged-only False --remove-output 0,a,y


    # just a summary of calls that I often use
    law run hbw.PlotShiftedInferencePlots --inference-model default_unblind --configs $all_configs --processes ddl4 --custom-style-config legend_single_col --general-settings unstacked
    claw run hbw.PrepareInferenceTaskCalls --inference-model dl_jerc_bjet_uncorr1 --configs "c22prev14,c22postv14,c23prev14,c23postv14" --remove-output 0,a,y --workers 6
    claw run hbw.CustomCreateYieldTable --processes data,background,tt,dy,dy_lf_m10to50,dy_lf_m50toinf,dy_hf_m10to50,dy_hf_m50toinf,st,w_lnu,vv,ttv,h,other --ratio data,background,tt,dy --remove-output 0,a,y --categories "sr__{1b,2b}__ml_{sig_ggf,sig_vbf}"
    claw run hbw.CustomCreateYieldTable --processes data,background,tt,dy,st,w_lnu,vv,ttv,h,other --ratio data,background,tt,dy --remove-output 0,a,y
    claw run cf.PlotVariables1D --processes ddl4 --variables mli_fj_particleNetWithMass_HbbvsQCD --categories incl,sr,dycr,ttcr --hist-hooks cumsum_reverse --remove-output 0,a,y --general-settings data_mc_plots_not_blinded --plot-suffix test1 --local-scheduler --configs $all_configs --shape-norm --plot-function columnflow.plotting.plot_functions_1d.plot_stack_test

    # making plots of the two selected VBF jets
    law run cf.PlotShiftedVariables1D --processes ddl4 --producers event_weights,pre_ml_cats,dl_ml_inputs,vbf_jets --variables "vbfjet{1,2}_{pt,eta}" --workers 123 --cf.MergeShiftedHistograms-{pilot,remote-claw-sandbox=venv_columnar,workflow=htcondor} --configs $all_configs --categories sr --shift-sources all
}

run_and_fetch_efficiency_plots() {
    local config_groups="${1:-$all_configs}"
    local categories="${2:-"incl,sr,dycr,ttcr"}"
    local variables="${3:-"fatbjet0_particlenetwithmass_hbbvsqcd"}"

    for configs in $config_groups; do
        local folder_name=${configs//,/_}
        echo "→ PlotVariables: config=$configs, categories=$categories, variables=$variables"
        run_and_fetch_cmd hbb_efficiency_pt300/$folder_name claw run cf.PlotVariables1D \
            --configs "$configs" \
            --ml-models $all_models \
            --variables $variables \
            --processes ddl4 \
            --categories "$categories" \
            --general-settings data_mc_plots_not_blinded \
            --plot-function columnflow.plotting.plot_functions_1d.plot_variable_efficiency \
            --remove-output 0,a,y \
            --plot-suffix efficiency \
            --workers 6 \
            --cf.CreateHistograms-pilot
    done
}

run_and_fetch_triggersf() {
    # TODO: the fetching seems to be rather inconsistent, changing hash every time?
    local configs="${1:-$all_configs}"
    local variables="${2:-"trg_lepton0_pt-trg_lepton1_pt-trig_ids"}"
    local processes="${3:-"data_met,sf_bkg_reduced"}"
    local uncs="${4:-"False"}"

    local suffix="V6"
    if [[ "$uncs" != "False" ]]; then
        suffix="unc_V6"
    fi

    local folder_name=triggersf/${configs//,/_}
    run_and_fetch_cmd $folder_name claw run hbw.ComputeTriggerSF \
        --configs $configs \
        --variables "$variables" \
        --processes "$processes" \
        --suffix "$suffix" \
        --plot-uncertainties "$uncs" \
        --cf.ReduceEvents-{workflow=htcondor,pilot,remote-claw-sandbox=venv_columnar} \
        --cf.CreateHistograms-{workflow=local,pilot,remote-claw-sandbox=venv_columnar,htcondor-memory=3GB} \
        --cf.MergeHistograms-{workflow=local,remote-claw-sandbox=venv_columnar,htcondor-memory=3GB} \
        --cf.BundleRepo-custom-checksum "$(checksum)" \
        --workers 6
}

run_and_fetch_all_triggersf() {
    local configs="${1:-$all_configs}"
    local variables="${2:-"trg_lepton0_pt-trig_ids"}"
    local processes="${3:-"data_met,sf_bkg_reduced"}"
    for config in ${all_configs//,/ }; do
        run_and_fetch_triggersf "$config" "$variables" "$processes" &
        pids+=($!)
    done
    # Wait for all config processes to finish
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    echo "All trigger SF fetches per config completed."
    run_and_fetch_triggersf "$all_configs" "$variables" "$processes"
}
run_and_fetch_twodim_triggersf() {
    # running 2D trigger SF as well as uncertainties (no parallel here due to high memory consumption)
    local configs="${1:-$all_configs}"
    local variables="${2:-"trg_lepton0_pt-trg_lepton1_pt-trig_ids"}"
    local processes="${3:-"data_met,sf_bkg_reduced"}"
    for config in ${all_configs//,/ }; do
        run_and_fetch_triggersf "$config" "$variables" "$processes"
        run_and_fetch_triggersf "$config" "$variables" "$processes" "True"
    done
    echo "All trigger SF fetches per config completed."
    run_and_fetch_triggersf "$all_configs" "$variables" "$processes"
    run_and_fetch_triggersf "$all_configs" "$variables" "$processes" "True"
}

run_triggersf_production() {
    # Full Trigger SF production workflow
    local configs="${1:-$all_configs}"
    local variables="${2:-"trg_lepton0_pt-trg_lepton1_pt-trig_ids"}"
    local datasets="${3:-"tt_*_powheg,data_met*,st_twchannel*dl*,dy_m50toinf*"}"
    local processes="${4:-"data_met,sf_bkg_reduced"}"

    # defaults that are encoded in the ComputeTriggerSF task
    local reducer="triggersf"
    local producers="event_weights,pre_ml_cats,trigger_prod_dls"
    local hist_producers="no_trig_sf,dl_orth2_with_l1_seeds"

    # handles to turn on/off parts of the workflow
    local do_reduction="true"
    local produce_histograms="true"
    local produce_scale_factors="true"

    if [[ "$do_reduction" == "true" ]]; then
        echo "→ TriggerSF MergeReducedEvents: configs=$configs, datasets=$datasets"
        run_cmd law run cf.MergeReducedEventsWrapper --configs $configs --datasets $datasets \
            --cf.MergeReducedEvents-reducer $reducer \
            --cf.MergeReducedEvents-{workflow=htcondor,pilot} \
            --workers 156 --cf.BundleRepo-custom-checksum $(checksum)
    fi

    if [[ "$produce_histograms" == "true" ]]; then
        for hist_producer in ${hist_producers//,/ }; do
            echo "→ TriggerSF CreateHistograms: configs=$configs, datasets=$datasets"
                run_cmd law run cf.CreateHistogramsWrapper \
                    --configs "$configs" \
                    --datasets "$datasets" \
                    --shifts nominal \
                    --cf.CreateHistograms-reducer "$reducer" \
                    --cf.CreateHistograms-producer "$producers" \
                    --cf.CreateHistograms-variables "$variables" \
                    --cf.CreateHistograms-hist-producer "$hist_producer" \
                    --cf.CreateHistograms-{workflow=htcondor,pilot,remote-claw-sandbox=venv_columnar} \
                    --cf.BundleRepo-custom-checksum "$(checksum)" \
                    --workers 20
        done
    fi

    if [[ "$produce_scale_factors" == "true" ]]; then
        run_and_fetch_all_triggersf "$configs" "$variables" "$processes"
    fi
}

run_and_fetch_all_plots() {

    # run_and_fetch_mcsyst_plots "$all_configs" "sr" "ml_inputs,mli_full_vbf_tag"

    # run_and_fetch_gen_plots
    # run_and_fetch_all_mcsyst_plots
    # run_and_fetch_mcsyst_plots "$all_configs" "incl" "mli_mll"
    # run_and_fetch_mcsyst_plots "$all_configs" "$inf_categories_sig_resolved" "$ml_inputs"
    # run_and_fetch_mcsyst_plots_boosted "$all_configs" "$inf_categories_sig_boosted" "$ml_inputs"

    # run_and_fetch_mcsyst_plots "$all_configs" "sr,sr__resolved__1b,sr__resolved__2b" "mli_full_vbf_tag,mli_full_vbf_mass,mli_full_vbf_pt,mli_full_vbf_eta,mli_full_vbf_phi,mli_full_vbf_deta,mli_vbfcand1_pt,mli_vbfcand2_pt,mli_vbfcand1_eta,mli_vbfcand2_eta"
    # run_and_fetch_mcsyst_plots_boosted "$all_configs" "sr__boosted" "mli_full_vbf_tag,mli_full_vbf_mass,mli_full_vbf_pt,mli_full_vbf_eta,mli_full_vbf_phi,mli_full_vbf_deta,mli_vbfcand1_pt,mli_vbfcand2_pt,mli_vbfcand1_eta,mli_vbfcand2_eta"
    # run_and_fetch_all_mcsyst_plots "$all_configs" "sr,sr__resolved__1b,sr__resolved__2b" "mli_full_vbf_tag,mli_full_vbf_mass,mli_full_vbf_pt,mli_full_vbf_eta,mli_full_vbf_phi,mli_full_vbf_deta,mli_vbfcand1_pt,mli_vbfcand2_pt,mli_vbfcand1_eta,mli_vbfcand2_eta"

    # run_and_fetch_mcsyst_plots "$all_configs" "$inf_categories_sig_resolved" "$ml_inputs"
    # run_and_fetch_mcsyst_plots_boosted "$all_configs" "$inf_categories_sig_boosted" "$ml_inputs"

    # run_and_fetch_all_mlscore_plots
    # run_and_fetch_all_yield_tables
    # run_and_fetch_btag_norm_sf
    # run_and_fetch_mlplots
    # fetch_ml_metrics

    # run_and_fetch_kinematics

    # run_and_fetch_mcstat_plots "$all_configs" "incl,sr,ttcr,dycr" "fatbjet0_pnet_hbb"
    # run_and_fetch_mcstat_plots "$all_configs" "incl,sr,ttcr,dycr" "fatbjet0_pnet_hbb,fatjet0_particlenet_xbbvsqcd_pass_fail,fatjet0_particlenetwithmass_hbbvsqcd_pass_fail,fatbjet0_particlenet_xbbvsqcd_pass_fail,fatbjet0_particlenetwithmass_hbbvsqcd_pass_fail"
    # run_and_fetch_efficiency_plots "$all_configs" "sr,sr__ml_bkg,sr__ml_sig_ggf,sr__ml_sig_vbf" "fatbjet0_pnet_hbb"
    # run_and_fetch_efficiency_plots "$configs_sep $all_configs" "sr,sr__ml_bkg,sr__ml_sig_ggf,sr__ml_sig_vbf" "fatbjet0_pnet_hbb"
}

run_dycorr_plots() {
    # to be run without any ml models
    claw run cf.PlotShiftedVariables1D --processes ddl4 --variables "mli_ll_pt,mli_n_jet,mli_mll" --workers 123 --categories dycr__1b,dycr__2b,dycr__boosted,dycr,dycr__2mu,dycr__2e,dycr__emu --hist-producer with_dy_corr --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,remote-claw-sandbox=venv_columnar} --shift-sources all
    claw run cf.PlotShiftedVariables1D --processes ddl4 --variables "mli_ll_pt,mli_n_jet,mli_mll" --workers 123 --categories sr__1b,sr__2b,sr__boosted,sr,sr__2mu,sr__2e --hist-producer with_dy_corr --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,remote-claw-sandbox=venv_columnar} --shift-sources all
    claw run cf.PlotShiftedVariables1D --processes ddl4 --variables "mli_ll_pt,mli_n_jet,mli_mll" --workers 123 --categories sr__1b,sr__2b,sr__boosted,sr,sr__2mu,sr__2e --hist-producer with_trigger_weight --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,remote-claw-sandbox=venv_columnar} --shift-sources all


    # to trigger all MergeShiftedHistograms at once.
    claw run cf.PlotShiftedVariables1D --configs $all_configs --processes "ddl4,hh_*" --variables "mli_ll_pt,mli_n_jet,mli_mll" --workers 123 --categories dycr --hist-producer with_dy_corr --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,remote-claw-sandbox=venv_columnar} --shift-sources all
    claw run cf.PlotShiftedVariables1D --configs $all_configs --processes "ddl4,hh_*" --variables "mli_ll_pt,mli_n_jet,mli_mll" --workers 123 --categories dycr --hist-producer with_trigger_weight --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,remote-claw-sandbox=venv_columnar} --shift-sources all

    claw run cf.PlotShiftedVariables1D --ml-models multiclassv3,ggfv3,vbfv3_tag --configs $all_configs --processes "ddl4" --variables "ml_inputs,mli_full_vbf_tag" --workers 123 --categories sr --hist-producer met_geq40_incl_dy_corr --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,no-poll,remote-claw-sandbox=venv_columnar} --shift-sources all
    claw run cf.PlotShiftedVariables1D --ml-models multiclassv3,ggfv3,vbfv3_tag --configs $all_configs --processes "ddl4,hh_*" --variables "mlscore.*,rebinlogit_mlscore.sig*binary,logit_mlscore.sig*binary" --workers 123 --categories sr --hist-producer met_geq40_incl_dy_corr --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,no-poll,remote-claw-sandbox=venv_columnar} --shift-sources all
    claw run cf.PlotShiftedVariables1D --ml-models multiclassv3,ggfv3,vbfv3_tag --configs $all_configs --processes "ddl4,hh_*" --variables "pas" --workers 12 --categories sr --hist-producer met_geq40_incl_dy_corr --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,no-poll,remote-claw-sandbox=venv_columnar} --shift-sources all
}

run_triggersf() {
    run_and_fetch_twodim_triggersf "$all_configs" "trg_lepton0_pt-trg_lepton1_pt-trig_ids" "data_met,sf_bkg_reduced"
    run_and_fetch_all_triggersf "$all_configs" "trg_lepton0_pt-trig_ids" "data_met,sf_bkg_reduced"
    run_and_fetch_all_triggersf "$all_configs" "trg_lepton1_pt-trig_ids" "data_met,sf_bkg_reduced"
    run_and_fetch_all_triggersf "$all_configs" "lepton0_eta-trig_ids" "data_met,sf_bkg_reduced"
    run_and_fetch_all_triggersf "$all_configs" "trg_n_jet-trig_ids" "data_met,sf_bkg_reduced"
}

# === Dispatcher ===
# TODO: this is still super messy (comment in & out whatever is currently requested),
# we might want to implement a proper dispatcher at some point
run_all() {
    # Set global checksum once for this entire workflow run
    global_checksum=$(checksum)
    run_cmd law run cf.BundleRepo --custom-checksum "$global_checksum"
    # recreate_campaign_summary

    # run_merge_reduced_events "$all_configs" "nominal"
    # run_merge_selection_stats "$all_configs" "nominal"

    # prepare_dy_corr "$all_configs"
    # run_ml_training
    # run_ml_training "multiclass_met40,ggf_met40,vbf_met40"
    # run_ml_training "vbfv3_mqq,multiclassv3_mqq,vbfv3_tag,multiclassv3_tag"
    # prepare_mlcolumns "$all_configs" "$nominal" "$dnn_multiclass,$dnn_ggf,$dnn_vbf" "$ml_scores"

    # run_merge_shifted_histograms_htcondor "$all_configs" "$all" "multiclassv3,ggfv3,vbfv3" "$ml_inputs,$all_ml_scores,mli_full_vbf_tag,mli_full_vbf_mass"

    # prepare_mlcolumns "$all_configs" "$nominal" "multiclass_met40,ggf_met40,vbf_met40" "$ml_scores"
    # prepare_mlcolumns "$all_configs" "$jerc_shifts" "multiclass_met40,ggf_met40,vbf_met40" "$ml_scores"
    # run_ml_training "multiclassv3_vbf_extended"
    # run_ml_training "vbfv3_vbf_extended"

    # prepare_mlcolumns "$all_configs" "$nominal" "multiclassv3,ggfv3,vbfv3" "$ml_scores"
    # prepare_mlcolumns "$all_configs" "$jerc_shifts" "multiclassv3,ggfv3,vbfv3" "$ml_scores"

    # prepare_mlcolumns "$all_configs" "$nominal" "multiclassv3,ggfv3,vbfv3_vbf_extended" "$ml_scores"
    # prepare_mlcolumns "$all_configs" "$jerc_shifts" "multiclassv3,ggfv3,vbfv3_vbf_extended" "$ml_scores"
    # run_datacards "vbfextended_unblind" $all_configs

    # run_merge_reduced_events "$all_configs" "jec_Total_up,jec_Total_down,jer_up,jer_down"
    # run_merge_selection_stats "$all_configs" "jec_Total_up,jec_Total_down,jer_up,jer_down"

    # prepare_mlcolumns "$all_configs" "nominal" "$dnn_multiclass,$dnn_ggf,$dnn_vbf" "$ml_inputs"
    # prepare_mlcolumns "$all_configs" "$jerc_shifts" "$dnn_multiclass,$dnn_ggf,$dnn_vbf" "$ml_inputs"

    # run_merge_shifted_histograms_htcondor "$all_configs" "$all" "multiclassv3_mqq,ggfv3,vbfv3_mqq"
    # run_merge_shifted_histograms_htcondor "$all_configs" "$all" "multiclassv3,ggfv3,vbfv3" "ml_inputs,ml_inputs_vbf_extended,mli_full_vbf_tag,mli_full_vbf_mass"

    # run_merge_shifted_histograms_htcondor "$all_configs" "$all" "multiclassv3_tag,ggfv3,vbfv3_tag"
    # run_datacards "vbfmqq1_unblind" $all_configs
    # run_datacards "vbftag1_unblind" $all_configs

    # run_merge_shifted_histograms_htcondor "$all_configs" "$all" "multiclassv3,ggfv3,vbfv3" "mli_full_vbf_tag,mli_full_vbf_mass,mli_full_vbf_deta,mli_vbfcand1_pt,mli_vbfcand2_pt,mli_vbfcand1_eta,mli_vbfcand2_eta"
    # run_merge_shifted_histograms_htcondor "$all_configs" "$all" "multiclassv3,ggfv3,vbfv3" "mli_full_vbf_mass,mli_full_vbf_pt,mli_full_vbf_eta,mli_full_vbf_phi"

    # run_and_fetch_all_plots
}

# === Example usage ===
# dry_run=true ./workflow.sh
# ./workflow.sh

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && run_all "$@"
