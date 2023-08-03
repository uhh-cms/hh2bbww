#!/bin/sh
# small script to source to quickly run tasks

# versioning
version="v1"

# possible config choices: "c17", "l17"
# NOTE: use "l17" for testing purposes
config="c17"
datasets="*"

# NOTE: running this starts quite a lot of jobs, so only submit if everything is ready
hbw_reduction(){
    law run cf.ReduceEventsWrapper --version $version --workers 50 \
	--configs $config \
	--shifts nominal \
	--datasets $datasets \
	--skip-datasets "qqHH*" \
	--cf.ReduceEvents-workflow htcondor \
	--cf.ReduceEvents-pilot \
	--cf.ReduceEvents-no-poll \
	--cf.ReduceEvents-parallel-jobs 4000 \
	--cf.ReduceEvents-retries 1 \
	--cf.ReduceEvents-tasks-per-job 1 \
	--cf.ReduceEvents-job-workers 1 \
	$@
}

hbw_cutflow(){
    for steps in "resolved" "boosted"
    do
	law run cf.PlotCutflow --version $version --workers 4 \
	    --config l17 \
	    --selector-steps $steps \
	    --shift nominal \
	    --processes with_qcd \
	    --process-settings unstack_all \
	    --shape-norm True --yscale log --cms-label simpw \
	    --remove-output 0,a,y --view-cmd imgcat \
	    $@
    done
}

ml_model="test"
ml_datasets="ml"

hbw_ml_training(){
    law run cf.MLTraining --version $version --workers 10 \
	--ml-model $ml_model \
	--workflow htcondor \
	--htcondor-gpus 1 \
	--htcondor-memory 40000 \
	--max-runtime 48h \
	--cf.MergeMLEvents-workflow local \
	--cf.PrepareMLEvents-workflow htcondor \
	--cf.PrepareMLEvents-htcondor-gpus 0 \
	--cf.PrepareMLEvents-htcondor-memory 4000 \
	--cf.PrepareMLEvents-max-runtime 3h \
	--cf.PrepareMLEvents-pilot True \
	--retries 1 \
	$@
}

hbw_ml_preparation(){
    for i in {0..4}
    do
	law run cf.MergeMLEventsWrapper --version $version --workers 10 \
	    --configs $config \
	    --cf.MergeMLEvents-ml-model $ml_model \
	    --cf.MergeMLEvents-fold $i \
	    --datasets $ml_datasets \
	    --skip-datasets "qqHH*,data*" \
	    --cf.MergeMLEvents-workflow local \
	    --cf.PrepareMLEvents-pilot True \
	    --cf.MergeMLEvents-parallel-jobs 4000 \
	    --cf.MergeMLEvents-retries 1 \
	    --cf.MergeMLEvents-tasks-per-job 1 \
	    --cf.MergeMLEvents-job-workers 1 \
	    $@
    done
}

processes="default"
categories="resolved,boosted,incl"
variables="mli_*"

# NOTE: running this starts quite a lot of jobs, so only submit if everything is ready
hbw_plot_variables(){
    law run cf.PlotVariables1D --version $version --workers 50 \
	--config $config \
	--processes $processes \
	--variables $variables \
	--categories $categories \
	--process-settings unstack_signal --shape-norm True --yscale log --cms-label simpw --skip-ratio True \
	--workflow htcondor \
	--pilot True \
	--parallel-jobs 4000 \
	--htcondor-share-software True \
	--tasks-per-job 1 \
	--job-workers 1 \
	$@
}

ml_model="default"
ml_output_variables="mlscore.*"
ml_categories="resolved,boosted,incl,ml_ggHH_kl_1_kt_1_sl_hbbhww,ml_tt,ml_st,ml_w_lnu,ml_dy_lep"

hbw_plot_ml_nodes(){
    law run cf.PlotVariables1D --version $version --workers 50 \
	--config $config \
	--ml-models $ml_model \
	--processes $processes \
	--variables $ml_output_variables \
	--categories $ml_categories \
	--process-settings unstack_signal --shape-norm True --yscale log --cms-label simpw --skip-ratio True \
	--workflow htcondor \
	--pilot True \
	--retries 1 \
    $@
}

hbw_control_plots_noData_much(){
    law run cf.PlotVariables1D --version $version --workers 50 \
	--config $config \
	--producers features \
	--processes much \
	--process-settings scale_signal \
	--variables "*" \
	--categories "1mu,1mu__resolved,1mu__boosted" \
	--yscale log --skip-ratio True --cms-label simpw \
	--workflow htcondor \
	$@
}

hbw_control_plots_much(){
    law run cf.PlotVariables1D --version $version --workers 50 \
	--config $config \
	--producers features \
	--processes dmuch \
	--process-settings scale_signal \
	--variables "*" \
	--categories "1mu,1mu__resolved,1mu__boosted" \
	--yscale log --cms-label pw \
	--workflow htcondor \
	$@
}

hbw_datacards(){
    law run cf.CreateDatacards --version $version --workers 50 \
	--config $config \
	--producers ml_inputs --ml-models $ml_model \
	--pilot --workflow htcondor \
	--retries 1 \
	$@
}
