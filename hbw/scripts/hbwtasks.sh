#!/bin/sh
# small script to source to quickly run tasks

# versioning and custom checksum (checksum only used for reduction, ml_training, datacards)
version="prod1"
checksum="prod1"

# possible config choices: "c17", "l17"
# NOTE: use "l17" for testing purposes
config="c17"
datasets="*"

hbw_selection(){
    law run cf.SelectEvents --version $version \
	--config $config \
	$@
}

#
# Production tasks (will submit jobs and use cf.BundleRepo outputs based on the checksum)
#

hbw_reduction(){
    law run cf.ReduceEventsWrapper --version $version --workers 20 \
	--configs $config \
	--shifts nominal \
	--datasets $datasets \
	--cf.ReduceEvents-workflow htcondor \
	--cf.ReduceEvents-pilot \
	--cf.ReduceEvents-no-poll \
	--cf.ReduceEvents-parallel-jobs 4000 \
	--cf.ReduceEvents-retries 1 \
	--cf.ReduceEvents-tasks-per-job 1 \
	--cf.ReduceEvents-job-workers 1 \
	--cf.BundleRepo-custom-checksum $checksum \
	$@
}

hbw_merge_reduction(){
    law run cf.MergeReducedEventsWrapper --version $version --workers 20 \
	--configs $config \
	--shifts nominal \
	--datasets $datasets \
	--cf.ReduceEvents-workflow htcondor \
	--cf.ReduceEvents-pilot \
	--cf.ReduceEvents-parallel-jobs 4000 \
	--cf.ReduceEvents-retries 1 \
	--cf.ReduceEvents-tasks-per-job 1 \
	--cf.ReduceEvents-job-workers 1 \
	--cf.BundleRepo-custom-checksum $checksum \
	$@
}

ml_model="dense_default"

hbw_ml_training(){
    law run cf.MLTraining --version $version --workers 20 \
	--configs $config \
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
	--cf.MergeReducedEvents-workflow local \
	--cf.MergeReductionStats-n-inputs -1 \
	--cf.ReduceEvents-workflow htcondor \
	--cf.ReduceEvents-pilot True \
	--cf.BundleRepo-custom-checksum $checksum \
	--retries 2 \
	$@
}

inference_model="rates_only"

hbw_datacards(){
    law run cf.CreateDatacards --version $version --workers 20 \
	--config $config \
	--inference-model $inference_model \
	--pilot --workflow htcondor \
	--retries 2 \
	--cf.MLTraining-htcondor-gpus 1 \
	--cf.MLTraining-htcondor-memory 40000 \
	--cf.MLTraining-max-runtime 48h \
	--cf.MergeMLEvents-workflow local \
	--cf.PrepareMLEvents-workflow htcondor \
	--cf.PrepareMLEvents-htcondor-gpus 0 \
	--cf.PrepareMLEvents-htcondor-memory 4000 \
	--cf.PrepareMLEvents-max-runtime 3h \
	--cf.MergeReducedEvents-workflow local \
	--cf.MergeReductionStats-n-inputs -1 \
	--cf.ReduceEvents-workflow htcondor \
	--cf.ReduceEvents-pilot True \
	--cf.SelectEvents-workflow htcondor \
	--cf.BundleRepo-custom-checksum $checksum \
	$@
}

hbw_rebin_datacards(){
	# same as `hbw_datacards`, but also runs the rebinning task
	law run hbw.ModifyDatacardsFlatRebin --version $version --workers \
	--config $config \
	--inference-model $inference_model \
	--pilot --workflow htcondor \
	--retries 2 \
	--cf.MLTraining-htcondor-gpus 1 \
	--cf.MLTraining-htcondor-memory 40000 \
	--cf.MLTraining-max-runtime 48h \
	--cf.MergeMLEvents-workflow local \
	--cf.PrepareMLEvents-workflow htcondor \
	--cf.PrepareMLEvents-htcondor-gpus 0 \
	--cf.PrepareMLEvents-htcondor-memory 4000 \
	--cf.PrepareMLEvents-max-runtime 3h \
	--cf.MergeReducedEvents-workflow local \
	--cf.MergeReductionStats-n-inputs -1 \
	--cf.ReduceEvents-workflow htcondor \
	--cf.ReduceEvents-pilot True \
	--cf.SelectEvents-workflow htcondor \
	--cf.BundleRepo-custom-checksum $checksum \
	$@
}

#
# Plotting tasks (no assumptions on workers, workflow etc.)
#

hbw_cutflow(){
    for steps in "resolved" "boosted"
    do
	law run cf.PlotCutflow --version $version \
	    --config l17 \
	    --selector-steps $steps \
	    --shift nominal \
	    --processes with_qcd \
	    --process-settings unstack_all \
	    --shape-norm True --yscale log --cms-label simpw \
	    --view-cmd imgcat \
	    $@
    done
}

processes="default"
categories="resolved,boosted,incl"
variables="mli_*"

hbw_plot_variables(){
    law run cf.PlotVariables1D --version $version \
	--config $config \
	--processes $processes \
	--variables $variables \
	--categories $categories \
	--process-settings unstack_signal --shape-norm True --yscale log --cms-label simpw --skip-ratio True \
	$@
}

ml_output_variables="mlscore.*"
ml_categories="resolved,boosted,incl,ml_ggHH_kl_1_kt_1_sl_hbbhww,ml_tt,ml_st,ml_w_lnu,ml_dy_lep"

hbw_plot_ml_nodes(){
    law run cf.PlotVariables1D --version $version \
	--config $config \
	--ml-models $ml_model \
	--processes $processes \
	--variables $ml_output_variables \
	--categories $ml_categories \
	--process-settings unstack_signal --shape-norm True --yscale log --cms-label simpw --skip-ratio True \
    $@
}

hbw_control_plots_noData_much(){
    law run cf.PlotVariables1D --version $version \
	--config $config \
	--producers features \
	--processes much \
	--process-settings scale_signal \
	--variables "*" \
	--categories "1mu,1mu__resolved,1mu__boosted" \
	--yscale log --skip-ratio True --cms-label simpw \
	$@
}

hbw_control_plots_much(){
    law run cf.PlotVariables1D --version $version \
	--config $config \
	--producers features \
	--processes dmuch \
	--process-settings scale_signal \
	--variables "*" \
	--categories "1mu,1mu__resolved,1mu__boosted" \
	--yscale log --cms-label pw \
	$@
}
