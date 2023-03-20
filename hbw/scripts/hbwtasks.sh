# small script to source to quickly run tasks

# versioning
version="v1"

# default calibrator, selector
calibrators="skip_jecunc"
selector="default"

# possible config choices: "config_2017", "config_2017_limited"
# NOTE: use "config_2017_limited" for testing purposes
config="config_2017"
datasets="*"

# NOTE: running this starts quite a lot of jobs, so only submit if everything is ready
hbw_reduction(){
    law run cf.ReduceEventsWrapper --version $version --workers 50 \
	--configs $config \
	--cf.ReduceEvents-calibrators $calibrators --cf.ReduceEvents-selector $selector \
	--shifts nominal \
	--datasets $datasets \
	--skip-datasets "qqHH*" \
	--cf.ReduceEvents-workflow htcondor \
	--cf.ReduceEvents-pilot \
	--cf.ReduceEvents-no-poll \
	--cf.ReduceEvents-parallel-jobs 4000 \
	--cf.ReduceEvents-retries 1 \
	--cf.ReduceEvents-tasks-per-job 1 \
	--cf.ReduceEvents-job-workers 1
	#--cf.ReduceEvents-htcondor-share-software True
}

hbw_cutflow(){
    for steps in "resolved" "boosted"
    do
	law run cf.PlotCutflow --version $version --workers 4 \
	    --config config_2017_limited \
	    --calibrators $calibrators \
	    --selector $selector \
	    --selector-steps $steps \
	    --shift nominal \
	    --processes with_qcd \
	    --process-settings unstack_all \
	    --shape-norm True --yscale log --cms-label simpw \
	    --remove-output 0,a,y --view-cmd imgcat
	    # --workflow htcondor --pilot True --retries 1
    done
}

producers="ml_inputs"
ml_model="test"
# ml_datasets="ggHH_kl_1_kt_1_sl_hbbhww_powheg,tt_sl_powheg,tt_fh_powheg,st_tchannel_t_powheg"
ml_datasets="ml"

hbw_ml_preparation(){
    for i in {0..4}
    do
	law run cf.MergeMLEventsWrapper --version $version --workers 10 \
	    --configs $config \
	    --cf.MergeMLEvents-producers $producers \
	    --cf.MergeMLEvents-ml-model $ml_model \
	    --cf.MergeMLEvents-fold $i \
	    --datasets $ml_datasets \
	    --skip-datasets "qqHH*,data*" \
	    --cf.MergeMLEvents-workflow local \
	    --cf.PrepareMLEvents-pilot True \
	    --cf.MergeMLEvents-parallel-jobs 4000 \
	    --cf.MergeMLEvents-retries 1 \
	    --cf.MergeMLEvents-tasks-per-job 1 \
	    --cf.MergeMLEvents-job-workers 1
    done
}

processes="default"
categories="resolved,boosted,incl"
variables="mli_*"

# NOTE: running this starts quite a lot of jobs, so only submit if everything is ready
hbw_plot_variables(){
    law run cf.PlotVariables1D --version $version --workers 50 \
	--config $config \
	--calibrators $calibrators --selector $selector \
	--producers $producers \
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
	--remove-output 0,a,y
}

ml_model="default"
ml_output_variables="default.*"
ml_categories="resolved,boosted,incl,ml_ggHH_kl_1_kt_1_sl_hbbhww,ml_tt,ml_st,ml_w_lnu,ml_dy_lep"

hbw_plot_ml_nodes(){
    law run cf.PlotVariables1D --version $version --workers 50 \
	--config $config \
	--calibrators $calibrators --selector $selector \
	--producers $producers \
	--ml-models $ml_model \
	--processes $processes \
	--variables $ml_output_variables \
	--categories $ml_categories \
	--process-settings unstack_signal --shape-norm True --yscale log --cms-label simpw --skip-ratio True \
	--workflow htcondor \
	--pilot True \
	--retries 1 \
    	--remove-output 0,a,y
}

hbw_control_plots_noData_much(){
    law run cf.PlotVariables1D --version $version --workers 50 \
	--config $config \
	--calibrators $calibrators --selector $selector \
	--producers features \
	--processes much \
	--process-settings scale_signal \
	--variables "*" \
	--categories "1mu,1mu__resolved,1mu__boosted" \
	--yscale log --skip-ratio True --cms-label simpw \
	--workflow htcondor \
	--remove-output 0,a,y
}

hbw_control_plots_much(){
    law run cf.PlotVariables1D --version $version --workers 50 \
	--config $config \
	--calibrators $calibrators --selector $selector \
	--producers features \
	--processes dmuch \
	--process-settings scale_signal \
	--variables "*" \
	--categories "1mu,1mu__resolved,1mu__boosted" \
	--yscale log --cms-label pw \
	--workflow htcondor \
	--remove-output 0,a,y
}

hbw_datacards(){
    law run cf.CreateDatacards --version $version --workers 50 \
	--config $config \
	--calibrators $calibrators --selector $selector \
	--producers ml_inputs --ml-models default \
	--pilot --workflow htcondor \
	--retries 1
}
