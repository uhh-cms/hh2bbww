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
	--cf.ReduceEvents-pilot True \
	--cf.ReduceEvents-parallel-jobs 4000 \
	--cf.ReduceEvents-retries 1 \
	--cf.ReduceEvents-tasks-per-job 1 \
	--cf.ReduceEvents-job-workers 1
	#--cf.ReduceEvents-htcondor-share-software True
}

producers="ml_inputs"
ml_model="test"
# ml_datasets="ggHH_kl_1_kt_1_sl_hbbhww_powheg,tt_sl_powheg,tt_fh_powheg,st_tchannel_t_powheg"
ml_datasets="ml"

hbw_ml_preparation(){
    law run cf.MergeMLEventsWrapper --version $version --workers 10 \
	--configs $config \
	--cf.MergeMLEvents-producers $producers \
	--cf.MergeMLEvents-ml-model $ml_model \
	--datasets $ml_datasets \
	--skip-datasets "qqHH*,data*" \
	--cf.MergeMLEvents-workflow htcondor \
	--cf.PrepareMLEvents-pilot True \
	--cf.MergeMLEvents-parallel-jobs 4000 \
	--cf.MergeMLEvents-retries 1 \
	--cf.MergeMLEvents-tasks-per-job 1 \
	--cf.MergeMLEvents-job-workers 1
}

processes="default"
categories="resolved,boosted,inclusive"
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
	--process-settings unstack_signal --shape-norm True --yscale log \
	--workflow htcondor \
	--pilot True \
	--parallel-jobs 4000 \
	--htcondor-share-software True \
	--tasks-per-job 1 \
	--job-workers 1
}
