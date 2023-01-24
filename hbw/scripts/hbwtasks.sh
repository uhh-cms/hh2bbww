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
	--cf.ReduceEvents-htcondor-share-software True \
	--cf.ReduceEvents-tasks-per-job 1
}

processes="default"
categories="resolved,boosted,inclusive"
producers="ml_inputs"
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
	--tasks-per-job 1
}
