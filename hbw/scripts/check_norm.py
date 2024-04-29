"""
Script to check normalization of datasets. Can be called via:
cf_sandbox venv_columnar_dev "python check_norm.py"
"""

import law

from columnflow.tasks.framework.base import AnalysisTask
from columnflow.tasks.production import ProduceColumns

import numpy as np
import awkward as ak

# setup objects of interest
version = "kl_test_9"
analysis = law.config.get_expanded("analysis", "default_analysis")
# config = law.config.get_expanded("analysis", "default_config")
config = "c17"
datasets = [f"ggHH_kl_{kl}_kt_1_dl_hbbhww_powheg" for kl in ["0", "1", "2p45", "5"]]
producer = "event_weights"

analysis_inst = AnalysisTask.get_analysis_inst(analysis)
config_inst = analysis_inst.get_config(config)


for dataset in datasets:
    dataset_inst = config_inst.get_dataset(dataset)
    process_inst = dataset_inst.processes.get_first()

    xsec = process_inst.xsecs[config_inst.campaign.ecm].nominal
    n_tot = dataset_inst.n_events

    task = ProduceColumns(
        version=version, 
        analysis=analysis,
        config=config,
        dataset=dataset,
        producer=producer,
        walltime="1h", 
    )
    print("running task", task)
    task.law_run()

    outp = task.output()["collection"]

    sum_w = 0
    n_selected = 0

    for branch in outp.keys():
        for key, item in outp[branch].items():
            print("=============== Output:", key)
            if "parquet" in item.fn:
                events = ak.from_parquet(item.fn)
                
                n_selected += len(events)
                sum_w += ak.sum(events.normalization_weight)

    print(dataset, "sum_w:", sum_w)
    print(dataset, "corresponds to lumi:", sum_w * n_tot / (n_selected * xsec))
