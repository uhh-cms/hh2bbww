"""
An example how to run tasks via python.
"""

from columnflow.tasks.selection import SelectEvents

import awkward as ak
# awkward not in conda environment -> source sandbox before calling this file:
# source $HBW_BASE/modules/columnflow/sandboxes/venv_columnar_dev.sh

task = SelectEvents(version="v1", dataset="tt_sl_powheg", walltime="30sec")
# task = SelectEvents(version="v1", dataset="ggHH_kl_1_kt_1_sl_hbbhww_powheg", selector="gen_hbw", walltime="30sec")
task.law_run()

outp = task.output()["collection"]

print(type(outp))
print(outp)

# check outputs of task and print fields of all parquet files
for branch in outp.keys():
    for key, item in outp[branch].items():
        print("=============== Output:", key)
        if "parquet" in item.fn:
            events = ak.from_parquet(item.fn)
            print("========== Fields:", events.fields)
            for f in events.fields:
                print("=====", f"{f}:", events[f].type)


# Debugger
from IPython import embed; embed()
