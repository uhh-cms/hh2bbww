"""
This is a helper script that automatically generates txt files combining all dataset DAS strings for a given campaign.
The script can be run with the following command:
```
python3 parse_datasets.py -cpn <campaign> -c keys

# examples
python3 parse_datasets.py -cpn Run3Summer22EEMiniAODv4 -d tt -p True -c smart
python3 parse_datasets.py -cpn Run3Summer22EEMiniAODv4 -d vvto_mini -p True -c nanogen

# testing
python3 parse_datasets.py -cpn RunIII2024Summer24NanoAODv15-150X -p True -d tt -c keys

# producing .txt files
python3 parse_datasets.py -cpn RunIII2024Summer24NanoAODv15-150X
```

Campaigns:
Run3Summer22NanoAODv12-130X
Run3Summer22EENanoAODv12-130X
Run3Summer23NanoAODv12-130X
Run3Summer23BPixNanoAODv12-130X

Run3Summer22NanoAODv13_133X
Run3Summer22EENanoAODv13_133X
Run3Summer23NanoAODv13_133X
Run3Summer23BPixNanoAODv13_133X

Run3Summer22EEMiniAODv4
Run3Summer22MiniAODv4
Run3Summer23BPixMiniAODv4
Run3Summer23MiniAODv4

RunIII2024Summer24NanoAODv15-150X

After running this script, the output files will be stored in the `datasets/<campaign>` directory.
These files should then be manually sorted and filtered and can then be used to generate the dataset definitions
for the campaign using the `build_campaign.py` script.

"""

import os
import argparse
import subprocess

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate and run dataset commands for a given campaign")
parser.add_argument("-cpn", "--campaign", required=True, help="Campaign name")
parser.add_argument("-c", "--convert", default="keys", help="Convert function")
parser.add_argument("-d", "--dataset-groups", nargs="+", required=False, help="Groups of datasets")
parser.add_argument("-p", "--print-output", required=False, type=bool, help="Print instead of write to file")
args = parser.parse_args()

print_output = args.print_output
campaign = args.campaign
dataset_groups = args.dataset_groups
convert = args.convert

output_dir = f"datasets/{campaign}"

# mapping from MC campaign to data campaign... not perfect, but whatever
data_campaign = (
    "Run2022*-22Sep2023_*" if "Run3Summer22" in campaign else
    "Run2023*-22Sep2023_v*-v1*" if "Run3Summer23" in campaign else
    "Run2024*"
)

# Define the mapping of output file names to dataset identifiersbuil
dataset_mapping = {
    # "data": [
    #     # f"/JetMET*/{data_campaign}/NANOAOD",
    #     # f"/Muon*/{data_campaign}/NANOAOD",
    #     # f"/EGamma*/{data_campaign}/NANOAOD",
    #     # f"/MuonEG/{data_campaign}/NANOAOD",
    # ],
    # "hbw_minis": [
    #     # f"/GluGlutoHHto2B2V*/{campaign}*/MINIAODSIM",
    #     # f"/GluGlutoHHto2B2W*/{campaign}*/MINIAODSIM",
    #     # f"/VBFHHto2B2V*/{campaign}*/MINIAODSIM",
    #     # f"/VBFHHto2B2W*/{campaign}*/MINIAODSIM",
    #     # f"/GluGluHto2W*M-125_TuneCP5_13p6TeV*/{campaign}*/MINIAODSIM",
    #     # f"/VBFHto2W*M-125_TuneCP5_13p6TeV*/{campaign}*/MINIAODSIM",
    #     # f"/ZH_Hto2B_Zto2L_M-125_TuneCP5_13p6TeV*/{campaign}*/MINIAODSIM",
    #     # f"/ZH_ZtoAll_Hto2Wto2L2Nu_M-125_TuneCP5_13p6TeV*/{campaign}*/MINIAODSIM",
    #     # f"/WplusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV*/{campaign}*/MINIAODSIM",
    #     # f"/WminusH_Hto2B_WtoLNu_M-125_TuneCP5_13p6TeV*/{campaign}*/MINIAODSIM",
    #     f"/TTTT*/{campaign}*/MINIAODSIM",
    #     f"/THQ*HIncl_M-125_*/{campaign}*/MINIAODSIM",
    #     f"/THW*HIncl_M-125_*/{campaign}*/MINIAODSIM",
    #     f"/TTZH*/{campaign}*/MINIAODSIM",
    #     f"/TTWH*/{campaign}*/MINIAODSIM",
    # ],
    "hh2bbvv": [
        f"/GluGlutoHHto2B2V*/{campaign}*/NANOAODSIM",
        f"/GluGlutoHHto2B2W*/{campaign}*/NANOAODSIM",
        f"/VBFHHto2B2V*/{campaign}*/NANOAODSIM",
        f"/VBFHHto2B2W*/{campaign}*/NANOAODSIM",
    ],
    # "higgs": [
    #     f"/GluGluH*M-125_*/{campaign}*/NANOAODSIM",
    #     f"/VBFH*M-125_*/{campaign}*/NANOAODSIM",
    #     f"/ZH*M-125_*/{campaign}*/NANOAODSIM",
    #     f"/ggZH*M-125_*/{campaign}*/NANOAODSIM",
    #     f"/WplusH*M-125_*/{campaign}*/NANOAODSIM",
    #     f"/WminusH*M-125_*/{campaign}*/NANOAODSIM",
    #     f"/TTH*M-125_*/{campaign}*/NANOAODSIM",
    #     f"/THQ*HIncl_M-125_*/{campaign}*/NANOAODSIM",
    #     f"/THW*HIncl_M-125_*/{campaign}*/NANOAODSIM",
    # ],
    "tt": [
        f"/TTtoLNu2Q*/{campaign}*/NANOAODSIM",
        f"/TTto2L2Nu*/{campaign}*/NANOAODSIM",
        f"/TTto4Q*/{campaign}*/NANOAODSIM",
    ],
    "st": [
        # s-channel
        f"/TBbarto*/{campaign}*/NANOAODSIM",
        f"/TbarBto*/{campaign}*/NANOAODSIM",
        # t-channel
        f"/TBbarQ*/{campaign}*/NANOAODSIM",
        f"/TbarBQ*/{campaign}*/NANOAODSIM",
        # tW
        f"/TWminustoLNu2Q_*/{campaign}*/NANOAODSIM",
        f"/TWminusto4Q_*/{campaign}*/NANOAODSIM",
        f"/TbarWplustoLNu2Q_*/{campaign}*/NANOAODSIM",
        f"/TbarWplusto2L2Nu_*/{campaign}*/NANOAODSIM",
        f"/TbarWplusto4Q_*/{campaign}*/NANOAODSIM",
    ],
    "ttv": [
        f"/TTZ-ZtoQQ*/{campaign}*/NANOAODSIM",
        f"/TTW-WtoQQ*/{campaign}*/NANOAODSIM",
        f"/TTLL*/{campaign}*/NANOAODSIM",
        f"/TTLNu*/{campaign}*/NANOAODSIM",
        f"/TTNuNu*/{campaign}*/NANOAODSIM",
    ],
    "ttvv": [
        f"/TTWZ*/{campaign}*/NANOAODSIM",
        f"/TTWW*/{campaign}*/NANOAODSIM",
        f"/TTZZ*/{campaign}*/NANOAODSIM",
    ],
    "tttt": [
        f"/TTTT*/{campaign}*/NANOAODSIM",
        # f"/TTG-1Jets_PTG-*/{campaign}*/NANOAODSIM",
        # f"/TGQB*/{campaign}*/NANOAODSIM",
    ],
    "dy": [
        f"/DYto2L*amcatnlo*/{campaign}*/NANOAODSIM",
    ],
    "w_lep": [
        # f"/WtoLNu*madgraph*/{campaign}*/NANOAODSIM"
        f"/WtoLNu*amcatnlo*/{campaign}*/NANOAODSIM",
    ],
    "vv": [
        # f"/WW*/{campaign}*/NANOAODSIM",
        # f"/ZZ*/{campaign}*/NANOAODSIM",
        # f"/WZ*/{campaign}*/NANOAODSIM"
        f"/WW_*/{campaign}*/NANOAODSIM",
        f"/ZZ_*/{campaign}*/NANOAODSIM",
        f"/WZ_*/{campaign}*/NANOAODSIM",
    ],
    "vvto": [
        f"/WWto*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/NANOAODSIM",
        f"/ZZto*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/NANOAODSIM",
        f"/WZto*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/NANOAODSIM",
    ],
    # "vvto_mini": [
    #     f"/WWto2L2Nu*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     f"/WWtoLNu2Q*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     f"/WWto4Q*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     f"/WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     f"/WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     f"/WZtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     f"/WZtoL3Nu_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     f"/WZto4Q_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     f"/ZZto4L*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     f"/ZZto2L2Nu*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     f"/ZZto2L2Q*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     f"/ZZto2Nu2Q*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     f"/ZZto4Q*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     # f"/WWto*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     # f"/ZZto*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    #     # f"/WZto*_TuneCP5_13p6TeV_powheg-pythia8/{campaign}*/MINIAODSIM",
    # ],
    "qcd": [
        f"/QCD_PT*_MuEnr*/{campaign}*/NANOAODSIM",
        f"/QCD_PT*_EMEnr*/{campaign}*/NANOAODSIM",
        f"/QCD_PT*_bcToE*/{campaign}*/NANOAODSIM",
        f"/QCD_PT*_DoubleEMEnr*/{campaign}*/NANOAODSIM",
    ],
}

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generate and run the commands
for dataset_group, dataset_list in dataset_mapping.items():
    output_file = f"{dataset_group}.txt"
    if dataset_groups and dataset_group not in dataset_groups:
        continue
    datasets = " ".join(dataset_list)
    command = f"python3 my_get_das_info.py -c {convert} -d {datasets}"
    if not print_output:
        output_path = os.path.join(output_dir, output_file)
        if os.path.exists(output_path):
            raise Exception(f"Output file already exists: {output_path}. Please remove or rename it first")
        command += f" >> {output_path}"
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)
