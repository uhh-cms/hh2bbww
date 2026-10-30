#!/bin/bash

# List of datasets to check
KEYS_LIST=(
    # HH GGF datasets
    # "/GluGlutoHH*/*24*v15*/NANOAODSIM" 
    # # # HH VBF datasets
    # "/VBFHHto2B2V*TuneCP5_13p6TeV*/*24*v15*/NANOAODSIM"
    # # ST (t channel) 4 FS datasets --> check nochmal 
    # # ttZ -> nunu dataset
    # # ttZ -> qq dataset
    # # tHq 4f 
    # # tHW 4f 
    # # ttZH 
    # # ZH
    # # WH?
    # # ttbb
    # "/TTBB*13p6TeV*/*24*v15*/NANOAODSIM"
    # "ST*E*F*A*N*/*24*v15*/NANOAODSIM"
    "/TBbarQ*4FS_TuneCP5_*/RunIII2024Summer24NanoAODv15-150X*/NANOAODSIM"
    "/TbarBQ*4FS_TuneCP5_*/RunIII2024Summer24NanoAODv15-150X*/NANOAODSIM"
    "/THQ_4F*/*24*v15*/NANOAODSIM"
    "/THW-5FS-ctcvcp*/*24*v15*/NANOAODSIM"
    "/TTZ-*/*24*v15*/NANOAODSIM"
    "/TTZH-*/*24*v15*/NANOAODSIM"
    "/ZH_*Hto2Wto2L2Nu*/*24*v15*/NANOAODSIM"
    # "/TTTT*/*24*v15*/NANOAODSIM"
    # check with xandras liste
    # HHH 
    # "/HHHto4B2W*_TuneCP5_13p6TeV*/*/NANOAODSIM"
    # "/TTHH*13p6TeV*/Run3*v1*/NANOAODSIM"
    # "/WHH*4B*1_0*1_0*_0_0*/Run3*v12*Run3*/NANOAODSIM"
    # "/ZHH*4B*1_0*1_0*_0_0*/Run3*v12*Run3*/NANOAODSIM"
    # "/*Z*Z*H*/*24*v15*/NANOAODSIM"
    # "/DYto2*MLL-10to50*amcatnlo*/*24*v15*/NANOAODSIM"
)

# Setup environmet by calling DB_SETUP
source /afs/desy.de/user/m/markusla/.zshrc
DBSETUP

# go into the right folder
cd $CF_REPO_BASE/hbw/scripts/cmsdb || exit 1

# Loop over each dataset key and check if it exists
for KEY in "${KEYS_LIST[@]}"; do
    echo "Checking dataset for key: $KEY"
    echo "---------------------------------------------------------------------------------------------------------"
    python3 get_das_info_nano_leg_v15.py -c smart -d "$KEY"
    echo "---------------------------------------------------------------------------------------------------------"
done
