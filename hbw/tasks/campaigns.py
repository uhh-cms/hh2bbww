# coding: utf-8

"""
Custom tasks for creating and managing campaigns.
"""

from collections import defaultdict
from functools import cached_property
import importlib

import law
import luigi

from columnflow.tasks.framework.base import AnalysisTask
from hbw.tasks.base import HBWTask


logger = law.logger.get_logger(__name__)


campaign_map = {
    "c17": {
        "cmsdb.campaigns.run2_2017_nano_v9": "campaign_run2_2017_nano_v9",
    },
    "c22pre": {
        "cmsdb.campaigns.run3_2022_preEE_nano_uhh_v12": "campaign_run3_2022_preEE_nano_uhh_v12",
        "cmsdb.campaigns.run3_2022_preEE_nano_v12": "campaign_run3_2022_preEE_nano_v12",
        "cmsdb.campaigns.run3_2022_preEE_nano_v13": "campaign_run3_2022_preEE_nano_v13",
    },
    "c22post": {
        "cmsdb.campaigns.run3_2022_postEE_nano_uhh_v12": "campaign_run3_2022_postEE_nano_uhh_v12",
        "cmsdb.campaigns.run3_2022_postEE_nano_v12": "campaign_run3_2022_postEE_nano_v12",
        "cmsdb.campaigns.run3_2022_postEE_nano_v13": "campaign_run3_2022_postEE_nano_v13",
    },
    "c22post_das": {
        "cmsdb.campaigns.run3_2022_postEE_nano_v12": "campaign_run3_2022_postEE_nano_v12",
        "cmsdb.campaigns.run3_2022_postEE_nano_v13": "campaign_run3_2022_postEE_nano_v13",
        "cmsdb.campaigns.run3_2022_postEE_nano_uhh_v12": "campaign_run3_2022_postEE_nano_uhh_v12",
    },
    "c22pre_das": {
        "cmsdb.campaigns.run3_2022_preEE_nano_v12": "campaign_run3_2022_preEE_nano_v12",
        "cmsdb.campaigns.run3_2022_preEE_nano_v13": "campaign_run3_2022_preEE_nano_v13",
        "cmsdb.campaigns.run3_2022_preEE_nano_uhh_v12": "campaign_run3_2022_preEE_nano_uhh_v12",
    },
    "c23pre": {
        "cmsdb.campaigns.run3_2023_preBPix_nano_v12": "campaign_run3_2023_preBPix_nano_v12",
        "cmsdb.campaigns.run3_2023_preBPix_nano_v13": "campaign_run3_2023_preBPix_nano_v13",
    },
    "c23post": {
        "cmsdb.campaigns.run3_2023_postBPix_nano_v12": "campaign_run3_2023_postBPix_nano_v12",
        "cmsdb.campaigns.run3_2023_postBPix_nano_v13": "campaign_run3_2023_postBPix_nano_v13",
    },
    # Nano V14
    "c22preV14": {
        "cmsdb.campaigns.run3_2022_preEE_nano_uhh_v14": "campaign_run3_2022_preEE_nano_uhh_v14",
    },
    "c22postV14": {
        "cmsdb.campaigns.run3_2022_postEE_nano_uhh_v14": "campaign_run3_2022_postEE_nano_uhh_v14",
    },
    "c23preV14": {
        "cmsdb.campaigns.run3_2023_preBPix_nano_uhh_v14": "campaign_run3_2023_preBPix_nano_uhh_v14",
    },
    "c23postV14": {
        "cmsdb.campaigns.run3_2023_postBPix_nano_uhh_v14": "campaign_run3_2023_postBPix_nano_uhh_v14",
    },
}

broken_files = {
    "run3_2022_postEE_nano_uhh_v12": {
        "dy_m10to50_amcatnlo": [
            # missing LHEScaleWeights
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/0/4B7063C8-D7B7-A45F-0B56-817AECEAFB43.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/0/D4D70538-4AF1-A95C-3A57-5EB5D2FFAB08.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/0/24934037-F730-CFB5-A82E-5D6669E8C85B.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/0/EB93CCFF-F013-D816-7586-1051CA0BC3C8.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/0/08C7ABCF-F7DE-F73F-218E-12A85C1A6E89.root",  # noqa: E501
        ],
        "dy_m50toinf_amcatnlo": [
            # broken
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/3FE6B8C0-4234-4EE4-5BEA-E232539E0D85.root",  # noqa: E501
            # missing LHEScaleWeights
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/0/10B3DD52-F1B9-F8FD-E6FD-D59ECCE90963.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/0/13DA9D04-5A59-51B8-67EC-54723C6DB4F3.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/F31A7CD7-F9CF-2A51-42B6-26E82E134DE7.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/5B7AFD98-EC30-D01C-59FA-162D86E82C61.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/4E19FA69-9612-E1AF-A537-099F0119CC60.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/197F9F10-660F-AC8B-83DF-AE02CA2AEA71.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/FAFAECB9-A1C2-A07C-16F7-C7A8008A404E.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/F541A987-BD0F-09AA-156F-2836570E8886.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/B600A38B-9418-1EA3-8B4E-8969BE8ECDDE.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/3B2EEDD7-0767-6112-8C60-B522A4A1910C.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/5AC98FFE-A2A1-EAD4-BFD5-59F64E2A3465.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/442D7323-2E81-9EFA-B9C1-E3414FF2C5B4.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/198C8C10-BD66-5B2B-C70A-34EC4EEFB65C.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/C0D75D2C-1A95-A416-E2BA-3E16E3249333.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext1-v1/0/DDBA1F4E-4795-A218-E0A0-4FF036B5CB68.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/770ADB5F-4F37-50A4-1FA2-34D04AD062B8.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/81E8769E-6D9A-674A-419A-40227862E8CC.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/7CA623D4-4E9F-E689-ECFA-6F251291FAB3.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/5FDA4334-32A3-0262-C0F5-5AA2AF906F94.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/CE34DA61-BB00-E50C-76F6-591032050F6F.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/0E2C013A-B1CD-63AA-4FBC-92AB1171BDF7.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/4CC2A468-5DC4-3513-C484-CF10B96DD7E1.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/0568BFAE-B3C4-FF86-2E8E-ABFEB3F418BC.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/A95D7402-87A5-C41B-3B89-211DCE48A4BB.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/836369D5-B667-F3DD-78D7-9D075766A182.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/65B3170C-8F95-F3AA-2B8F-056AAF05905D.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/EE0F4D89-CC83-02CC-19EE-8BEA0AC9EB88.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/69DBFCA4-B503-DA49-8972-D8EFFA69DA7C.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/20988817-825D-C5DD-3AC0-5A929F768A5F.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/BD755398-6344-C786-2BBE-B648C5056544.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/E80FA718-9A41-22EF-B2C2-ABE91B334447.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/F095A2DC-3D9D-540A-D77A-E0881A062F06.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/7BAFD1A3-6EF1-18A5-AC03-9158A4D965E0.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/C71C0C70-9F3D-C581-219E-6FE00957D3CB.root",  # noqa: E501
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/0/2307BFB8-74FF-BB2A-52A2-909D5F57C221.root",  # noqa: E501
            # missing LHEScaleWeights
        ],
        "dy_m50toinf_2j_amcatnlo": [
            # broken
            "/store/mc/Run3Summer22EEMiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/0/FD427E18-2F78-5055-7B38-8929DDF4F1EA.root",  # noqa: E501
        ],
    },
    "run3_2022_preEE_nano_uhh_v12": {
        "dy_m50toinf_amcatnlo": [
            # missing LHEScaleWeights
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/0/140BABD5-F5C1-543C-7425-92CDA4A385B9.root",  # noqa: E501
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/0/F96C5BD4-8AFF-3B01-A17B-62F17F74895B.root",  # noqa: E501
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5_ext2-v2/0/39B93A78-FF63-8552-5C58-257144882E6B.root",  # noqa: E501
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5_ext2-v2/0/12615068-4201-0739-6128-21B694B3CF6E.root",  # noqa: E501
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5_ext2-v2/0/3163B05D-3FFB-1C6B-60BB-B5CD14166ACE.root",  # noqa: E501
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/0/7BE26782-9B31-D8AC-E317-EF6F32C391BF.root",  # noqa: E501
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5_ext1-v1/0/17AB951D-549D-89D1-345D-CE6CD5B5B3D0.root",  # noqa: E501
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5_ext2-v2/0/22B6C39B-3332-7E8A-B8C7-F23367A5F297.root",  # noqa: E501
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5_ext2-v2/0/BAE33AA8-086B-7D5D-26EA-C52C9C6D31FE.root",  # noqa: E501
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5_ext2-v2/0/C70989AC-334C-EB82-4ACC-B8C48FFE2433.root",  # noqa: E501
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5_ext2-v2/0/385E6DCC-4FB6-ED71-2B3A-5B23C5A3ACC2.root",  # noqa: E501

        ],
        "dy_m10to50_amcatnlo": [
            # missing LHEScaleWeights
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/0/315BBEDB-FF7D-B3FB-0355-F6DA23E297BE.root",  # noqa: E501
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/0/2E0573CC-695C-340B-5720-85278B31496E.root",  # noqa: E501
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5_ext1-v1/0/31669005-269B-419C-B93E-E3E4A607B644.root",  # noqa: E501
        ],
        "w_lnu_amcatnlo": [
            # missing LHEScaleWeights
            "/store/mc/Run3Summer22MiniAODv4_NanoAODv12UHH/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/0/33E9A5A9-73C5-42C6-D337-08D23E9144BF.root",  # noqa: E501
        ],
    },
}


class BuildCampaignSummary(
    HBWTask,
    AnalysisTask,
):

    config = luigi.Parameter()
    # TODO: set campaigns as part of this function instead of configuring in the config?

    recreate_backup_summary = luigi.BoolParameter(default=False)

    def requires(self):
        return {}

    def store_parts(self):
        parts = super().store_parts()

        # add the config name
        parts.insert_after("task_family", "config", self.config)

        return parts

    @cached_property
    def campaigns(self):
        if self.config not in campaign_map:
            raise ValueError(f"Unknown config {self.config}")
        return campaign_map[self.config]

    def modify_campaign(self, campaign_inst):
        """
        Modify the campaign instance, e.g. by adding datasets or changing dataset properties.
        """
        if campaign_inst.name not in broken_files:
            return

        for dataset_name, broken_files_list in broken_files[campaign_inst.name].items():
            dataset_inst_nominal = campaign_inst.get_dataset(dataset_name).info["nominal"]

            if len(set(broken_files_list)) != len(broken_files_list):
                raise ValueError(f"Duplicate broken files in {dataset_name}")

            dataset_inst_nominal.x.broken_files = dataset_inst_nominal.x("broken_files", []) + broken_files_list
            dataset_inst_nominal.n_files = dataset_inst_nominal.n_files - len(broken_files_list)
            # n_events not known for all broken files, but is not used anyways
            dataset_inst_nominal.n_events = -1

    @cached_property
    def campaign_insts(self):
        campaign_insts = []
        for mod, campaign in self.campaigns.items():
            campaign_inst = getattr(importlib.import_module(mod), campaign).copy()
            self.modify_campaign(campaign_inst)
            campaign_insts.append(campaign_inst)
        return campaign_insts

    def get_dataset_prio(self, dataset_name, campaign):
        """
        If dataset should be overwritten from this campaign, return True.
        Otherwise, return False.
        (not currently used, but could be used to prioritize e.g. the central tt dataset (less stats))
        """
        if "v12" in campaign.name and "uhh" not in campaign.name:
            # Take data from the central v12 campaign
            if "data" in dataset_name:
                return True

        return False

    def output(self):
        output = {
            "dataset_summary": self.target("dataset_summary.yaml"),
            "campaign_summary": self.target("campaign_summary.yaml"),
            "hbw_campaign_inst": self.target("hbw_campaign_inst.pickle"),
        }
        return output

    @cached_property
    def dataset_summary(self):
        dataset_summary = defaultdict(dict)
        used_datasets = set()
        # create campaign summary with one key per dataset (to fulfill dataset uniqueness)
        for campaign in self.campaign_insts:
            for dataset in campaign.datasets:
                if dataset.name not in used_datasets or self.get_dataset_prio(dataset.name, campaign):
                    dataset_summary[dataset.name] = {
                        "campaign": campaign.name,
                        "n_events": dataset.n_events,
                        "n_files": dataset.n_files,
                    }
                    used_datasets.add(dataset.name)

        return dict(dataset_summary)

    @cached_property
    def campaign_summary(self):
        campaign_summary = {
            campaign.name: {} for campaign in self.campaign_insts
        }

        for dataset, dataset_info in self.dataset_summary.items():
            campaign_summary[dataset_info["campaign"]][dataset] = {
                "n_events": dataset_info["n_events"],
                "n_files": dataset_info["n_files"],
            }
        return campaign_summary

    def get_custom_campaign(self):
        hbw_campaign_inst = self.campaign_insts[0].copy()
        hbw_campaign_inst.clear_datasets()
        for campaign_inst in self.campaign_insts:
            campaign_info = self.campaign_summary[campaign_inst.name]
            for dataset in campaign_info.keys():
                dataset_inst = campaign_inst.get_dataset(dataset)
                dataset_inst.x.campaign = campaign_inst.name
                hbw_campaign_inst.add_dataset(dataset_inst)

        hbw_campaign_inst.x.campaigns = list(self.campaigns)

        return hbw_campaign_inst

    from hbw.util import timeit_multiple

    @timeit_multiple
    def run(self):
        from hbw.analysis.processes import modify_cmsdb_processes
        modify_cmsdb_processes()
        output = self.output()

        # cross check if the dataset summary did change
        backup_target = self.target("backup_dataset_summary.yaml")
        if backup_target.exists():
            backup_dataset_summary = backup_target.load(formatter="yaml")
            if backup_dataset_summary != self.dataset_summary:
                from hbw.util import gather_dict_diff
                logger.warning(
                    "Backup dataset summary does not match the current one \n"
                    f"{gather_dict_diff(backup_dataset_summary, self.dataset_summary)}",
                )
                if self.recreate_backup_summary:
                    logger.warning("Recreating backup dataset summary")
                    backup_target.dump(self.dataset_summary, formatter="yaml")
                else:
                    logger.warning(
                        "Run the following command to recreate the backup dataset summary:\n"
                        f"law run {self.task_family} --recreate-backup-summary --config {self.config} --remove-output 0,a,y",  # noqa
                    )
        else:
            logger.warning("No backup dataset summary found, creating one now")
            backup_target.dump(self.dataset_summary, formatter="yaml")

        output["dataset_summary"].dump(self.dataset_summary, formatter="yaml")
        output["campaign_summary"].dump(self.campaign_summary, formatter="yaml")

        import sys
        orig_rec_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(orig_rec_limit, 100000))
        output["hbw_campaign_inst"].dump(self.get_custom_campaign(), formatter="pickle")
        sys.setrecursionlimit(orig_rec_limit)
