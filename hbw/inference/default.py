# coding: utf-8

"""
hbw inference model.
"""

from columnflow.inference import inference_model, ParameterType, ParameterTransformation
from columnflow.config_util import get_datasets_from_process
import hbw.inference.constants as const  # noqa

#
# Defaults for all the Inference Model parameters
#

# used to set default requirements for cf.CreateDatacards based on the config
ml_model_name = "dense_default"
# default_producers = [f"ml_{ml_model_name}", "event_weights"]

# All processes to be included in the final datacard
processes = [
    "ggHH_kl_0_kt_1_dl_hbbhww",
    "ggHH_kl_1_kt_1_dl_hbbhww",
    "ggHH_kl_2p45_kt_1_dl_hbbhww",
    "ggHH_kl_5_kt_1_dl_hbbhww",
    "tt",
    # "ttv", "ttvv",
    "st_schannel", "st_tchannel", "st_twchannel",
    "dy_lep",
    "w_lnu",
    # "vv",
    # "vvv",
    # "qcd",
    # "ggZH", "tHq", "tHW", "ggH", "qqH", "ZH", "WH", "VH", "ttH", "bbH",
]

# All inference channels to be included in the final datacard
channels = [
    #"cat_2e_sg",
    "cat_2e_ggHH_kl_5_kt_1_dl_hbbhww",
    "cat_2e_t_bkg",
    # "cat_1e_st",
    "cat_2e_v_lep",
    #"cat_2mu_sg",
    "cat_2mu_ggHH_kl_5_kt_1_dl_hbbhww",
    "cat_2mu_t_bkg",
    # "cat_1mu_st",
    "cat_2mu_v_lep",
]

rate_systematics = [
    # Lumi: should automatically choose viable uncertainties based on campaign
    "lumi_13TeV_2016",
    "lumi_13TeV_2017",
    "lumi_13TeV_1718",
    "lumi_13TeV_correlated",
    # Rate QCDScale uncertainties
    "QCDScale_ttbar",
    "QCDScale_V",
    "QCDScale_VV",
    "QCDScale_VVV",
    "QCDScale_ggH",
    "QCDScale_qqH",
    "QCDScale_VH",
    "QCDScale_ttH",
    "QCDScale_bbH",
    "QCDScale_ggHH",  # should be included in inference model (THU_HH)
    "QCDScale_qqHH",
    "QCDScale_VHH",
    "QCDScale_ttHH",
    # Rate PDF uncertainties
    "pdf_gg",
    "pdf_qqbar",
    "pdf_qg",
    "pdf_Higgs_gg",
    "pdf_Higgs_qqbar",
    "pdf_Higgs_qg",  # none so far
    "pdf_Higgs_ttH",
    "pdf_Higgs_bbH",  # removed
    "pdf_Higgs_ggHH",
    "pdf_Higgs_qqHH",
    "pdf_VHH",
    "pdf_ttHH",
]

shape_systematics = [
    # Shape Scale uncertainties
    # "murf_envelope_ggHH_kl_1_kt_1_sl_hbbhww",
    "murf_envelope_tt",
    "murf_envelope_st_schannel",
    "murf_envelope_st_tchannel",
    "murf_envelope_st_twchannel",
    "murf_envelope_dy_lep",
    "murf_envelope_w_lnu",
    "murf_envelope_ttV",
    "murf_envelope_VV",
    # Shape PDF Uncertainties
    "pdf_shape_tt",
    "pdf_shape_st_schannel",
    "pdf_shape_st_tchannel",
    "pdf_shape_st_twchannel",
    "pdf_shape_dy_lep",
    "pdf_shape_w_lnu",
    "pdf_shape_ttV",
    "pdf_shape_VV",
    # Scale Factors (TODO)
    "btag_hf",
    "btag_lf",
    "btag_hfstats1_2017",
    "btag_hfstats2_2017"
    "btag_lfstats1_2017"
    "btag_lfstats2_2017"
    "btag_cferr1",
    "btag_cferr2",
    "mu_sf",
    # "mu_trig",
    "e_sf",
    # "e_trig",
    # "minbias_xs",
    # "top_pt",
]

# All systematics to be included in the final datacard
systematics = rate_systematics #+ shape_systematics


default_cls_dict = {
    "ml_model_name": ml_model_name,
    "processes": processes,
    "channels": channels,
    "systematics": systematics,
    "mc_stats": True,
    "skip_data": True,
}


@inference_model(
    **default_cls_dict,
)
def default(self):
    """
    This is the default Inference model.
    Idea: first build an inclusive Inference Model with all Channels/Processes/Systematics,
    then remove anything not listed in the attributes.
    """
    year = self.config_inst.campaign.x.year  # noqa; not used right now
    ecm = self.config_inst.campaign.ecm

    #
    # categories
    #

    # TODO: use ML model inst if possible
    ml_model_processes = [
        "ggHH_kl_5_kt_1_dl_hbbhww",
        #"sg",
        "t_bkg",
        # "st",
        "v_lep",
        # "w_lnu",
        # "dy_lep",
    ]

    # if process names need to be changed to fit some convention
    inference_procnames = {
        "foo": "bar",
        # "st": "ST",
        # "tt": "TT",
    }

    for proc in ml_model_processes:
        for lep in ("e", "mu"):
            cat_name = f"cat_2{lep}_{proc}"
            if cat_name not in self.channels:
                continue

            cat_kwargs = {
                "config_category": f"1{lep}__ml_{proc}",
                # "config_variable": f"mlscore.{proc}_rebin", what is was before merge 
                "config_variable": f"mlscore.{proc}_manybins",
                "mc_stats": self.mc_stats,
            }
            if self.skip_data:
                cat_kwargs["data_from_processes"] = self.processes
            else:
                cat_kwargs["config_data_datasets"] = [f"data_e_{i}" for i in ["b", "c", "d", "e", "f"]]

            self.add_category(cat_name, **cat_kwargs)  # noqa

    # add processes with corresponding datasets to all categories of the inference model
    used_datasets = set()
    for proc in self.processes:
        if not self.config_inst.has_process(proc):
            raise Exception(f"Process {proc} not included in the config {self.config_inst.name}")

        # get datasets corresponding to this process
        datasets = [
            d.name for d in
            get_datasets_from_process(self.config_inst, proc, strategy="inclusive")
        ]

        # check that no dataset is used multiple times
        if datasets_already_used := used_datasets.intersection(datasets):
            raise Exception(f"{datasets_already_used} datasets are used for multiple processes")
        used_datasets |= set(datasets)

        self.add_process(
            inference_procnames.get(proc, proc),
            config_process=proc,
            is_signal=("HH_" in proc),
            config_mc_datasets=datasets,
        )

    #
    # parameters
    #

    # groups
    self.add_parameter_group("experiment")
    self.add_parameter_group("theory")

    # lumi
    lumi = self.config_inst.x.luminosity
    for unc_name in lumi.uncertainties:
        if unc_name not in self.systematics:
            continue

        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
            transformations=[ParameterTransformation.symmetrize],
        )

    # add QCD scale (rate) uncertainties to inference model
    # TODO: combine scale and mtop uncertainties for specific processes?
    # TODO: some scale/pdf uncertainties should be rounded to 3 digits, others to 4 digits
    # NOTE: it might be easier to just take the recommended uncertainty values from HH conventions at
    #       https://gitlab.cern.ch/hh/naming-conventions instead of taking the values from CMSDB
    for k, procs in const.processes_per_QCDScale.items():
        syst_name = f"QCDScale_{k}"
        if syst_name not in self.systematics:
            continue

        for proc in procs:
            if proc not in self.processes:
                continue
            process_inst = self.config_inst.get_process(proc)
            if "scale" not in process_inst.xsecs[ecm]:
                continue
            self.add_parameter(
                syst_name,
                process=inference_procnames.get(proc, proc),
                type=ParameterType.rate_gauss,
                effect=tuple(map(
                    lambda f: round(f, 3),
                    process_inst.xsecs[ecm].get(names=("scale"), direction=("down", "up"), factor=True),
                )),
            )
        self.add_parameter_to_group(syst_name, "theory")

    # add PDF rate uncertainties to inference model
    for k, procs in const.processes_per_pdf_rate.items():
        syst_name = f"pdf_{k}"
        if syst_name not in self.systematics:
            continue

        for proc in procs:
            if proc not in self.processes:
                continue
            process_inst = self.config_inst.get_process(proc)
            if "pdf" not in process_inst.xsecs[ecm]:
                continue

            self.add_parameter(
                f"pdf_{k}",
                process=inference_procnames.get(proc, proc),
                type=ParameterType.rate_gauss,
                effect=tuple(map(
                    lambda f: round(f, 3),
                    process_inst.xsecs[ecm].get(names=("pdf"), direction=("down", "up"), factor=True),
                )),
            )
        self.add_parameter_to_group(syst_name, "theory")

    for shape_uncertainty, shape_processes in const.processes_per_shape.items():
        if shape_uncertainty not in self.systematics:
            continue

        # If "all" is included, takes all processes except for the ones specified (starting with !)
        if "all" in shape_processes:
            _remove_processes = {proc[:1] for proc in shape_processes if proc.startswith("!")}
            shape_processes = set(self.processes) - _remove_processes

        self.add_parameter(
            shape_uncertainty,
            process=shape_processes,
            type=ParameterType.shape,
            config_shift_source=const.source_per_shape[shape_uncertainty],
        )

        is_theory = "pdf" in shape_uncertainty or "murf" in shape_uncertainty
        if is_theory:
            self.add_parameter_to_group(syst_name, "theory")
        else:
            self.add_parameter_to_group(syst_name, "experiment")

    #
    # post-processing
    #

    self.cleanup()


#
# derive some additional Inference Models
#

cls_dict = default_cls_dict.copy()

cls_dict["systematics"] = rate_systematics

# inference model with only rate uncertainties
rates_only = default.derive("rates_only", cls_dict=cls_dict)

cls_dict["processes"] = [
    "ggHH_kl_0_kt_1_dl_hbbhww",
    "ggHH_kl_1_kt_1_dl_hbbhww",
    "ggHH_kl_2p45_kt_1_dl_hbbhww",
    "ggHH_kl_5_kt_1_dl_hbbhww",
    "v_lep",
]

cls_dict["channels"] = [
    "cat_2e_ggHH_kl_1_kt_1_dl_hbbhww",
    "cat_2mu_ggHH_kl_1_kt_1_dl_hbbhww",
    "cat_2e_v_lep",
]

cls_dict["systematics"] = [
    "lumi_13TeV_2017",
]

# minimal model for quick test purposes
test = default.derive("test", cls_dict=cls_dict)
