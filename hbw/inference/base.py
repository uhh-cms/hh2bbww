# coding: utf-8

"""
Collection of helper functions for creating inference models
"""

import law
import order as od

from columnflow.inference import InferenceModel, ParameterType, ParameterTransformation
from columnflow.ml import MLModel
from columnflow.util import maybe_import
from columnflow.config_util import get_datasets_from_process

import hbw.inference.constants as const  # noqa


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


class HBWInferenceModelBase(InferenceModel):
    """
    This is the base for all Inference models in our hbw analysis.
    """

    #
    # Model attributes, can be changed via `cls.derive`
    #

    # default ml model
    ml_model_name = "dense_default"

    # list of all processes/channels/systematics to include in the datacards
    processes = []
    channels = []
    systematics = []

    mc_stats = True
    skip_data = True

    def __init__(self, config_inst: od.Config, *args, **kwargs):
        super().__init__(config_inst)

        self.add_inference_categories()
        self.add_inference_processes()
        self.add_inference_parameters()

        #
        # post-processing
        #

        self.cleanup()

    def add_inference_categories(self: InferenceModel):
        """
        This function creates categories for the inference model
        """

        # get processes used in MLTraining
        ml_model_inst = MLModel.get_cls(self.ml_model_name)(self.config_inst)
        ml_model_processes = ml_model_inst.processes

        lepton_channels = self.config_inst.x.lepton_channels

        for proc in ml_model_processes:
            for lep in lepton_channels:
                cat_name = f"cat_{lep}_{proc}"
                if cat_name not in self.channels:
                    continue

                cat_kwargs = {
                    "config_category": f"{lep}__ml_{proc}",
                    # "config_variable": f"mlscore.{proc}_rebin",
                    "config_variable": f"mlscore.{proc}_manybins",
                    "mc_stats": self.mc_stats,
                }
                if self.skip_data:
                    cat_kwargs["data_from_processes"] = self.processes
                else:
                    cat_kwargs["config_data_datasets"] = const.data_datasets[lep]

                self.add_category(cat_name, **cat_kwargs)

                # get the inference category to do some customization
                cat = self.get_category(cat_name)
                # variables that are plotting via hbw.InferencePlots for this category
                cat.plot_variables = [f"mlscore.{proc}", "jet1_pt"]

    def add_inference_processes(self: InferenceModel):
        """
        Function that adds processes with their corresponding datasets to all categories
        of the inference model
        """
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
                const.inference_procnames.get(proc, proc),
                config_process=proc,
                is_signal=("HH_" in proc),
                config_mc_datasets=datasets,
            )

    def add_inference_parameters(self: InferenceModel):
        """
        Function that adds all parameters (systematic variations) to the inference model
        """
        # define the two relevant types of parameter groups
        self.add_parameter_group("experiment")
        self.add_parameter_group("theory")

        # add rate + shape parameters to the inference model
        self.add_rate_parameters()
        self.add_shape_parameters()

        # TODO: check that all requested systematics are in final MLModel?

    def add_rate_parameters(self: InferenceModel):
        """
        Function that adds all rate parameters to the inference model
        """
        ecm = self.config_inst.campaign.ecm

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
                    process=const.inference_procnames.get(proc, proc),
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
                    process=const.inference_procnames.get(proc, proc),
                    type=ParameterType.rate_gauss,
                    effect=tuple(map(
                        lambda f: round(f, 3),
                        process_inst.xsecs[ecm].get(names=("pdf"), direction=("down", "up"), factor=True),
                    )),
                )
            self.add_parameter_to_group(syst_name, "theory")

    def add_shape_parameters(self: InferenceModel):
        """
        Function that adds all rate parameters to the inference model
        """
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
                self.add_parameter_to_group(shape_uncertainty, "theory")
            else:
                self.add_parameter_to_group(shape_uncertainty, "experiment")
