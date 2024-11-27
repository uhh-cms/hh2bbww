# coding: utf-8

"""
Collection of helper functions for creating inference models
"""

import re

import law
import order as od

from columnflow.inference import InferenceModel, ParameterType, ParameterTransformation
from columnflow.util import maybe_import, DotDict
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

    # default ml model, used for resolving defaults
    ml_model_name: str = "dense_default"

    # list of all processes/channels/systematics to include in the datacards
    processes: list = []
    config_categories: list = []
    systematics: list = []

    # customization of channels
    mc_stats: bool = True
    skip_data: bool = True

    # dummy variations when some datasets are missing
    dummy_ggf_variation: bool = False
    dummy_vbf_variation: bool = False

    version = 1

    #
    # helper functions and properties
    #

    def inf_proc(self, proc):
        """
        Helper function that translates our process names to the inference model process names
        """
        if proc in const.inference_procnames:
            return const.inference_procnames[proc]
        elif proc.startswith("hh_ggf_"):
            pattern = r"hh_ggf_([a-zA-Z\d]+)_([a-zA-Z\d]+)_kl([mp\d]+)_kt([mp\d]+)"
            replacement = r"ggHH_kl_\3_kt_\4_\1\2"
            return re.sub(pattern, replacement, proc)
        elif proc.startswith("hh_vbf_"):
            pattern = r"hh_vbf_([a-zA-Z\d]+)_([a-zA-Z\d]+)_kv([mp\d]+)_k2v([mp\d]+)_kl([mp\d]+)"
            replacement = r"qqHH_CV_\3_C2V_\4_kl_\5_\1\2"
            return re.sub(pattern, replacement, proc)
        else:
            return proc

    # inf_proc = lambda self, proc: const.inference_procnames.get(proc, proc)

    @property
    def inf_processes(self):
        return [self.inf_proc(proc) for proc in self.processes]

    def cat_name(self: InferenceModel, config_cat_inst: od.Category):
        """ Function to determine inference category name from config category """
        # NOTE: the name of the inference category cannot start with a Number
        # -> use config category with single letter added at the start?
        return f"cat_{config_cat_inst.name}"

    def config_variable(self: InferenceModel, config_cat_inst: od.Config):
        """ Function to determine inference variable name from config category """
        root_cats = config_cat_inst.x.root_cats
        if dnn_cat := root_cats.get("dnn"):
            dnn_proc = dnn_cat.replace("ml_", "")
            return f"mlscore.{dnn_proc}"
        else:
            return "mli_mbb"

    def customize_category(self: InferenceModel, cat_inst: DotDict, config_cat_inst: od.Config):
        """ Function to allow customizing the inference category """
        # root_cats = config_cat_inst.x.root_cats
        # variables = ["jet1_pt"]
        # if dnn_cat := root_cats.get("dnn"):
        #     dnn_proc = dnn_cat.replace("ml_", "")
        #     variables.append(f"mlscore.{dnn_proc}")
        # cat_inst.variables_to_plot = variables
        return

    def __init__(self, config_inst: od.Config, *args, **kwargs):
        super().__init__(config_inst)
        year = config_inst.campaign.x.year
        self.systematics = [syst.format(year=year) for syst in self.systematics]

        self.add_inference_categories()
        self.add_inference_processes()
        self.add_inference_parameters()
        # self.print_model()

        #
        # post-processing
        #

        self.cleanup()

    def print_model(self):
        """ Helper to print categories, processes and parameters of the InferenceModel """
        for cat in self.categories:
            print(f"{'=' * 20} {cat.name}")
            print(f"Variable {cat.config_variable} \nCategory {cat.config_category}")
            print(f"Processes {[p.name for p in cat.processes]}")
            print(f"Parameters {set().union(*[[param.name for param in proc.parameters] for proc in cat.processes])}")

    def add_inference_categories(self: InferenceModel):
        """
        This function creates categories for the inference model
        """
        # get the MLModel inst
        # ml_model_inst = MLModel.get_cls(self.ml_model_name)(self.config_inst)
        for config_category in self.config_categories:
            cat_inst = self.config_inst.get_category(config_category)
            # root_cats = cat_inst.x.root_cats

            cat_name = self.cat_name(cat_inst)
            cat_kwargs = dict(
                config_category=config_category,
                config_variable=self.config_variable(cat_inst),
                mc_stats=self.mc_stats,
                flow_strategy="move",
                empty_bin_value=0.0,  # NOTE: remove this when removing custom rebin task
            )
            if self.skip_data:
                cat_kwargs["data_from_processes"] = self.inf_processes
            else:
                cat_kwargs["config_data_datasets"] = [
                    dataset_inst.name for dataset_inst in
                    get_datasets_from_process(self.config_inst, "data", strategy="all")
                ]

            # add the category to the inference model
            self.add_category(cat_name, **cat_kwargs)
            # do some customization of the inference category
            self.customize_category(self.get_category(cat_name), cat_inst)

    def add_dummy_variation(self, proc, datasets):
        if self.dummy_ggf_variation and "_kl1_kt1" in proc:
            for missing_kl_variations in ("_kl0_kt1", "_kl2p45_kt1", "_kl5_kt1"):
                missing_proc = proc.replace("_kl1_kt1", missing_kl_variations)
                if missing_proc not in self.processes:
                    self.add_process(
                        self.inf_proc(missing_proc),
                        config_process=self.inf_proc(proc),
                        is_signal=("hh_" in proc.lower()),
                        config_mc_datasets=datasets,
                    )

        if self.dummy_vbf_variation and "_kv1_k2v1_kl1" in proc:
            for missing_vbf_variations in (
                "_kv1_k2v1_kl0", "_kv1_k2v1_kl2", "_kv1_k2v2_kl1", "_kv1_k2v0_kl1",
                "_kv1p5_k2v1_kl1", "_kv0p5_k2v1_kl1",
            ):
                missing_proc = proc.replace("_kv1_k2v1_kl1", missing_vbf_variations)
                if missing_proc not in self.processes:
                    self.add_process(
                        self.inf_proc(missing_proc),
                        config_process=proc,
                        is_signal=("hh_" in proc.lower()),
                        config_mc_datasets=datasets,
                    )

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
                get_datasets_from_process(self.config_inst, proc, strategy="all", check_deep=True, only_first=False)
            ]

            if not datasets:
                raise Exception(f"No datasets found for process {proc}")
            logger.debug(f"Process {proc} was assigned datasets: {datasets}")

            # check that no dataset is used multiple times
            if datasets_already_used := used_datasets.intersection(datasets):
                logger.warning(f"{datasets_already_used} datasets are used for multiple processes")
            used_datasets |= set(datasets)

            self.add_process(
                self.inf_proc(proc),
                config_process=proc,
                is_signal=("hh_" in proc.lower()),
                config_mc_datasets=datasets,
            )

            # add dummy variations if requested
            self.add_dummy_variation(proc, datasets)

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
                    process=self.inf_proc(proc),
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
                    process=self.inf_proc(proc),
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
        year = self.config_inst.campaign.x.year
        for shape_uncertainty, shape_processes in const.processes_per_shape.items():
            shape_uncertainty_formatted = shape_uncertainty.format(year=year)
            if shape_uncertainty_formatted not in self.systematics:
                continue

            # If "all" is included, takes all processes except for the ones specified (starting with !)
            if "all" in shape_processes:
                _remove_processes = {proc[:1] for proc in shape_processes if proc.startswith("!")}
                shape_processes = set(self.processes) - _remove_processes

            self.add_parameter(
                shape_uncertainty_formatted,
                process=shape_processes,
                type=ParameterType.shape,
                config_shift_source=const.source_per_shape[shape_uncertainty].format(year=year),
            )

            is_theory = "pdf" in shape_uncertainty or "murf" in shape_uncertainty
            if is_theory:
                self.add_parameter_to_group(shape_uncertainty, "theory")
            else:
                self.add_parameter_to_group(shape_uncertainty, "experiment")
