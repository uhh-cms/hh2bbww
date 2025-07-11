# coding: utf-8

"""
Collection of helper functions for creating inference models
"""

import re

import law
import order as od

from collections import defaultdict

from columnflow.inference import InferenceModel, ParameterType, ParameterTransformation
from columnflow.util import maybe_import, DotDict
from columnflow.config_util import get_datasets_from_process

from hbw.util import timeit_multiple
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

    # dictionary to allow skipping specfic datasets for a process
    skip_datasets: dict = {}

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

    @classmethod
    def used_datasets(cls, config_inst):
        """
        This classmethod defines, which datasets are used from the inference model
        and is used to declare for which datasets to run TAF init functions.
        """
        used_datasets = {"hh_ggf_hbb_hvv2l2nu_kl1_kt1_powheg"}
        # used_datasets = set().union(*[
        #     get_datasets_from_process(config_inst, proc, strategy="all")
        #     for proc in cls.processes
        # ])
        # logger.warning(f"used datasets {used_datasets}")
        return used_datasets

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
        # if dnn_cat := root_cats.get("dnn"):
        if root_cats.get("dnn"):
            # dnn_proc = dnn_cat.replace("ml_", "")
            # return f"mlscore.{dnn_proc}"
            return "mlscore.max_score"
        else:
            return "mli_lep_pt"

    def customize_category(self: InferenceModel, cat_inst: DotDict, config_cat_inst: od.Config):
        """ Function to allow customizing the inference category """
        # root_cats = config_cat_inst.x.root_cats
        # variables = ["jet1_pt"]
        # if dnn_cat := root_cats.get("dnn"):
        #     dnn_proc = dnn_cat.replace("ml_", "")
        #     variables.append(f"mlscore.{dnn_proc}")
        # cat_inst.variables_to_plot = variables
        return

    @timeit_multiple
    def __init__(self, config_insts: od.Config, *args, **kwargs):
        super().__init__(config_insts)

        self.format_systematics()
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

    def format_systematics(self):
        """ Function to format the systematics and prepare the processes_per_* dictionaries
        """
        years = {config_inst.campaign.x.year for config_inst in self.config_insts}

        systematics_formatted = sorted({
            syst.format(year=year, bjet_cat=bjet_cat)
            for syst in self.systematics
            for year in years
            for bjet_cat in ("1b", "2b")
        })

        available_procs = set(self.processes)

        self.processes_per_QCDScale = {
            unc_formatted: available_procs
            for unc, procs in const.processes_per_QCDScale.items()
            if (
                (unc_formatted := f"QCDScale_{unc}")
                in self.systematics and
                any(available_procs := [proc for proc in procs if proc in self.processes])
            )
        }
        self.processes_per_pdf_rate = {
            unc_formatted: available_procs
            for unc, procs in const.processes_per_pdf_rate.items()
            if (
                (unc_formatted := f"pdf_{unc}")
                in self.systematics and
                any(available_procs := [proc for proc in procs if proc in self.processes])
            )
        }

        self.processes_per_rate_unconstrained = {
            unc_formatted: available_procs
            for year in years
            for bjet_cat in ("1b", "2b")
            for unc, procs in const.processes_per_rate_unconstrained.items()
            if (
                (unc_formatted := "rate_" + unc.format(year=year, bjet_cat=bjet_cat))
                in systematics_formatted and
                any(available_procs := [proc for proc in procs if proc in ["all", *self.processes]])
            )
        }
        self.processes_per_shape = {
            unc_formatted: available_procs
            for year in years
            for bjet_cat in ("1b", "2b")
            for unc, procs in const.processes_per_shape.items()
            if (
                (unc_formatted := unc.format(year=year, bjet_cat=bjet_cat))
                in systematics_formatted and
                any(available_procs := [proc for proc in procs if proc in ["all", *self.processes]])
            )
        }
        # check that all systematics are considered and warn if not
        all_systs = set()
        for processes_per_syst in (
            self.processes_per_QCDScale,
            self.processes_per_pdf_rate,
            self.processes_per_rate_unconstrained,
            self.processes_per_shape,
        ):
            for syst, procs in processes_per_syst.items():
                if procs:
                    all_systs.add(syst)
                else:
                    logger.warning(
                        f"No processes found for systematic {syst}. "
                        "Please check your configuration.",
                    )
                    processes_per_syst.pop(syst)
        if not_considered := set(systematics_formatted) - all_systs:
            logger.warning(
                f"The following systematics were not considered in the inference model: "
                f"{', '.join(not_considered)}. Please check your configuration.",
            )

    def add_inference_categories(self: InferenceModel):
        """
        This function creates categories for the inference model
        """
        # get the MLModel inst
        # ml_model_inst = MLModel.get_cls(self.ml_model_name)(self.config_inst)
        for config_category in self.config_categories:
            # TODO: loop over configs here

            cat_insts = [config_inst.get_category(config_category) for config_inst in self.config_insts]

            var_names = {self.config_variable(cat_inst) for cat_inst in cat_insts}
            if len(var_names) > 1:
                raise ValueError(
                    f"Multiple variables found for category {config_category}: {var_names}. "
                    "Please ensure that all config categories use the same variable.",
                )
            var_name = var_names.pop()
            if not all(has_var := [config_inst.has_variable(var_name) for config_inst in self.config_insts]):
                missing_var_configs = [
                    config_inst.name for config_inst, has_var in zip(self.config_insts, has_var) if not has_var
                ]
                raise ValueError(
                    f"Variable {var_name} not found in configs {', '.join(missing_var_configs)} "
                    f"for {config_category}. Please ensure that {var_name} is part of all configs.",
                )

            cat_names = {self.cat_name(cat_inst) for cat_inst in cat_insts}
            if len(cat_names) > 1:
                raise ValueError(
                    f"Multiple category names found for category {config_category}: {cat_names}. "
                    "Please ensure that all config categories use the same name.",
                )
            cat_name = cat_names.pop()
            cat_kwargs = dict(
                config_data={
                    config_inst.name: self.category_config_spec(
                        category=config_category,
                        variable=var_name,
                        data_datasets=[
                            dataset_inst.name for dataset_inst in
                            get_datasets_from_process(config_inst, "data", strategy="all", only_first=False)
                        ],
                    )
                    for config_inst in self.config_insts
                },
                mc_stats=self.mc_stats,
                flow_strategy="move",
                empty_bin_value=0.0,  # NOTE: remove this when removing custom rebin task
            )
            # TODO: check that data datasets are requested as expected
            if self.skip_data:
                cat_kwargs["data_from_processes"] = self.inf_processes

            # add the category to the inference model
            self.add_category(cat_name, **cat_kwargs)
            # do some customization of the inference category
            self.customize_category(self.get_category(cat_name), cat_insts[0])

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
        # sanity checks
        if "dy" in self.processes and any(proc.startswith("dy_") for proc in self.processes):
            dy_procs = [proc for proc in self.processes if proc.startswith("dy")]
            raise Exception(f"Overlapping DY processes detected: {', '.join(dy_procs)}. ")
        used_datasets = defaultdict(set)
        for proc in self.processes:
            if any(missing_proc := [not config.has_process(proc) for config in self.config_insts]):
                config_missing = [
                    config_inst.name for config_inst, missing in zip(self.config_insts, missing_proc)
                    if missing
                ]
                raise Exception(f"Process {proc} is not defined in configs {', '.join(config_missing)}.")

            # get datasets corresponding to this process
            datasets = {config_inst.name: [
                d.name for d in
                get_datasets_from_process(config_inst, proc, strategy="all", check_deep=True, only_first=False)
                if d.name not in self.skip_datasets.get(proc, [])
            ] for config_inst in self.config_insts}

            for config, _datasets in datasets.items():
                if not _datasets:
                    raise Exception(
                        f"No datasets found for process {proc} in config {config}. "
                        "Please check your configuration.",
                    )
                logger.debug(f"Process {proc} in config {config} was assigned datasets: {datasets}")

                # check that no dataset is used multiple times (except for some well-defined processes where we know
                # that multiple processes share the same datasets)
                skip_check_multiple = (proc == "dy_lf") or ("hbb_hzz" in proc)
                if not skip_check_multiple and (
                    datasets_already_used := used_datasets[config].intersection(_datasets)
                ):
                    logger.warning(
                        f"{datasets_already_used} datasets are used for multiple processes in "
                        f"config {config}. This might lead to unexpected results. ",
                    )
                used_datasets[config] |= set(_datasets)

            self.add_process(
                name=self.inf_proc(proc),
                config_data={
                    config_inst.name: self.process_config_spec(
                        process=proc,
                        mc_datasets=datasets[config_inst.name],
                    )
                    for config_inst in self.config_insts
                },
                is_signal=("hh_" in proc.lower()),
                # is_dynamic=??,
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
        # lumi
        lumi_uncertainties = {
            lumi_unc: config_inst.x.luminosity.get(names=lumi_unc, direction=("down", "up"), factor=True)
            for config_inst in self.config_insts
            for lumi_unc in config_inst.x.luminosity.uncertainties
        }
        for lumi_unc_name, effect in lumi_uncertainties.items():
            if lumi_unc_name not in self.systematics:
                continue
            self.add_parameter(
                lumi_unc_name,
                type=ParameterType.rate_gauss,
                effect=effect,
                transformations=[ParameterTransformation.symmetrize],
            )

        # assuming campaign independent rate uncertainties
        # -> use the first config instance to get the campaign
        # NOTE: this might get tricky when including Run-2 (different ecm)
        config_inst = self.config_insts[0]
        ecm = config_inst.campaign.ecm

        proc_handled_by_unconstrained_rate = set()
        for syst_name, procs in self.processes_per_rate_unconstrained.items():
            # if syst_name not in self.systematics:
            #     continue

            param_kwargs = {
                "type": ParameterType.rate_unconstrained,
                "effect": ["1", "[0,2]"],
            }

            for bjet_cat in ("1b", "2b"):
                if syst_name.endswith(bjet_cat):
                    param_kwargs["category"] = f"*_{bjet_cat}_*"
                    param_kwargs["category_match_mode"] = "all"

            for proc in procs:
                if proc not in self.processes:
                    continue
                process_inst = config_inst.get_process(proc)
                self.add_parameter(
                    syst_name,
                    process=self.inf_proc(proc),
                    **param_kwargs,
                )
                proc_handled_by_unconstrained_rate.add(proc)

        # add QCD scale (rate) uncertainties to inference model
        # TODO: combine scale and mtop uncertainties for specific processes?
        # TODO: some scale/pdf uncertainties should be rounded to 3 digits, others to 4 digits
        # NOTE: it might be easier to just take the recommended uncertainty values from HH conventions at
        #       https://gitlab.cern.ch/hh/naming-conventions instead of taking the values from CMSDB
        for syst_name, procs in self.processes_per_QCDScale.items():
            # syst_name = f"QCDScale_{k}"
            # if syst_name not in self.systematics:
            #     continue

            for proc in procs:
                if proc not in self.processes:
                    continue
                elif proc in proc_handled_by_unconstrained_rate:
                    logger.warning(
                        f"Process {proc} is already handled by rate_unconstrained. Skipping "
                        f"{syst_name} for process {proc}.")
                    continue
                process_inst = config_inst.get_process(proc)
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
            # self.add_parameter_to_group(syst_name, "theory")

        # add PDF rate uncertainties to inference model
        for syst_name, procs in self.processes_per_pdf_rate.items():
            # syst_name = f"pdf_{k}"
            # if syst_name not in self.systematics:
            #     continue

            for proc in procs:
                if proc not in self.processes:
                    continue
                elif proc in proc_handled_by_unconstrained_rate:
                    logger.warning(
                        f"Process {proc} is already handled by rate_unconstrained. Skipping "
                        f"{syst_name} for process {proc}.")
                    continue
                process_inst = config_inst.get_process(proc)
                if "pdf" not in process_inst.xsecs[ecm]:
                    continue

                self.add_parameter(
                    syst_name,
                    process=self.inf_proc(proc),
                    type=ParameterType.rate_gauss,
                    effect=tuple(map(
                        lambda f: round(f, 3),
                        process_inst.xsecs[ecm].get(names=("pdf"), direction=("down", "up"), factor=True),
                    )),
                )
            # self.add_parameter_to_group(syst_name, "theory")

    def add_shape_parameters(self: InferenceModel):
        """
        Function that adds all rate parameters to the inference model
        """
        for shape_uncertainty, shape_processes in self.processes_per_shape.items():

            # If "all" is included, takes all processes except for the ones specified (starting with !)
            if "all" in shape_processes:
                _remove_processes = {proc[:1] for proc in shape_processes if proc.startswith("!")}
                shape_processes = set(self.processes) - _remove_processes

            param_kwargs = {
                "type": ParameterType.shape,
                "process": [self.inf_proc(proc) for proc in shape_processes],
            }
            shift_source = const.source_per_shape.get(shape_uncertainty, shape_uncertainty)
            for bjet_cat in ("1b", "2b"):
                if shape_uncertainty.endswith(bjet_cat):
                    param_kwargs["category"] = f"*_{bjet_cat}_*"
                    param_kwargs["category_match_mode"] = "all"
                    shift_source = shift_source.replace(f"_{bjet_cat}", "")

            param_kwargs["config_data"] = {
                config_inst.name: self.parameter_config_spec(
                    shift_source=shift_source,
                )
                for config_inst in self.config_insts
            }

            self.add_parameter(
                shape_uncertainty,
                **param_kwargs,
            )

            # is_theory = "pdf" in shape_uncertainty or "murf" in shape_uncertainty
            # if is_theory:
            #     self.add_parameter_to_group(shape_uncertainty, "theory")
            # else:
            #     self.add_parameter_to_group(shape_uncertainty, "experiment")
