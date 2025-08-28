# coding: utf-8
# flake8: noqa: Q003

"""
Tasks related to the creation of datacards for inference purposes.
"""

# output_collection_

from __future__ import annotations

import law
import order as od

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.inference import SerializeInferenceModelBase
from columnflow.tasks.histograms import MergeHistograms, MergeShiftedHistograms
from hbw.tasks.base import HBWTask
from columnflow.util import DotDict, maybe_import
from columnflow.tasks.framework.remote import RemoteWorkflow

hist = maybe_import("hist")


class CreateMultipleDatacards(
    HBWTask,
    # ShiftTask,
    SerializeInferenceModelBase,
    law.LocalWorkflow,
    RemoteWorkflow,
):

    resolution_task_cls = MergeHistograms
    output_collection_cls = law.NestedSiblingFileCollection

    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeHistograms=MergeHistograms,
        MergeShiftedHistograms=MergeShiftedHistograms,
    )

    @classmethod
    def get_mc_datasets(cls, config_inst: od.Config, proc_obj: DotDict) -> list[str]:
        """
        Helper to find mc datasets.

        :param config_inst: The config instance.
        :param proc_obj: process object from an InferenceModel
        :return: List of dataset names corresponding to the process *proc_obj*.
        """
        # the config instance should be specified in the config data of the proc_obj
        if not (config_data := proc_obj.config_data.get(config_inst.name)):
            return []

        # when datasets are defined on the process object itself, interpret them as patterns
        if config_data.mc_datasets:
            return [
                dataset.name
                for dataset in config_inst.datasets
                if (
                    dataset.is_mc and
                    law.util.multi_match(dataset.name, config_data.mc_datasets, mode=any)
                )
            ]

        # if the proc object is dynamic, it is calculated and the fly (e.g. via a hist hook)
        # and doesn't have any additional requirements
        if proc_obj.is_dynamic:
            return []

        # otherwise, check the config
        # return [
        #     dataset_inst.name
        #     for dataset_inst in get_datasets_from_process(config_inst, config_data.process)
        # ]

    @classmethod
    def get_data_datasets(cls, config_inst: od.Config, cat_obj: DotDict) -> list[str]:
        """
        Helper to find data datasets.

        :param config_inst: The config instance.
        :param cat_obj: category object from an InferenceModel
        :return: List of dataset names corresponding to the category *cat_obj*.
        """
        # the config instance should be specified in the config data of the proc_obj
        if not (config_data := cat_obj.config_data.get(config_inst.name)):
            return []

        if not config_data.data_datasets:
            return []

        return [
            dataset.name
            for dataset in config_inst.datasets
            if (
                dataset.is_data and
                law.util.multi_match(dataset.name, config_data.data_datasets, mode=any)
            )
        ]

    def create_branch_map(self):
        return list(self.inference_model_inst.categories)

    def _requires_cat_obj(self, cat_obj: DotDict, merge_variables: bool = False, **req_kwargs):
        """
        Helper to create the requirements for a single category object.

        :param cat_obj: category object from an InferenceModel
        :param merge_variables: whether to merge the variables from all requested category objects
        :return: requirements for the category object
        """
        reqs = {}
        for config_inst in self.config_insts:
            if not (config_data := cat_obj.config_data.get(config_inst.name)):
                continue

            if merge_variables:
                for _cat_obj in self.branch_map.values():
                    s = _cat_obj.config_data.get(config_inst.name).variable
                    parsed = s.strip("[]").replace("'", "").split(", ")
                    variables = tuple(parsed)
            else:
                s = config_data.variable
                parsed = s.strip("[]").replace("'", "").split(", ")
                variables = tuple(parsed)
                # variables = (config_data.variable,)

            # add merged shifted histograms for mc
            reqs[config_inst.name] = {
                proc_obj.name: {
                    dataset: self.reqs.MergeShiftedHistograms.req_different_branching(
                        self,
                        config=config_inst.name,
                        dataset=dataset,
                        shift_sources=tuple(
                            param_obj.config_data[config_inst.name].shift_source
                            for param_obj in proc_obj.parameters
                            if (
                                config_inst.name in param_obj.config_data and
                                self.inference_model_inst.require_shapes_for_parameter(param_obj)
                            )
                        ),
                        variables=variables,
                        **req_kwargs,
                    )
                    for dataset in self.get_mc_datasets(config_inst, proc_obj)
                }
                for proc_obj in cat_obj.processes
                if config_inst.name in proc_obj.config_data and not proc_obj.is_dynamic
            }
            # add merged histograms for data, but only if
            # - data in that category is not faked from mc, or
            # - at least one process object is dynamic (that usually means data-driven)
            if (
                (not cat_obj.data_from_processes or any(proc_obj.is_dynamic for proc_obj in cat_obj.processes)) and
                (data_datasets := self.get_data_datasets(config_inst, cat_obj))
            ):
                reqs[config_inst.name]["data"] = {
                    dataset: self.reqs.MergeHistograms.req_different_branching(
                        self,
                        config=config_inst.name,
                        dataset=dataset,
                        variables=variables,
                        **req_kwargs,
                    )
                    for dataset in data_datasets
                }

        return reqs

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["merged_hists"] = hist_reqs = {}
        for cat_obj in self.branch_map.values():
            cat_reqs = self._requires_cat_obj(cat_obj, merge_variables=True)
            for config_name, proc_reqs in cat_reqs.items():
                hist_reqs.setdefault(config_name, {})
                for proc_name, dataset_reqs in proc_reqs.items():
                    hist_reqs[config_name].setdefault(proc_name, {})
                    for dataset_name, task in dataset_reqs.items():
                        hist_reqs[config_name][proc_name].setdefault(dataset_name, set()).add(task)
        return reqs

    def requires(self):
        cat_obj = self.branch_data
        return self._requires_cat_obj(cat_obj, branch=-1, workflow="local")

    @property
    def variables(self):
        cat_obj = self.branch_data
        config_data = cat_obj.config_data.get(self.config_insts[0].name)
        s = config_data.variable
        parsed = s.strip("[]").replace("'", "").split(", ")
        return parsed

    @property
    def workflow_variables(self):
        for _cat_obj in self.branch_map.values():
            s = _cat_obj.config_data.get(self.config_inst.name).variable
            parsed = s.strip("[]").replace("'", "").split(", ")
            # variables = tuple(parsed)
        return parsed

    def basename(self, name: str, ext: str) -> str:
        cat_obj = self.branch_data
        parts = [name, cat_obj.name]
        if self.hist_hooks_repr:
            parts.append(f"hooks_{self.hist_hooks_repr}")
        if cat_obj.postfix is not None:
            parts.append(cat_obj.postfix)
        return f"{'__'.join(map(str, parts))}.{ext}"

    def output(self):

        outputs = {}
        for var in self.variables:
            self._active_var = var

            card_str = f"{var}/{self.basename('datacard', 'txt')}"
            shape_str = f"{var}/{self.basename('shapes', 'root')}"

            outputs[var] = {
                "card": self.target(card_str, type="f"),
                "shapes": self.target(shape_str, type="f"),
            }
        outputs["run_inference_task"] = self.target("inference_task_call", type="f")
        outputs["inference_task_script"] = self.target("inference_task_script", type="f")
        return outputs

    def load_process_hists(
        self,
        inputs: dict,
        cat_obj: DotDict,
        config_inst: od.Config,
        variable,
    ) -> dict[od.Process, hist.Hist]:
        # loop over all configs required by the datacard category and gather histograms
        config_data = cat_obj.config_data.get(config_inst.name)

        # collect histograms per config process
        hists: dict[od.Process, hist.Hist] = {}
        with self.publish_step(
            f"extracting {variable} in {config_data.category} for config {config_inst.name}...",
        ):
            for proc_obj_name, inp in inputs[config_inst.name].items():
                if proc_obj_name == "data":
                    process_inst = config_inst.get_process("data")
                else:
                    proc_obj = self.inference_model_inst.get_process(proc_obj_name, category=cat_obj.name)
                    process_inst = config_inst.get_process(proc_obj.config_data[config_inst.name].process)
                sub_process_insts = [sub for sub, _, _ in process_inst.walk_processes(include_self=True)]

                # loop over per-dataset inputs and extract histograms containing the process
                h_proc = None
                for dataset_name, _inp in inp.items():
                    dataset_inst = config_inst.get_dataset(dataset_name)

                    # skip when the dataset is already known to not contain any sub process
                    if not any(map(dataset_inst.has_process, sub_process_insts)):
                        self.logger.warning(
                            f"dataset '{dataset_name}' does not contain process '{process_inst.name}' or any of "
                            "its subprocesses which indicates a misconfiguration in the inference model "
                            f"'{self.inference_model}'",
                        )
                        continue

                    # open the histogram and work on a copy
                    h = _inp["collection"][0]["hists"][variable].load(formatter="pickle").copy()

                    # axis selections
                    h = h[{
                        "process": [
                            hist.loc(p.name)
                            for p in sub_process_insts
                            if p.name in h.axes["process"]
                        ],
                    }]

                    # axis reductions
                    h = h[{"process": sum}]

                    # add the histogram for this dataset
                    if h_proc is None:
                        h_proc = h
                    else:
                        h_proc += h

                # there must be a histogram
                if h_proc is None:
                    raise Exception(f"no histograms found for process '{process_inst.name}'")

                # save histograms mapped to processes
                hists[process_inst] = h_proc

        return hists

    def full_cmd(self, var):
        # version_str = self.inference_model
        export_path = self.output()[var]["card"].path
        export_path = export_path.rsplit("/", 1)[0]
        apply_card = "--apply-fit-datacards "
        config_cats = self.inference_model_inst.config_categories
        for cat in config_cats:
            apply_card = apply_card + f"cat_{cat}=$APPLY_FIT_CARDS/datacard__{cat}.txt,"
        cmd = f"{'-' * 20}" + f" Apply fit on variable: {var} " + f"{'-' * 50}" + "\n\n"
        cmd = cmd + f"export APPLY_FIT_CARDS={export_path}" + "\n\n\n\n"
        cmd = cmd + apply_card + "\n\n"
        cmd = cmd + "Possible custom arguments that might be important for those fits: "
        cmd = cmd + "--custom-args ""--samples <NUMBER_OF_TOY_SAMPLES> --skip-prefit"" --unblinded" + "\n\n"

        return cmd

    def get_workflow_script(self, var_lst):
        # Get export base path (two levels up from card.path)
        export_path = self.output()[var_lst[0]]["card"].path
        export_path = export_path.rsplit("/", 1)[0]
        export_path = export_path.rsplit("/", 1)[0]

        version_str = f"{self.inference_model}"

        # Start of shell script
        default_str = "export CARDS_PATH=\"\"\n\ndefault_fit_cards=\"\"\n\nno_of_toy_samples=\"\"\n\n"
        script = "#!/bin/bash\n\n" + default_str

        # Array of variables
        script += "vars=(\n"
        for var in var_lst:
            script += f"    {var}\n"
        script += ")\n\n"
        script += "OUTPUT_LOG=\"part1_outputs.txt\"\n"
        script += "> \"$OUTPUT_LOG\"\n"

        # Loop for Part 1
        script += "for var in \"${vars[@]}\"; do\n"
        script += "    (\n"
        script += f"    export APPLY_FIT_CARDS=\"{export_path}/${{var}}\"\n"
        script += "    law run PreAndPostFitShapes \\\n"
        script += f"        --version \"{version_str}__${{var}}\" \\\n"
        script += "        --datacards \"$default_fit_cards\" \\\n"
        script += "        --apply-fit-datacards "

        # Build category string
        config_cats = self.inference_model_inst.config_categories
        apply_card = ",".join(
            [f"cat_{cat}=$APPLY_FIT_CARDS/datacard__{cat}.txt" for cat in config_cats],
        )
        script += apply_card + " \\\n"

        # Custom args
        script += "        --custom-args \"--samples $no_of_toy_samples --skip-prefit\" \\\n"
        script += "        --unblinded\n"
        script += "    ) >> \"$OUTPUT_LOG\" 2>&1 & \n"
        script += "done\n"
        script += "wait\n"

        ############################
        # PART 2: Auto-generate CF script
        ############################
        script += "\n# Generate Part 2 script automatically\n"
        script += "OUTPUT_SCRIPT=\"PreAndPostfit_in_CF_script.sh\"\n> \"$OUTPUT_SCRIPT\"\n\n"

        # Grep ROOT_FILE_PATH base (strip last folder)
        script += "output_root_file=$(grep -oP '(?<=export ROOT_FILE_PATH=)[^\\\"]+' \"$OUTPUT_LOG\" | "
        script += "head -n 1 | xargs dirname)\n"
        # Grep inference model value
        script += "inference_model=$(grep -oP '(?<=--inference-model )\\S+' \"$OUTPUT_LOG\" | head -n 1)\n\n"

        # Write script header
        script += "{\n"
        script += "echo 'vars=('\n"
        script += "for v in \"${vars[@]}\"; do\n"
        script += "    echo \"    $v\"\n"
        script += "done\n"
        script += "echo ')'\n"
        script += "echo ''\n"
        script += "echo 'output_root_file=\"'\"$output_root_file\"'\"'\n"
        script += "echo 'default_processes=\"ddl4\"'\n"
        script += "echo 'inference_model=\"'\"$inference_model\"'\"'\n"
        script += "echo ''\n"
        script += "echo 'for var in \"${vars[@]}\"; do'\n"
        script += "echo '    ('\n"
        script += "echo '    export ROOT_FILE_PATH=\"$output_root_file/$inference_model\"\"__${var}\"'\n"
        script += "echo '    law run hbw.PlotPostfitShapes \\'\n"
        script += "echo '        --processes $default_processes \\'\n"
        script += "echo '        --version ${var} \\'\n"
        script += "echo '        --inference-model \"$inference_model\" \\'\n"
        script += "echo '        --fit-diagnostics-file \"$ROOT_FILE_PATH/hh_root_shapes.root\"'\n"
        script += "echo '    ) &'\n"
        script += "echo 'done'\n"
        script += "echo ''\n"
        script += "} >> \"$OUTPUT_SCRIPT\"\n\n"

        # Make script executable
        script += "chmod +x \"$OUTPUT_SCRIPT\"\n"
        script += "echo \"Template Part 2 script generated: $OUTPUT_SCRIPT\"\n"

        return script

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):

        import hist
        from columnflow.inference.cms.datacard import DatacardHists, ShiftHists, DatacardWriter

        # prepare inputs
        inputs = self.input()
        # loop over all configs required by the datacard category and gather histograms
        cat_obj = self.branch_data
        datacard_hists: DatacardHists = {cat_obj.name: {}}

        def get_single_variable_datacards(reduce_axis: str, outputs):
            # step 1: gather histograms per process for each config
            input_hists: dict[od.Config, dict[od.Process, hist.Hist]] = {}
            for config_inst in self.config_insts:
                # skip configs that are not required
                if not cat_obj.config_data.get(config_inst.name):
                    continue
                # load them
                input_hists[config_inst] = self.load_process_hists(inputs, cat_obj, config_inst, reduce_axis)

            # step 2: apply hist hooks
            input_hists = self.invoke_hist_hooks(input_hists)

            # step 3: transform to nested histogram as expected by the datacard writer
            for config_inst in input_hists.keys():
                config_data = cat_obj.config_data.get(config_inst.name)

                # determine leaf categories to gather
                category_inst = config_inst.get_category(config_data.category)
                leaf_category_insts = category_inst.get_leaf_categories() or [category_inst]

                # start the transformation
                proc_objs = list(cat_obj.processes)
                if config_data.data_datasets and not cat_obj.data_from_processes:
                    proc_objs.append(self.inference_model_inst.process_spec(name="data"))
                for proc_obj in proc_objs:
                    # get the corresponding process instance
                    if proc_obj.name == "data":
                        process_inst = config_inst.get_process("data")
                    elif config_inst.name in proc_obj.config_data:
                        process_inst = config_inst.get_process(proc_obj.config_data[config_inst.name].process)
                    else:
                        # skip process objects that rely on data from a different config
                        continue

                    # extract the histogram for the process
                    if not (h_proc := input_hists[config_inst].get(process_inst, None)):
                        self.logger.warning(
                            f"found no histogram to model datacard process '{proc_obj.name}', please check your "
                            f"inference model '{self.inference_model}'",
                        )
                        continue

                    # select relevant categories
                    h_proc = h_proc[{
                        "category": [
                            hist.loc(c.name)
                            for c in leaf_category_insts
                            if c.name in h_proc.axes["category"]
                        ],
                    }][{"category": sum}]

                    # TODO: maybe i can also just take this as a new task in our analysis and

                    # create the nominal hist
                    datacard_hists[cat_obj.name].setdefault(proc_obj.name, {}).setdefault(config_inst.name, {})
                    shift_hists: ShiftHists = datacard_hists[cat_obj.name][proc_obj.name][config_inst.name]
                    shift_hists["nominal"] = h_proc[{

                        "shift": hist.loc(config_inst.get_shift("nominal").name),
                    }]

                    # no additional shifts need to be created for data
                    if proc_obj.name == "data":
                        continue

                    # create histograms per shift
                    for param_obj in proc_obj.parameters:
                        # skip the parameter when varied hists are not needed
                        if not self.inference_model_inst.require_shapes_for_parameter(param_obj):
                            continue
                        # store the varied hists
                        shift_source = param_obj.config_data[config_inst.name].shift_source
                        for d in ["up", "down"]:
                            shift_hists[(param_obj.name, d)] = h_proc[{
                                "shift": hist.loc(config_inst.get_shift(f"{shift_source}_{d}").name),
                            }]

            writer = DatacardWriter(self.inference_model_inst, datacard_hists)
            with outputs["card"].localize("w") as tmp_card, outputs["shapes"].localize("w") as tmp_shapes:
                writer.write(tmp_card.abspath, tmp_shapes.abspath, shapes_path_ref=outputs["shapes"].basename)

        output = self.output()
        inference_cmd = ""
        for var in self.variables:
            var_cmd = self.full_cmd(var)
            inference_cmd = inference_cmd + var_cmd
            get_single_variable_datacards(var, outputs=output[var])

        inference_script = self.get_workflow_script(self.variables)

        if cat_obj.name == f"cat_{self.inference_model_inst.config_categories[0]}":
            print(inference_cmd)

            self.output()["run_inference_task"].dump(inference_cmd, formatter="text")

            # Write the script string there
            with open(self.output()["inference_task_script"].path, "w") as f:
                f.write(inference_script)

        elif cat_obj.name == self.inference_model_inst.config_categories[0]:
            print(inference_cmd)

            self.output()["run_inference_task"].dump(inference_cmd, formatter="text")
            with open(self.output()["inference_task_script"].path, "w") as f:
                f.write(inference_script)
