"""
Tasks for creating correctionlib files.
"""

import law
import luigi
import order as od

from functools import cached_property
from collections import OrderedDict, defaultdict

from columnflow.tasks.framework.base import Requirements, ShiftTask
from columnflow.tasks.framework.mixins import (
    SelectorClassMixin, CalibratorClassesMixin,
    DatasetsProcessesMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.selection import MergeSelectionStats
from columnflow.util import maybe_import, dev_sandbox, DotDict

# For NJet COrrections
from columnflow.tasks.framework.histograms import (
    HistogramsUserSingleShiftBase,
)
from columnflow.tasks.framework.parameters import MultiSettingsParameter
from columnflow.tasks.histograms import MergeHistograms

# from columnflow.config_util import get_datasets_from_process
from hbw.tasks.base import HBWTask

ak = maybe_import("awkward")
hist = maybe_import("hist")
np = maybe_import("numpy")


logger = law.logger.get_logger(__name__)
logger_dev = law.logger.get_logger(f"{__name__}-dev")


class GetBtagNormalizationSF(
    HBWTask,
    SelectorClassMixin,
    CalibratorClassesMixin,
    DatasetsProcessesMixin,
    ShiftTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    resolution_task_cls = MergeSelectionStats

    single_config = True
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeSelectionStats=MergeSelectionStats,
    )

    store_as_dict = False

    reweighting_step = "selected_no_bjet"

    rescale_mode = luigi.ChoiceParameter(
        default="nevents",
        choices=("nevents", "xs"),
    )
    base_key = luigi.ChoiceParameter(
        default="rescaled_sum_mc_weight",
        # NOTE: "num_events" does not work because I did not store the corresponding key in the stats :/
        choices=("rescaled_sum_mc_weight", "sum_mc_weight", "num_events"),
    )

    # default sandbox, might be overwritten by selector function
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    processes = DatasetsProcessesMixin.processes.copy(
        default=("tt", "dy_m50toinf", "dy_m10to50"),
        description="Processes to consider for the scale factors",
        add_default_to_description=True,
    )

    def create_branch_map(self):
        # single branch without payload
        return {0: None}

    @cached_property
    def process_insts(self):
        process_insts = [self.config_inst.get_process(process) for process in self.processes]
        return process_insts

    @cached_property
    def dataset_insts(self):
        dataset_insts = [self.config_inst.get_dataset(dataset) for dataset in self.datasets]
        return dataset_insts

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["selection_stats"] = {
            dataset.name: self.reqs.MergeSelectionStats.req_different_branching(
                self,
                dataset=dataset.name,
                branch=-1,
            )
            for dataset in self.dataset_insts
        }
        return reqs

    def requires(self):
        reqs = {}
        reqs["selection_stats"] = {
            dataset.name: self.reqs.MergeSelectionStats.req_different_branching(
                self,
                dataset=dataset.name,
                branch=-1,
            )
            for dataset in self.dataset_insts
        }
        return reqs

    def store_parts(self):
        parts = super().store_parts()

        processes_repr = "__".join(self.processes)
        parts.insert_before("version", "processes", processes_repr)

        significant_params = (self.rescale_mode, self.base_key)
        parts.insert_before("version", "params", "__".join(significant_params))

        return parts

    def output(self):
        return {
            "btag_renormalization_sf": self.target("btag_renormalization_sf.json"),
        }

    def reduce_hist(self, hist, mode: list[str]):
        """
        Helper function that reduces the histogram to the requested axes based on the mode.
        """
        hist = hist[{"process": sum, "steps": self.reweighting_step}]

        # check validity of mode
        ax_names = [ax.name for ax in hist.axes]
        if not all(ax in ax_names for ax in mode):
            raise ValueError(f"Invalid mode {mode} for axes {ax_names}")

        # remove axes not in mode
        for ax in hist.axes:
            if ax.name not in mode:
                hist = hist[{ax.name: sum}]

        return hist

    def run(self):
        import correctionlib
        import correctionlib.convert
        outputs = self.output()
        inputs = self.input()

        # load the selection merged_hists
        hists_per_dataset = {
            dataset: inp["collection"][0]["hists"].load(formatter="pickle")
            for dataset, inp in inputs["selection_stats"].items()
        }

        def safe_div(num, den):
            return np.where(
                (num > 0) & (den > 0),
                num / den,
                1.0,
            )

        # rescale the histograms
        if "rescaled" in self.base_key:
            for dataset, hists in hists_per_dataset.items():
                process = self.config_inst.get_dataset(dataset).processes.get_first()
                if self.rescale_mode == "xs":
                    # scale such that the sum of weights is the cross section
                    xs = process.get_xsec(self.config_inst.campaign.ecm).nominal
                    dataset_factor = xs / hists["sum_mc_weight"][{"steps": "Initial"}].value
                elif self.rescale_mode == "nevents":
                    # scale such that mean weight is 1
                    n_events = hists["num_events"][{"steps": self.reweighting_step}].value
                    dataset_factor = n_events / hists["sum_mc_weight"][{"steps": self.reweighting_step}].value
                else:
                    raise ValueError(f"Invalid rescale mode {self.rescale_mode}")
                for key in tuple(hists.keys()):
                    if "sum" not in key:
                        continue
                    h = hists[key].copy() * dataset_factor
                    hists[f"rescaled_{key}"] = h

        # if necessary, merge the histograms across datasets
        if len(hists_per_dataset) > 1:
            merged_hists = {}
            from columnflow.tasks.selection import MergeSelectionStats
            for dataset, hists in hists_per_dataset.items():
                MergeSelectionStats.merge_counts(merged_hists, hists)
        else:
            merged_hists = hists_per_dataset[self.dataset_insts[0].name]

        # initialize the scale factor map
        sf_map = {}

        # TODO: mode "" (reduce everything to a single bin) not yet working
        for mode in (
            ("ht", "njet", "nhf"),
            ("ht", "njet"),
            ("ht",),
            ("njet",),
            ("nhf",),
            # ("",),
        ):
            mode_str = "_".join(mode)

            numerator = merged_hists[f"{self.base_key}_per_process_ht_njet_nhf"]
            numerator = self.reduce_hist(numerator, mode).values()

            for key in merged_hists.keys():
                if (
                    not key.startswith(f"{self.base_key}_btag_weight") or
                    not key.endswith("_per_process_ht_njet_nhf")
                ):
                    continue

                # extract the weight name
                weight_name = key.replace(f"{self.base_key}_", "").replace("_per_process_ht_njet_nhf", "")

                # create the scale factor histogram
                h = merged_hists[key]
                h = self.reduce_hist(h, mode)
                denominator = h.values()

                # get axes for the output histogram
                out_axes = []
                for ax in h.axes:
                    if isinstance(ax, hist.axis.Variable):
                        out_axes.append(ax)
                    elif isinstance(ax, hist.axis.Integer):
                        out_axes.append(hist.axis.Variable(ax.edges, name=ax.name, label=ax.label))
                    else:
                        raise ValueError(f"Unsupported axis type {type(ax)}")

                # calculate the scale factor and store it as a correctionlib evaluator
                sf = safe_div(numerator, denominator)
                sfhist = hist.Hist(*out_axes, data=sf)
                sfhist.name = f"{mode_str}_{weight_name}"
                sfhist.label = "out"

                # import correctionlib.convert
                btag_renormalization = correctionlib.convert.from_histogram(sfhist)
                btag_renormalization.description = f"{weight_name} per {mode_str} re-normalization"

                # set overflow bins behavior (default is to raise an error when out of bounds)
                # NOTE: claming seems to not work for int axes. Hopefully the number of jets considered to
                # create these SFs is always large enough to not hit the overflow bin.
                if any(isinstance(ax, hist.axis.Variable) for ax in out_axes):
                    btag_renormalization.data.flow = "clamp"

                # store the evaluator
                sf_map[sfhist.name] = btag_renormalization

        # create correction set and store it
        logger.info(f"Storing corrections with keys {sf_map.keys()}")
        cset = correctionlib.schemav2.CorrectionSet(
            schema_version=2,
            description="btag re-normalization SFs",
            corrections=list(sf_map.values()),
        )
        cset_json = cset.json(exclude_unset=True)
        if self.store_as_dict:
            import json
            cset_json = json.loads(cset_json)

        outputs["btag_renormalization_sf"].dump(
            cset_json,
            formatter="json",
        )


class GetNJetCorrections(
    HBWTask,
    # DatasetsProcessesMixin,
    HistogramsUserSingleShiftBase,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    Loads histograms containing the trigger informations for data and MC and calculates the trigger scale factors.
    """

    # hist_producers = law.CSVParameter(
    #     default=("with_dy_weight"),
    #     description="Weight producers to use for plotting",
    # )

    variables = law.CSVParameter(
        # default=("n_jet-n_forwardjet",),
        default=("n_jet",),
        description="Weight producers to use for plotting",
    )

    categories = law.CSVParameter(
        default=("dycr_nonmixed",),
        description="Weight producers to use for plotting",
    )

    # TODO: muss noch angewandt werden oder hardcoded argument anfodern fürs rebinning 
    # variable_settings = MultiSettingsParameter(
    #     default=DotDict(),
    #     significant=False,
    #     description="parameter for changing different variable settings; format: "
    #     "'var1,option1=value1,option3=value3:var2,option2=value2'; options implemented: "
    #     "rebin; can also be the key of a mapping defined in 'variable_settings_groups; "
    #     "default: value of the 'default_variable_settings' if defined, else empty default",
    #     brace_expand=True,
    # )

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeHistograms=MergeHistograms,
    )

    sm_processes = [
        "vv", "w_lnu", "st",
        "tt", "ttv", "h",
    ]

    sm_processes = DatasetsProcessesMixin.processes.copy(
        default=(sm_processes),
        description="Processes to consider for the scale factors",
        add_default_to_description=True,
    )

    corrected_processes = DatasetsProcessesMixin.processes.copy(
        default=("dy",),
        description="Processes to consider for the scale factors",
        add_default_to_description=True,
    )

    # @property
    # def corr_tag(self):
    #     if self.corrected_process[0] == "dy":
    #         return "is_dy"
    #     elif self.corrected_process[0] == "tt":
    #         return "is_ttbar"
    
    @cached_property
    def dataset_insts(self):
        dataset_insts = [self.config_inst.get_dataset(dataset) for dataset in self.datasets]
        return dataset_insts

    def output(self):
        return {
            f"{self.corrected_processes[0]}_njet_corrections": self.target(f"{self.corrected_processes[0]}_njet_corrections.json")
            # "trigger_sf_plot": self.target(f"{self.trigger}_{self.variables[0]}_sf_plot{self.suffix}.pdf"),
        }
    
    def create_branch_map(self):
        # single branch without payload
        return {0: None}


    def run(self):
        import correctionlib
        import correctionlib.convert

        def safe_div(num, den):
            return np.where(
                (num > 0) & (den > 0),
                num / den,
                1.0,
            )
    

        outputs = self.output()
        inputs = self.input()

        hists_in = defaultdict(OrderedDict)
        hists = {}  # defaultdict(OrderedDict)
        
        # sorted_processes = {
        #     "data": [],
        #     "MC_corr_process": [],
        #     "MC_subtract": [],
        # }
        # sorted_datasets = {
        #     "data": [],
        #     "MC_corr_process": [],
        #     "MC_subtract": [],
        # }
        # 
        # for variable in self.variables:
        #     for dataset_inst in self.dataset_insts: #TODO: vllt nicht beides dataset/process abspeciehrn gibt sicher was smartes was man in CF machen kann. Aber bis ich weiß was hier überhaupt passiert
        #         proc_name = dataset_inst.processes.get_first().name
        #         if dataset_inst.is_data:
        #             sorted_processes["data"].append(proc_name)
        #             sorted_datasets["data"].append(dataset_inst)
        #         # TODO: wenn meherer muss man auch mehrere angeben können 
        #         elif dataset_inst.has_tag(self.corr_tag):
        #             sorted_processes["MC_corr_process"].append(proc_name)
        #             sorted_datasets["MC_corr_process"].append(dataset_inst)
        #         else:
        #             sorted_processes["MC_subtract"].append(proc_name)
        #             sorted_datasets["MC_subtract"].append(dataset_inst)
                # hists_in[variable][dataset_inst.name] = self.load_histogram(dataset_inst.name, variable)

        for variable in self.variables:
            for dataset in self.datasets:
    
                h_in = self.load_histogram(dataset, variable)
                h_in = self.slice_histogram(h_in, self.processes, self.categories, self.shift)
                # __import__("IPython").embed()

                if variable in hists.keys():
                    hists[variable] += h_in
                else:
                    hists[variable] = h_in

        data_proc = self.config_inst.get_process('data')
        hists_corr = defaultdict(OrderedDict)
        hists_corr[variable]["data"] = self.slice_histogram(hists[variable], data_proc, self.categories, self.shift, reduce_axes=True)
        hists_corr[variable]["MC_subtract"] = self.slice_histogram(hists[variable], self.sm_processes, self.categories, self.shift, reduce_axes=True)
        hists_corr[variable]["MC_corr_process"] = self.slice_histogram(hists[variable], self.corrected_processes, self.categories, self.shift, reduce_axes=True)


                # h_data = self.slice_histogram(h_in, data_proc, self.categories, self.shift, reduce_axes=True)
                # h_non_corr = self.slice_histogram(h_in, self.sm_processes, self.categories, self.shift, reduce_axes=True)
                # h_corr = self.slice_histogram(h_in, self.corrected_processes, self.categories, self.shift, reduce_axes=True)
                # if variable in hists.keys():
                #     hists[variable]["data"] += h_data
                #     hists[variable]["MC_subtract"] += h_non_corr
                #     hists[variable]["MC_corr_process"] += h_corr
                # else:
                #     hists[variable]["data"] = h_data
                #     hists[variable]["MC_subtract"] = h_non_corr
                #     hists[variable]["MC_corr_process"] = h_corr
        

            # for mode in ["data", "MC_corr_process", "MC_subtract"]:
            #     hists[mode] = sum([
            #         hists_in[variable][dataset.name] for dataset in sorted_datasets[mode]
            #         if dataset.name in hists_in[variable].keys()
            #     ])
            #     hists[mode] = self.slice_histogram(hists[mode], sorted_processes[mode], self.categories, self.shift, reduce_axes=True)

        #TODO: Check how this is with 2d variables 

        nominator = hists_corr[variable]["data"].values() - hists_corr[variable]["MC_subtract"].values()
        denominator = hists_corr[variable]["MC_corr_process"].values()
        njet_correction = safe_div(nominator, denominator)

        __import__("IPython").embed()

        ax = hists_corr[variable]["data"].axes
        out_axes = hist.axis.Variable(ax[0].edges, name=ax[0].name, label=ax[0].label)

        out_axes = []
        for ax in hists_corr[variable]["data"].axes:
            if isinstance(ax, hist.axis.Variable):
                out_axes.append(ax)
            elif isinstance(ax, hist.axis.Integer):
                out_axes.append(hist.axis.Variable(ax.edges, name=ax.name, label=ax.label))
            else:
                raise ValueError(f"Unsupported axis type {type(ax)}")
                
        correction_hist = hist.Hist(*out_axes, data=njet_correction) # zero need to be here otherwise 1 length array is passed 
        correction_hist.name = f"{self.corrected_processes[0]}_njet_corrections"
        correction_hist.label = "out"
        
        njet_corrections = correctionlib.convert.from_histogram(correction_hist)
        njet_corrections.description = f"corrections for {self.corrected_processes[0]}, binned in njet"

        # TODO: Decied what do do with overflow bins
        njet_corrections.data.flow = "clamp"
        # create correction set and store it
        cset = correctionlib.schemav2.CorrectionSet(
            schema_version=2,
            description="njet corrections",
            corrections=[
                njet_corrections,
            ],
        )
        cset_json = cset.json(exclude_unset=True)

        outputs[f"{self.corrected_processes[0]}_njet_corrections"].dump(
            cset_json,
            formatter="json",
        )