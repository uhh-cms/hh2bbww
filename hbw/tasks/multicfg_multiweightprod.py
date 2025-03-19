# flake8: noqa
class PlotVariablesMultiConfigMultiWeightProducer(
    HBWTask,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ProducerClassesMixin,
    MLModelsMixin,
    CategoriesMixin,
    ProcessPlotSettingMixin,
    DatasetsProcessesMixin,
    VariablePlotSettingMixin,
    # HistHookMixin,
    ShiftTask,
    PlotBase1D,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    # use the MergeHistograms task to trigger upstream TaskArrayFunction initialization
    single_config = False
    resolution_task_class = MergeHistograms
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    weight_producers = law.CSVParameter(
        default=(),
        description="Weight producers to use for plotting",
    )

    plot_function = PlotBase.plot_function.copy(
        default="hbw.tasks.plotting.plot_multi_weight_producer",
        add_default_to_description=True,
    )

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeHistograms=MergeHistograms,
    )

    @classmethod
    def resolve_instances(cls, params, shifts: TaskShifts | None = None):
        if not cls.resolution_task_class:
            raise ValueError(f"resolution_task_class must be set for multi-config task {cls.task_family}")

        if shifts is None:
            shifts = TaskShifts()
        # we loop over all configs/datasets, but return initial params
        for i, config_inst in enumerate(params["config_insts"]):
            if cls.has_single_config():
                datasets = params["datasets"]
            else:
                datasets = params["datasets"][i]

            for weight_producer in params["weight_producers"]:
                for dataset in datasets:
                    # NOTE: we need to copy here, because otherwise taf inits will only be triggered once
                    _params = params.copy()
                    _params["config_inst"] = config_inst
                    _params["config"] = config_inst.name
                    _params["dataset"] = dataset
                    _params["weight_producer"] = weight_producer
                    logger.warning(f"building taf insts for {config_inst.name}, {dataset}")
                    _params = cls.resolution_task_class.resolve_instances(_params, shifts)
                    cls.resolution_task_class.get_known_shifts(_params, shifts)

        params["known_shifts"] = shifts

        return params

    def requires(self):
        req = {}
        for weight_producer in self.weight_producers:
            req[weight_producer] = {}
            for i, config_inst in enumerate(self.config_insts):
                sub_datasets = self.datasets[i]
                req[weight_producer][config_inst.name] = {}
                for d in sub_datasets:
                    if d in config_inst.datasets.names():
                        req[weight_producer][config_inst.name][d] = self.reqs.MergeHistograms.req(
                            self,
                            config=config_inst.name,
                            # shift=self.global_shift_insts[config_inst.name],
                            dataset=d,
                            weight_producer=weight_producer,
                            branch=-1,
                            _exclude={"branches"},
                            _prefer_cli={"variables"},
                        )
        return req

    def output(self):
        b = self.branch_data
        return {"plots": [
            self.target(name)
            for name in self.get_plot_names(f"plot__proc_{self.processes_repr}__cat_{b.category}__var_{b.variable}")
        ]}

    def get_plot_shifts(self):
        return [self.shift]

    @property
    def weight_producers_repr(self):
        return "_".join(self.weight_producers)

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "plot", f"datasets_{self.datasets_repr}")
        parts.insert_before("version", "weights", f"weights_{self.weight_producers_repr}")
        return parts

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name, "variable": var_name})
            for cat_name in sorted(self.categories)
            for var_name in sorted(self.variables)
        ]

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["merged_hists"] = self.requires_from_branch()

        return reqs

    @law.decorator.log
    @view_output_plots
    def run(self):
        import hist

        # get the shifts to extract and plot
        plot_shifts = law.util.make_list(self.get_plot_shifts())

        # prepare config objects
        variable_tuple = self.variable_tuples[self.branch_data.variable]
        variable_insts = [
            self.config_insts[0].get_variable(var_name)
            for var_name in variable_tuple
        ]

        # histogram data per process
        hists = defaultdict(OrderedDict)

        # NOTE: loading histograms as implemented here might not be consistent anymore with
        # how it's done in PlotVariables1D (important when datasets/shifts differ per config/weightProducer/...)
        with self.publish_step(f"plotting {self.branch_data.variable} in {self.branch_data.category}"):
            for weight_producer, inputs in self.input().items():
                for config, _inputs in inputs.items():
                    config_inst = self.analysis_inst.get_config(config)
                    plot_shift_insts = [config_inst.get_shift(shift) for shift in plot_shifts]

                    category_inst = config_inst.get_category(self.branch_data.category)
                    leaf_category_insts = category_inst.get_leaf_categories() or [category_inst]
                    process_insts = list(map(config_inst.get_process, self.processes))
                    sub_process_insts = {
                        proc: [sub for sub, _, _ in proc.walk_processes(include_self=True)]
                        for proc in process_insts
                    }

                    for dataset, inp in _inputs.items():
                        dataset_inst = config_inst.get_dataset(dataset)
                        h_in = inp["collection"][0]["hists"].targets[self.branch_data.variable].load(formatter="pickle")

                        # loop and extract one histogram per process
                        for process_inst in process_insts:
                            # skip when the dataset is already known to not contain any sub process
                            if not any(map(dataset_inst.has_process, sub_process_insts[process_inst])):
                                continue

                            # work on a copy
                            h = h_in.copy()

                            # axis selections
                            h = h[{
                                "process": [
                                    hist.loc(p.id)
                                    for p in sub_process_insts[process_inst]
                                    if p.id in h.axes["process"]
                                ],
                                "category": [
                                    hist.loc(c.name)
                                    for c in leaf_category_insts
                                    if c.name in h.axes["category"]
                                ],
                                "shift": [
                                    hist.loc(s.name)
                                    for s in plot_shift_insts
                                    if s.name in h.axes["shift"]
                                ],
                            }]

                            # axis reductions
                            h = h[{"process": sum, "category": sum}]

                            # add the histogram
                            if weight_producer in hists and process_inst in hists[weight_producer]:
                                hists[weight_producer][process_inst] = hists[weight_producer][process_inst] + h
                            else:
                                hists[weight_producer][process_inst] = h

                # there should be hists to plot
                if not hists:
                    raise Exception(
                        "no histograms found to plot; possible reasons:\n" +
                        "  - requested variable requires columns that were missing during histogramming\n" +
                        "  - selected --processes did not match any value on the process axis of the input histogram",
                    )

                # sort hists by process order
                hists[weight_producer] = OrderedDict(
                    (process_inst.copy_shallow(), hists[weight_producer][process_inst])
                    for process_inst in sorted(hists[weight_producer], key=process_insts.index)
                )

            hists = dict(hists)

            if len(self.config_insts) == 0:
                config_inst = self.config_insts[0]
            else:
                # take first config and correct luminosity label in case of multiple configs
                config_inst = self.config_insts[0].copy(id=-1, name=f"{self.config_insts[0].name}_merged")
                config_inst.x.luminosity = sum([_config_inst.x.luminosity for _config_inst in self.config_insts])

            # call the plot function
            fig, _ = self.call_plot_func(
                self.plot_function,
                hists=hists,
                config_inst=config_inst,
                category_inst=category_inst.copy_shallow(),
                variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
                **self.get_plot_parameters(),
            )

            # save the plot
            for outp in self.output()["plots"]:
                outp.dump(fig, formatter="mpl")
