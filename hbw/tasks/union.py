# coding: utf-8

"""
Task to unite columns horizontally into a single file for further, possibly external processing.
"""

import luigi
import law

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    ProducerMixin,
    ProducersMixin, MLModelsMixin, ChunkedIOMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.reduction import ReducedEventsUser, MergeReducedEvents
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.ml import MLEvaluation
from columnflow.util import dev_sandbox
from columnflow.columnar_util import maybe_import

from hbw.tasks.base import HBWTask


np = maybe_import("numpy")
pd = maybe_import("pandas")


def sorted_ak_to_csv(events, path):
    is_int = lambda col: "int" in events[col].__repr__()
    fmt = ", ".join([
        "%i" if is_int(col) else "%1.4f"
        for col in events.fields
    ])
    header = ", ".join(events.fields)
    np.savetxt(
        path,
        np.array(events).T,
        delimiter=", ",
        header=header,
        fmt=fmt,
    )


def merge_csv_files(inputs: list[str], output: str):
    """
    Merge multiple CSV files into a single file.

    :param inputs: List of input file paths.
    :param output: Output file path.
    """
    dfs = [pd.read_csv(inp) for inp in inputs]
    df = pd.concat(dfs, axis=0)
    df.to_csv(output, index=False)


def merge_csv_task(task, inputs, output, local: bool = False, force: bool = True):
    """
    Merge multiple CSV files into a single file.

    :param task: The task object.
    :param inputs: List of input CSV files to merge.
    :param output: The task output for the CSV file.
    :param local: Flag indicating whether the merge should be performed locally or remotely. Default is False.
    :param force: Flag indicating whether to overwrite the output file if it already exists. Default is True.

    :raises Exception: If the output file is not created during merging.

    :return: None
    """
    with task.publish_step("merging {} csv files ...".format(len(inputs)), runtime=True):
        # clear the output if necessary
        if output.exists() and force:
            output.remove()

        # merge
        merge_csv_files(
            [inp.path for inp in inputs],
            output.path,
        )

    stat = output.exists(stat=True)
    if not stat:
        raise Exception("output '{}' not creating during merging".format(output.path))

    # print the size
    output_size = law.util.human_bytes(stat.st_size, fmt=True)
    task.publish_message("merged file size: {}".format(output_size))


class CustomUniteColumns(
    HBWTask,
    MLModelsMixin,
    ProducersMixin,
    ChunkedIOMixin,
    ReducedEventsUser,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    file_type = luigi.ChoiceParameter(
        default="parquet",
        choices=("csv", "parquet", "root"),
        description="the file type to create; choices: parquet,root; default: parquet",
    )

    union_producer = luigi.Parameter(
        default="sync_exercise",
        description="the producer to apply to the events; default: sync_exercise",
    )

    # upstream requirements
    reqs = Requirements(
        MergeReducedEvents.reqs,
        RemoteWorkflow.reqs,
        MergeReducedEvents=MergeReducedEvents,
        ProduceColumns=ProduceColumns,
        MLEvaluation=MLEvaluation,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # cache for producer inst
        self._union_producer_inst = law.no_value

    @property
    def union_producer_inst(self):
        if self._union_producer_inst is not law.no_value:
            # producer has already been cached
            return self._union_producer_inst

        producer = self.union_producer

        if not producer:
            # set producer inst to None when no producer is requested
            self._union_producer_inst = None
            return self._union_producer_inst

        self._union_producer_inst = ProducerMixin.get_producer_inst(producer, {"task": self})

        # overwrite the sandbox when set
        if self._union_producer_inst.sandbox:
            self.sandbox = self._union_producer_inst.sandbox

        return self._union_producer_inst

    def workflow_requires(self):
        reqs = super().workflow_requires()

        # require the full merge forest
        reqs["events"] = self.reqs.MergeReducedEvents.req(self, tree_index=-1)

        if not self.pilot:
            if self.producer_insts:
                reqs["producers"] = [
                    self.reqs.ProduceColumns.req(
                        self,
                        producer=producer_inst.cls_name,
                        producer_inst=producer_inst,
                    )
                    for producer_inst in self.producer_insts
                    if producer_inst.produced_columns
                ]
            if self.ml_model_insts:
                reqs["ml"] = [
                    self.reqs.MLEvaluation.req(self, ml_model=m)
                    for m in self.ml_models
                ]

        return reqs

    def requires(self):
        reqs = {
            "events": self.reqs.MergeReducedEvents.req(self, tree_index=self.branch, _exclude={"branch"}),
        }

        if self.producer_insts:
            reqs["producers"] = [
                self.reqs.ProduceColumns.req(
                    self,
                    producer=producer_inst.cls_name,
                    producer_inst=producer_inst,
                )
                for producer_inst in self.producer_insts
                if producer_inst.produced_columns
            ]
        if self.ml_model_insts:
            reqs["ml"] = [
                self.reqs.MLEvaluation.req(self, ml_model=m)
                for m in self.ml_models
            ]

        return reqs

    def output(self):
        return {"events": self.target(f"data_{self.branch}.{self.file_type}")}

    @law.decorator.log
    @law.decorator.localize(input=True, output=True)
    @law.decorator.safe_output
    def run(self):
        from columnflow.columnar_util import (
            ColumnCollection, Route, RouteFilter, mandatory_coffea_columns, update_ak_array,
            sorted_ak_to_parquet, sorted_ak_to_root,
        )

        # prepare inputs and outputs
        inputs = self.input()
        output = self.output()
        output_chunks = {}

        # create a temp dir for saving intermediate files
        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()

        # define columns that will be written
        write_columns = set()
        for c in self.config_inst.x.keep_columns.get(self.task_family, ["*"]):
            if isinstance(c, ColumnCollection):
                write_columns |= self.find_keep_columns(c)
            else:
                write_columns.add(Route(c))
        route_filter = RouteFilter(write_columns)

        # define columns that need to be read
        read_columns = write_columns | set(mandatory_coffea_columns)
        read_columns = {Route(c) for c in read_columns}

        # iterate over chunks of events and diffs
        files = [inputs["events"]["collection"][0]["events"].path]
        if self.producer_insts:
            files.extend([inp["columns"].path for inp in inputs["producers"]])
        if self.ml_model_insts:
            files.extend([inp["mlcolumns"].path for inp in inputs["ml"]])
        for (events, *columns), pos in self.iter_chunked_io(
            files,
            source_type=len(files) * ["awkward_parquet"],
            read_columns=len(files) * [read_columns],
        ):
            # optional check for overlapping inputs
            if self.check_overlapping_inputs:
                self.raise_if_overlapping([events] + list(columns))

            # add additional columns
            events = update_ak_array(events, *columns)

            # remove columns
            events = route_filter(events)

            # run union Producer
            if self.union_producer_inst:
                events = self.union_producer_inst(events)

            # optional check for finite values
            if self.check_finite_output:
                self.raise_if_not_finite(events)

            # save as parquet or root via a thread in the same pool
            chunk = tmp_dir.child(f"file_{pos.index}.{self.file_type}", type="f")
            output_chunks[pos.index] = chunk

            if self.file_type == "csv":
                self.chunked_io.queue(sorted_ak_to_csv, (events, chunk.path))
            elif self.file_type == "parquet":
                self.chunked_io.queue(sorted_ak_to_parquet, (events, chunk.path))
            else:  # root
                self.chunked_io.queue(sorted_ak_to_root, (events, chunk.path))

        # merge output files
        sorted_chunks = [output_chunks[key] for key in sorted(output_chunks)]
        if self.file_type == "csv":
            # probably not existing function
            merge_csv_task(self, sorted_chunks, output["events"], local=True)
        elif self.file_type == "parquet":
            law.pyarrow.merge_parquet_task(
                self, sorted_chunks, output["events"], local=True, writer_opts=self.get_parquet_writer_opts(),
            )
        else:  # root
            law.root.hadd_task(self, sorted_chunks, output["events"], local=True)
