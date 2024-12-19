# coding: utf-8

"""
Column production methods related to sample normalization event weights.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, InsertableDict
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
sp = maybe_import("scipy")
maybe_import("scipy.sparse")
ak = maybe_import("awkward")


@producer(
    uses={"mc_weight"},
    produces={"dataset_normalization_weight"},
    # only run on mc
    mc_only=True,
)
def dataset_normalization_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Uses luminosity information of internal py:attr:`config_inst`, the cross section of a process
    obtained from the dataset inst and the sum of event weights from the
    py:attr:`selection_stats` attribute to assign each event a normalization weight
    independent of the sub-processes of the dataset.
    Can only be used when there is a one-to-one mapping between datasets and processes.
    """
    # get the lumi
    lumi = self.config_inst.x.luminosity.nominal

    # compute the weight and store it
    norm_weight = events.mc_weight * lumi * self.xs / self.sum_weights
    events = set_ak_column(events, "dataset_normalization_weight", norm_weight, value_type=np.float32)

    return events


@dataset_normalization_weight.requires
def dataset_normalization_weight_requires(self: Producer, reqs: dict) -> None:
    """
    Adds the requirements needed by the underlying py:attr:`task` to access selection stats into
    *reqs*.
    """
    # TODO: for actual sample stitching, we don't need the selection stats for that dataset, but
    #       rather the one merged for either all datasets, or the "stitching group"
    #       (i.e. all datasets that might contain any of the sub processes found in a dataset)
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req(
        self.task,
        tree_index=0,
        branch=-1,
        _exclude=MergeSelectionStats.exclude_params_forest_merge,
    )


@dataset_normalization_weight.setup
def dataset_normalization_weight_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    """
    Load inclusive selection stats and cross sections for the normalization weight calculation.
    """
    # load the selection stats
    selection_stats = inputs["selection_stats"]["collection"][0]["stats"].load(formatter="json")

    process_inst = self.dataset_inst.processes.get_first()

    xs = process_inst.xsecs.get(self.config_inst.campaign.ecm, None)
    if not xs:
        raise Exception(f"no cross section found for process {process_inst.name}")

    self.xs = xs.nominal
    self.sum_weights = selection_stats["sum_mc_weight"]
