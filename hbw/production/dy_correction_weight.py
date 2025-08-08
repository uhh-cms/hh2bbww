# coding: utf-8

"""
Event weight producer.
"""

import law

from columnflow.util import maybe_import

from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column
from hbw.production.prepare_objects import prepare_objects

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


# Apply dy weight from ExportDYWeights task.
@producer(
    uses={
        "{Jet,Electron,Muon}.{pt,eta,phi,mass}",
        "gen_dilepton_pt",
        prepare_objects,
    },
    # both used columns and dependent shifts are defined in init below
    weight_columns=None,
    # only run on mc
    mc_only=True,
    # optional categorizer to obtain baseline event mask
    categorizer_cls=None,
    # corrected_process=None,
    from_file=False,
    # produced dy weight column
    produced_column="dy_correction_weight",
    uses_column=None,
    version=0,
)
def dy_correction_weight(
    self: Producer,
    events: ak.Array,
    task: law.Task,
    **kwargs,
) -> ak.Array:
    """
    Creates trigger scale factor weights using correctionlib. Requires external file.
    """
    events = self[prepare_objects](events, **kwargs)
    njet = ak.num(events.Jet.pt, axis=1)

    njets = ak.where(njet > 7, 7, njet)

    var_map = {
        "era": f"{task.config_inst.campaign.x.year}{task.config_inst.campaign.x.postfix}",
        # "era": self.dy_corr_era,
        "njets": njets,
        "ptll": events.gen_dilepton_pt,  # NOTE: reco ptll: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
        "syst": "nominal",
    }
    corrector = self.correction_set["dy_correction_weight"]

    inputs = [var_map[inp.name] for inp in corrector.inputs]

    dy_correction_weight = corrector.evaluate(*inputs)

    events = set_ak_column(events, self.produced_column, dy_correction_weight)
    return events


@dy_correction_weight.setup
def dy_correction_weight_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
) -> None:
    # create the corrector
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    # used when correction is stored as a JSON string

    self.correction_set = correctionlib.CorrectionSet.from_string(
        inputs["dy_correction_weight"].load(formatter="gzip").decode("utf-8"),
    )


@dy_correction_weight.init
def dy_correction_weight_init(self: Producer) -> None:

    if not self.dataset_inst.has_tag("is_dy"):
        return
    self.corrected_process = "dy"
    if "22" in self.config_inst.name:
        self.dy_corr_era = "2022_2022EE"
        self.dy_corr_configs = ("c22prev14", "c22postv14")
    else:
        self.dy_corr_era = "2023_2023BPix"
        self.dy_corr_configs = ("c23prev14", "c23postv14")
    # self._dy_corr_era = "2022_2022EE_2023_2023BPix"
    self.produces.add(self.produced_column)
    self.uses.add(self.uses_column)


@dy_correction_weight.requires
def dy_correction_weight_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    from hbw.tasks.dy_corrections import ExportDYWeights

    reqs["dy_correction_weight"] = ExportDYWeights.req(
        task,
        configs=(task.config,),
        # configs=self.dy_corr_configs,
        shift="nominal",
        processes=((
            "vv", "w_lnu", "st",
            "dy_m10to50", "dy_m50toinf",
            "tt", "ttv", "h", "data",
        ),),
        # Use Hist producer without dy corrections and cut for MET < 70 GeV to deplete ttbar contributions
        hist_producer="met70",
        # Use 2µ category when calculating weight with gen level ptll because of linear behaviour
        categories=("dycr__2mu",),
        variables=("n_jet-ptll_for_dy_corr",),
        njet_overflow=2,
    )


@dy_correction_weight.skip
def dy_correction_weight_skip(self: Producer) -> bool:
    """
    Skip if running on anything except ttbar MC simulation.
    """
    # never skip when there is no dataset
    if not getattr(self, "dataset_inst", None):
        return False

    return self.dataset_inst.is_data or not (
        self.dataset_inst.has_tag("is_dy")
    )


njet_weight_test = dy_correction_weight.derive("njet_weight_test", cls_dict={
    "produced_column": "njet_weight_test",
    "uses_column": "{Jet,Muon,Electron}.{pt,eta,phi,mass}",
})
