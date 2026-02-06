# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from __future__ import annotations

import law
import functools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from hbw.production.prepare_objects import prepare_objects
# from hbw.production.jets import vbf_candidates
from hbw.config.dl.variables import add_dl_ml_variables

from hbw.production.ml_inputs import common_ml_inputs  # , vbf_jets, METCorr

ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

ZERO_PADDING_VALUE = -10


@producer(uses={prepare_objects, "*"}, produces={"dummy"})
def check_columns(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Check that all columns are present in the events.
    """
    # apply behavior (for variable reconstruction)
    events = self[prepare_objects](events, **kwargs)

    from hbw.util import debugger
    debugger()
    return events


def check_variable_existence(self: Producer) -> None:
    """
    Helper to check that all requested columns define a variable in the config
    """
    # check that all variables are defined in the config
    for column in self.ml_input_columns:
        if not self.config_inst.has_variable(column):
            raise ValueError(f"Variable {column} is not defined in the config.")


def check_column_bookkeeping(self: Producer, events: ak.Array) -> None:
    """
    Helper to check that all produced "mli" columns are bookkept in the config.
    """
    mli_fields = {field for field in events.fields if "mli_" in field}
    if diff := mli_fields - self.config_inst.x.ml_input_columns:
        raise ValueError(f"Extra fields in events: {diff}")


@producer(
    uses={common_ml_inputs},
    produces={common_ml_inputs},
    # produced columns set in the init function
    version=law.config.get_expanded("analysis", "dl_ml_inputs_version", 2),
)
def dl_ml_inputs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer used for ML Training in the DL analysis.
    """
    met_name = self.config_inst.x.met_name
    if self.dataset_inst.has_tag("is_dy"):
        met_name = "RecoilCorrMET"

    # produce common input features
    events = self[common_ml_inputs](events, **kwargs)

    # object padding
    events = set_ak_column(events, "Lepton", ak.pad_none(events.Lepton, 2))

    for var in ["pt", "eta"]:
        events = set_ak_column_f32(events, f"mli_lep2_{var}".lower(), events.Lepton[:, 1][var])

    events = set_ak_column_f32(events, "mli_lep_tag", abs(events.Lepton[:, 0]["pdgId"]) == 13)
    events = set_ak_column_f32(events, "mli_lep2_tag", abs(events.Lepton[:, 1]["pdgId"]) == 13)
    events = set_ak_column_f32(events, "mli_mixed_channel", events.mli_lep_tag != events.mli_lep2_tag)

    # create ll object and ll variables
    hll = (events.Lepton[:, 0] + events.Lepton[:, 1])
    events = set_ak_column_f32(events, "mli_ll_pt", hll.pt)
    events = set_ak_column_f32(events, "mli_mll", hll.mass)
    events = set_ak_column_f32(events, "mli_mllMET", (hll + events[met_name][:]).mass)
    events = set_ak_column_f32(events, "mli_dr_ll", events.Lepton[:, 0].delta_r(events.Lepton[:, 1]))
    events = set_ak_column_f32(events, "mli_dphi_ll", abs(events.Lepton[:, 0].delta_phi(events.Lepton[:, 1])))
    events = set_ak_column_f32(events, "mli_deta_ll", abs(events.Lepton[:, 0].eta - (events.Lepton[:, 1]).eta))

    # minimum deltaR between lep and jet
    hbb = (events.Bjet[:, 0] + events.Bjet[:, 1]) * 1  # NOTE: *1 so it is a Lorentzvector not a candidate vector

    llbb_pairs = ak.cartesian([events.Lepton, events.Bjet], axis=1)
    lep, jet = ak.unzip(llbb_pairs)
    min_dr_llbb = (ak.min(lep.delta_r(jet), axis=-1))
    events = set_ak_column_f32(events, "mli_min_dr_llbb", min_dr_llbb)

    events = set_ak_column_f32(events, "mli_dr_ll_bb", hll.delta_r(hbb))

    # hh system
    events = set_ak_column_f32(events, "mli_mbbllMET", (hll + hbb + events[met_name][:]).mass)
    events = set_ak_column_f32(events, "mli_dr_bb_llMET", hbb.delta_r(hll + events[met_name][:]))
    events = set_ak_column_f32(events, "mli_dphi_bb_nu", abs(hbb.delta_phi(events[met_name])))
    events = set_ak_column_f32(events, "mli_dphi_bb_llMET", hbb.delta_phi(hll + events[met_name][:]))

    # fill nan/none values of all produced columns
    for col in self.ml_input_columns:
        events = set_ak_column_f32(events, col, ak.fill_none(ak.nan_to_none(events[col]), ZERO_PADDING_VALUE))
    check_column_bookkeeping(self, events)
    return events


@dl_ml_inputs.init
def dl_ml_inputs_init(self: Producer) -> None:
    # define ML input separately to self.produces
    self.ml_input_columns = {
        # ll system
        "mli_mll", "mli_dr_ll", "mli_dphi_ll", "mli_deta_ll", "mli_ll_pt",
        "mli_min_dr_llbb",
        # hh system
        "mli_dr_ll_bb",
        "mli_dphi_bb_nu", "mli_dphi_bb_llMET", "mli_mllMET",
        "mli_mbbllMET", "mli_dr_bb_llMET",
        # low-level features
        "mli_lep2_pt", "mli_lep2_eta",
        "mli_lep_tag", "mli_lep2_tag", "mli_mixed_channel",
    }
    self.produces |= self.ml_input_columns

    # bookkeep used ml_input_columns over multiple Producers
    self.config_inst.x.ml_input_columns = self.config_inst.x("ml_input_columns", set()) | self.ml_input_columns

    # add variable instances to config
    add_dl_ml_variables(self.config_inst)
    check_variable_existence(self)
