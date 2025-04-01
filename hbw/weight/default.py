# coding: utf-8

"""
Event weight producer.
"""

import law
from functools import partial

from columnflow.util import maybe_import, InsertableDict
from columnflow.weight import WeightProducer, weight_producer
from columnflow.config_util import get_shifts_from_sources
from columnflow.columnar_util import Route, EMPTY_FLOAT
from hbw.production.prepare_objects import prepare_objects

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@weight_producer(uses={"mc_weight"}, mc_only=True)
def mc_weight(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, events.mc_weight


@weight_producer(uses={"normalization_weight"}, mc_only=True)
def norm(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, events.normalization_weight


@weight_producer(uses={"stitched_normalization_weight_brs_from_processes"}, mc_only=True)
def norm_brs_cmsdb(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, events.stitched_normalization_weight_brs_from_processes


@weight_producer(uses={"stitched_normalization_weight"}, mc_only=True)
def stitched_norm(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, events.stitched_normalization_weight


@weight_producer(mc_only=True)
def no_weights(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, ak.Array(np.ones(len(events), dtype=np.float32))

# @weight_producer(uses={"gen_hbw_decay.*.*"}, mc_only=True)
# def gen_barrel_test(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
#     for gp in events.gen_hbw_decay.fields:
#         mask = abs(events.gen_hbw_decay[gp]['eta']) >= 2.4 # purposely inverted 
#         # events = ak.where(mask, events, EMPTY_FLOAT)
#     return events, mask


@weight_producer(
    uses={prepare_objects},
    # both used columns and dependent shifts are defined in init below
    weight_columns=None,
    # only run on mc
    mc_only=False,
    # applying a mask
    mask_fn=None,
    # mask columns
    mask_columns=None,
    # optional categorizer to obtain baseline event mask
    categorizer_cls=None,
    # combines masks with or boolean
    combine_or=False,
)
def base(self: WeightProducer, events: ak.Array, **kwargs) -> ak.Array:
    # apply behavior (for variable reconstruction)
    events = self[prepare_objects](events, **kwargs)

    # apply mask
    if self.categorizer_cls:
        events, mask = self[self.categorizer_cls](events, **kwargs)
        events = events[mask]

    #apply cutsom masks
    if self.mask_fn:
        # __import__("IPython").embed()
        if self.combine_or:
            mask = events.process_id < -9999
            for m in self.mask_fn.keys():
                # mask = immer false 
                mask = self.mask_fn[m](events) | mask
            events = events[mask]
        else: 
            for m in self.mask_fn.keys():
                mask = self.mask_fn[m](events)
                events = events[mask]

    if self.dataset_inst.is_data:
        return events, ak.Array(np.ones(len(events), dtype=np.float32))

    # build the full event weight
    weight = ak.Array(np.ones(len(events), dtype=np.float32))
    for column in self.local_weight_columns.keys():
        weight = weight * Route(column).apply(events)

    # implement dummy shift by varying weight by factor of 2
    if "dummy" in self.local_shift_inst.name:
        logger.warning("Applying dummy weight shift (should never be use for real analysis)")
        variation = self.local_shift_inst.name.split("_")[-1]
        weight = weight * {"up": 2.0, "down": 0.5}[variation]

    return events, weight


@base.setup
def base_setup(
    self: WeightProducer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    logger.info(
        f"WeightProducer '{self.cls_name}' (dataset {self.dataset_inst}) uses weight columns: \n"
        f"{', '.join(self.weight_columns.keys())}",
    )


@base.init
def base_init(self: WeightProducer) -> None:
    # NOTE: this might be called multiple times, might be quite inefficient
    # if not getattr(self, "config_inst", None) or not getattr(self, "dataset_inst", None):
    #     return

    if not getattr(self, "config_inst"):
        return

    if self.categorizer_cls:
        self.uses.add(self.categorizer_cls)
    
    if self.mask_columns:
        for col in self.mask_columns:
            self.uses.add(col)

    dataset_inst = getattr(self, "dataset_inst", None)
    if dataset_inst and dataset_inst.is_data:
        return

    year = self.config_inst.campaign.x.year
    cpn_tag = self.config_inst.x.cpn_tag

    if not self.weight_columns:
        raise Exception("weight_columns not set")
    self.local_weight_columns = self.weight_columns.copy()

    if dataset_inst and dataset_inst.has_tag("skip_scale"):
        # remove dependency towards mur/muf weights
        for column in [
            "normalized_mur_weight", "normalized_muf_weight", "normalized_murmuf_envelope_weight",
            "mur_weight", "muf_weight", "murmuf_envelope_weight",
        ]:
            self.local_weight_columns.pop(column, None)

    if dataset_inst and dataset_inst.has_tag("skip_pdf"):
        # remove dependency towards pdf weights
        for column in ["pdf_weight", "normalized_pdf_weight"]:
            self.local_weight_columns.pop(column, None)

    if dataset_inst and not dataset_inst.has_tag("is_ttbar"):
        # remove dependency towards top pt weights
        self.local_weight_columns.pop("top_pt_weight", None)

    if dataset_inst and not dataset_inst.has_tag("is_v_jets"):
        # remove dependency towards vjets weights
        self.local_weight_columns.pop("vjets_weight", None)

    self.shifts = set()

    # when jec sources are known btag SF source, then propagate the shift to the WeightProducer
    # TODO: we should do this somewhere centrally
    btag_sf_jec_sources = (
        (set(self.config_inst.x.btag_sf_jec_sources) | {"Total"}) &
        set(self.config_inst.x.jec.Jet["uncertainty_sources"])
    )
    self.shifts |= set(get_shifts_from_sources(
        self.config_inst,
        *[f"jec_{jec_source}" for jec_source in btag_sf_jec_sources],
    ))

    for weight_column, shift_sources in self.local_weight_columns.items():
        shift_sources = law.util.make_list(shift_sources)
        shift_sources = [s.format(year=year, cpn_tag=cpn_tag) for s in shift_sources]
        shifts = get_shifts_from_sources(self.config_inst, *shift_sources)
        for shift in shifts:
            if weight_column not in shift.x("column_aliases").keys():
                # make sure that column aliases are implemented
                raise Exception(
                    f"Weight column {weight_column} implements shift {shift}, but does not use it "
                    f"in 'column_aliases' aux {shift.x('column_aliases')}",
                )

            # declare shifts that the produced event weight depends on
            self.shifts |= set(shifts)

    # remove dummy column from weight columns and uses
    self.local_weight_columns.pop("dummy_weight", "")

    # store column names referring to weights to multiply
    self.uses |= self.local_weight_columns.keys()


btag_uncs = [
    "hf", "lf", "hfstats1_{year}", "hfstats2_{year}",
    "lfstats1_{year}", "lfstats2_{year}", "cferr1", "cferr2",
]


default_correction_weights = {
    # "dummy_weight": ["dummy_{cpn_tag}"],
    "normalized_pu_weight": ["minbias_xs"],
    "muon_id_weight": ["mu_id_sf"],
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
    "normalized_murmuf_envelope_weight": ["murf_envelope"],
    "normalized_mur_weight": ["mur"],
    "normalized_muf_weight": ["muf"],
    "normalized_pdf_weight": ["pdf"],
    "top_pt_weight": ["top_pt"],
}

default_weight_columns = {
    "stitched_normalization_weight": [],
    **default_correction_weights,
}
default_weight_producer = base.derive("default", cls_dict={"weight_columns": default_weight_columns})
with_vjets_weight = default_weight_producer.derive("with_vjets_weight", cls_dict={"weight_columns": {
    **default_correction_weights,
    "vjets_weight": [],  # TODO: corrections/shift missing
    "stitched_normalization_weight": [],
}})
with_trigger_weight = default_weight_producer.derive("with_trigger_weight", cls_dict={"weight_columns": {
    **default_correction_weights,
    "vjets_weight": [],  # TODO: corrections/shift missing
    "trigger_weight": [],  # TODO: corrections/shift missing
    "stitched_normalization_weight": [],
}})


base.derive("unstitched", cls_dict={"weight_columns": {
    **default_correction_weights, "normalization_weight": [],
}})

base.derive("minimal", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "muon_id_weight": ["mu_id_sf"],
    "muon_iso_weight": ["mu_iso_sf"],
    "electron_weight": ["e_sf"],
    "top_pt_weight": ["top_pt"],
}})

weight_columns_execpt_btag = default_weight_columns.copy()
weight_columns_execpt_btag.pop("normalized_ht_njet_nhf_btag_weight")

no_btag_weight = base.derive("no_btag_weight", cls_dict={"weight_columns": weight_columns_execpt_btag})
base.derive("btag_not_normalized", cls_dict={"weight_columns": {
    **weight_columns_execpt_btag,
    "btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("btag_njet_normalized", cls_dict={"weight_columns": {
    **weight_columns_execpt_btag,
    "normalized_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("btag_ht_njet_normalized", cls_dict={"weight_columns": {
    **weight_columns_execpt_btag,
    "normalized_ht_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("btag_ht_njet_nhf_normalized", cls_dict={"weight_columns": {
    **weight_columns_execpt_btag,
    "normalized_ht_njet_nhf_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("btag_ht_normalized", cls_dict={"weight_columns": {
    **weight_columns_execpt_btag,
    "normalized_ht_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})

# weight sets for closure tests
base.derive("norm_and_btag", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("norm_and_btag_njet", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "normalized_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("norm_and_btag_ht_njet", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "normalized_ht_njet_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})
base.derive("norm_and_btag_ht", cls_dict={"weight_columns": {
    "stitched_normalization_weight": [],
    "normalized_ht_btag_weight": [f"btag_{unc}" for unc in btag_uncs],
}})

# applying barrel region cut 
gen_barrel = base.derive("gen_barrel", cls_dict={ 
    "mc_only": True,
    "weight_columns": default_weight_columns,
    "mask_fn": {
        "sec1": lambda events: abs(events.gen_hbw_decay['sec1']['eta']) < 2.4, 
        "sec2": lambda events: abs(events.gen_hbw_decay['sec2']['eta']) < 2.4 ,
    },
    # "mask_fn": {
    #     gp: partial(lambda self, events, gp: abs(events.gen_hbw_decay[gp]['eta']) < 2.4, self, gp=gp)
    #     for gp in ["sec1", "sec2", "h1", "h2", "b1", "b2"] --> can be used, if i exclude self etc... 

    # },
          #  for gp in events.gen_hbw_decay.fields),
    "mask_columns": ["gen_hbw_decay.*.*"],
})

ptgeq50 = base.derive("ptgeq50", cls_dict={ 
    "mc_only": False,
    "weight_columns": default_weight_columns,
    "mask_fn": {
        "jet0": lambda events: abs(events.customjet0_pt) >= 50,
        "jet1": lambda events: abs(events.customjet1_pt) >= 50,

    },
    "mask_columns": ["customjet*"],
})

ptgeq50horn = base.derive("ptgeq50horn", cls_dict={ 
    "mc_only": False,
    "combine_or": True,
    "weight_columns": default_weight_columns,
    "mask_fn": {
        "horn": lambda events: (abs(events.customjet0_eta) > 2.6) & (abs(events.customjet0_eta) < 3.1) & (abs(events.customjet0_pt) >= 50),
        "outsidehorn": lambda events: ((abs(events.customjet0_eta) <= 2.6) | (abs(events.customjet0_eta) >= 3.1)) & (abs(events.customjet0_pt) >= 30),

    },
    "mask_columns": ["customjet*"],
})

hornhandle = base.derive("hornhandle", cls_dict={ 
"mc_only": False,
    "combine_or": True,
    "weight_columns": default_weight_columns,
    "mask_fn": {
        "horn": lambda events: (abs(events.customjet0_eta) >= 2.6) & (abs(events.customjet0_eta) <= 3.1) & (abs(events.customjet0_pt) >= 50),
        "outsidehorn": lambda events: ((abs(events.customjet0_eta) < 2.6) | (abs(events.customjet0_eta) > 3.1)) & (abs(events.customjet0_pt) >= 30),

    },
    "mask_columns": ["customjet*"],
})

horn50forward40 = base.derive("horn50forward40", cls_dict={ 
    "mc_only": False,
    "combine_or": True,
    "weight_columns": default_weight_columns,
    "mask_fn": {
        "horn": lambda events: (abs(events.customjet0_eta) > 2.6) & (abs(events.customjet0_eta) < 3.1) & (abs(events.customjet0_pt) >= 50),
        "forward": lambda events: ((abs(events.customjet0_eta) >= 3.1)) & (abs(events.customjet0_pt) >= 40),

    },
    "mask_columns": ["customjet*"],
})

horn50forward40barrel30 = base.derive("horn50forward40barrel30", cls_dict={ 
    "mc_only": False,
    "combine_or": True,
    "weight_columns": default_weight_columns,
    "mask_fn": {
        "horn": lambda events: (abs(events.customjet0_eta) > 2.6) & (abs(events.customjet0_eta) < 3.1) & (abs(events.customjet0_pt) >= 50),
        "forward": lambda events: ((abs(events.customjet0_eta) >= 3.1)) & (abs(events.customjet0_pt) >= 40),
        "barrel": lambda events: ((abs(events.customjet0_eta) <= 2.6)) & (abs(events.customjet0_pt) >= 30),

    },
    "mask_columns": ["customjet*"],
})

forward50barrel30 = base.derive("forward50barrel30", cls_dict={ 
    "mc_only": False,
    "combine_or": True,
    "weight_columns": default_weight_columns,
    "mask_fn": {
        "forward0": lambda events: (abs(events.customjet0_eta) > 2.6) & (abs(events.customjet0_pt) > 50),
        "barrel0": lambda events: ((abs(events.customjet0_eta) <= 2.6)) & (abs(events.customjet0_pt) > 30),
        "forward1": lambda events: (abs(events.customjet1_eta) > 2.6) & (abs(events.customjet1_pt) > 40),
        "barrel1": lambda events: ((abs(events.customjet1_eta) <= 2.6)) & (abs(events.customjet1_pt) > 30),
        #"barrel1": lambda events: ((abs(events.customjet1_eta) <= 2.6)) & (abs(events.customjet1_pt) > 30),

    },
    "mask_columns": ["customjet*"],
})

horn70forward50barrel30 = base.derive("horn70forward50barrel30", cls_dict={ 
    "mc_only": False,
    "combine_or": True,
    "weight_columns": default_weight_columns,
    "mask_fn": {
        "horn": lambda events: (abs(events.customjet0_eta) > 2.6) & (abs(events.customjet0_eta) < 3.1) & (abs(events.customjet0_pt) > 70),
        "forward": lambda events: ((abs(events.customjet0_eta) >= 3.1)) & (abs(events.customjet0_pt) > 50),
        "barrel": lambda events: ((abs(events.customjet0_eta) <= 2.6)) & (abs(events.customjet0_pt) > 30),

    },
    "mask_columns": ["customjet*"],
})

horn50forward50barrel30 = base.derive("horn50forward50barrel30", cls_dict={ 
    "mc_only": False,
    "combine_or": True,
    "weight_columns": default_weight_columns,
    "mask_fn": {
        "horn": lambda events: (abs(events.customjet0_eta) > 2.6) & (abs(events.customjet0_eta) < 3.1) & (abs(events.customjet0_pt) >= 50),
        "forward": lambda events: ((abs(events.customjet0_eta) >= 3.1)) & (abs(events.customjet0_pt) >= 50),
        "barrel": lambda events: ((abs(events.customjet0_eta) <= 2.6)) & (abs(events.customjet0_pt) >= 30),

    },
    "mask_columns": ["customjet*"],
})

gen_barrel_sec1 = base.derive("gen_barrel_sec1", cls_dict={ 
    "mc_only": True,
    "weight_columns": default_weight_columns,
    "mask_fn": {
        "sec1": lambda events: abs(events.gen_hbw_decay['sec1']['eta']) < 2.4, 
        # "sec2": lambda events: abs(events.gen_hbw_decay['sec2']['eta']) < 2.4 ,
    },
    # "mask_fn": {
    #     gp: partial(lambda self, events, gp: abs(events.gen_hbw_decay[gp]['eta']) < 2.4, self, gp=gp)
    #     for gp in ["sec1", "sec2", "h1", "h2", "b1", "b2"] --> can be used, if i exclude self etc... 

    # },
          #  for gp in events.gen_hbw_decay.fields),
    "mask_columns": ["gen_hbw_decay.*.*"],
})

from hbw.categorization.categories import mask_fn_highpt


no_btag_weight.derive("no_btag_weight_highpt", cls_dict={"categorizer_cls": mask_fn_highpt})
