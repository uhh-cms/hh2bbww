# coding: utf-8

"""
Config-related object definitions and utils.
"""

from __future__ import annotations


from typing import Callable, Any, Sequence

import order as od
from order import UniqueObject, TagMixin, AuxDataMixin
from order.util import typed


class TriggerLeg(object):
    """
    Container class storing information about trigger legs:

        - *pdg_id*: The id of the object that should have caused the trigger leg to fire.
        - *min_pt*: The minimum transverse momentum in GeV of the triggered object.
        - *trigger_bits*: Integer bit mask or masks describing whether the last filter of a trigger fired.
          See https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
          Per mask, any of the bits should match (*OR*). When multiple masks are configured, each of
          them should match (*AND*).

    For accepted types and conversions, see the *typed* setters implemented in this class.
    """

    def __init__(
        self,
        pdg_id: int | None = None,
        min_pt: float | int | None = None,
        trigger_bits: int | Sequence[int] | None = None,
    ):
        super().__init__()

        # instance members
        self._pdg_id = None
        self._min_pt = None
        self._trigger_bits = None

        # set initial values
        self.pdg_id = pdg_id
        self.min_pt = min_pt
        self.trigger_bits = trigger_bits

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"'pdg_id={self.pdg_id}, min_pt={self.min_pt}, trigger_bits={self.trigger_bits}' "
            f"at {hex(id(self))}>"
        )

    @typed
    def pdg_id(self, pdg_id: int | None) -> int | None:
        if pdg_id is None:
            return None

        if not isinstance(pdg_id, int):
            raise TypeError(f"invalid pdg_id: {pdg_id}")

        return pdg_id

    @typed
    def min_pt(self, min_pt: int | float | None) -> float | None:
        if min_pt is None:
            return None

        if isinstance(min_pt, int):
            min_pt = float(min_pt)
        if not isinstance(min_pt, float):
            raise TypeError(f"invalid min_pt: {min_pt}")

        return min_pt

    @typed
    def trigger_bits(
        self,
        trigger_bits: int | Sequence[int] | None,
    ) -> list[int] | None:
        if trigger_bits is None:
            return None

        # cast to list
        if isinstance(trigger_bits, tuple):
            trigger_bits = list(trigger_bits)
        elif not isinstance(trigger_bits, list):
            trigger_bits = [trigger_bits]

        # check bit types
        for bit in trigger_bits:
            if not isinstance(bit, int):
                raise TypeError(f"invalid trigger bit: {bit}")

        return trigger_bits


class Trigger(UniqueObject, AuxDataMixin, TagMixin):
    """
    Container class storing information about triggers:

        - *name*: The path name of a trigger that should have fired.
        - *id*: A unique id of the trigger.
        - *run_range*: An inclusive range describing the runs where the trigger is to be applied
          (usually only defined by data). None in the tuple means no lower or upper boundary.
        - *legs*: A list of :py:class:`TriggerLeg` objects contraining additional information and
          constraints of particular trigger legs.
        - *applies_to_dataset*: A function that obtains an ``order.Dataset`` instance to decide
          whether the trigger applies to that dataset. Defaults to *True*.

    For accepted types and conversions, see the *typed* setters implemented in this class.

    In addition, a base class from *order* provides additional functionality via mixins:

        - *tags*: Trigger objects can be assigned *tags* that can be checked later on, e.g. to
          describe the type of the trigger ("single_mu", "cross", ...).
    """

    def __init__(
        self,
        name: str,
        id: int,
        run_range: tuple[int | None, int | None] | None = None,
        legs: Sequence[TriggerLeg] | None = None,
        applies_to_dataset: Callable | bool | Any = True,
        aux: Any = None,
        tags: Any = None,
    ):
        UniqueObject.__init__(self, name, id)
        AuxDataMixin.__init__(self, aux=aux)
        TagMixin.__init__(self, tags=tags)

        # force the id to be positive
        if self.id < 0:
            raise ValueError(f"trigger id must be positive, but found {self.id}")

        # instance members
        self._run_range = None
        self._leg = None
        self._applies_to_dataset = None

        # set initial values
        self.run_range = run_range
        self.legs = legs
        self.applies_to_dataset = applies_to_dataset

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} 'name={self.name}, nlegs={self.n_legs}' "
            f"at {hex(id(self))}>"
        )

    @typed
    def name(self, name: str) -> str:
        if not isinstance(name, str):
            raise TypeError(f"invalid name: {name}")
        if not name.startswith("HLT_"):
            raise ValueError(f"invalid name: {name}")

        return name

    @typed
    def run_range(
        self,
        run_range: Sequence[int] | None,
    ) -> tuple[int] | None:
        if run_range is None:
            return None

        # cast list to tuple
        if isinstance(run_range, list):
            run_range = tuple(run_range)

        # run_range must be a tuple with two integers
        if not isinstance(run_range, tuple):
            raise TypeError(f"invalid run_range: {run_range}")
        if len(run_range) != 2:
            raise ValueError(f"invalid run_range length: {run_range}")
        if not isinstance(run_range[0], int):
            raise ValueError(f"invalid run_range start: {run_range[0]}")
        if not isinstance(run_range[1], int):
            raise ValueError(f"invalid run_range end: {run_range[1]}")

        return run_range

    @typed
    def legs(
        self,
        legs: (
            dict |
            tuple[dict] |
            list[dict] |
            TriggerLeg |
            tuple[TriggerLeg] |
            list[TriggerLeg] |
            None
        ),
    ) -> list[TriggerLeg]:
        if legs is None:
            return None

        if isinstance(legs, tuple):
            legs = list(legs)
        elif not isinstance(legs, list):
            legs = [legs]

        _legs = []
        for leg in legs:
            if isinstance(leg, dict):
                leg = TriggerLeg(**leg)
            if not isinstance(leg, TriggerLeg):
                raise TypeError(f"invalid trigger leg: {leg}")
            _legs.append(leg)

        return _legs or None

    @typed
    def applies_to_dataset(self, func: Callable | bool | Any) -> Callable:
        if not callable(func):
            decision = True if func is None else bool(func)
            func = lambda dataset_inst: decision

        return func

    @property
    def has_legs(self):
        return bool(self._legs)

    @property
    def n_legs(self):
        return len(self.legs) if self.has_legs else 0

    @property
    def hlt_field(self):
        # remove the first four "HLT_" characters
        return self.name[4:]


from hbw.util import call_once_on_config


@call_once_on_config()
def add_triggers(config: od.Config) -> od.UniqueObjectIndex[Trigger]:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    Electron Trigger: https://twiki.cern.ch/twiki/bin/view/CMS/EgHLTRunIIISummary
    Muon Trigger: https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLT2022

    trigger_bits are obtained from the TrigObj.filterBits docstring, by running some task and
    starting an embed shell, e.g. via:
        law run cf.SelectEvents --selector check_columns
        events.TrigObj.filterBits?

    Auxiliary data in use:
    - "channels": list of channels during selection that the trigger applies to,
    e.g. ["e", "ee", "emu", "mue"] (TODO: use this in SL aswell)
    - "data_stream": the data stream that is associated to events that fired the trigger to
    prevent double counting of data events

    Tags are not being used at the moment.
    """
    # only 2022 triggers for now
    single_mu = Trigger(
        name="HLT_IsoMu24",
        id=101,
        legs=[
            TriggerLeg(
                pdg_id=13,
                min_pt=24.0,
                # filter names:
                # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p08 (1mu + Iso)
                trigger_bits=2**1 + 2**3,  # Iso (bit 1) + 1mu (bit 3)
            ),
        ],
        aux={
            "channels": ["mu", "mm", "emu", "mue", "mixed"],
            "data_stream": "data_mu",
        },
        tags={"single_trigger", "single_mu"},
    )
    di_mu = Trigger(
        name="HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        id=102,
        legs=[
            TriggerLeg(
                pdg_id=13,
                min_pt=17.0,
                # filter names:
                # TODO
                trigger_bits=2**0 + 2**4,  # TrkIsoVVL (bit 0) + 2mu (bit 4)
            ),
            TriggerLeg(
                pdg_id=13,
                min_pt=8.0,
                # filter names:
                # TODO
                trigger_bits=2**0 + 2**4,  # TrkIsoVVL (bit 0) + 2mu (bit 4) + DZ_Mass3p8 (bit ?)
            ),
        ],
        aux={
            "channels": ["mm"],
            "data_stream": "data_mu",
        },
        tags={"di_trigger", "di_mu"},
    )
    single_e = Trigger(
        name="HLT_Ele30_WPTight_Gsf",
        id=201,
        legs=[
            TriggerLeg(
                pdg_id=11,
                min_pt=30.0,
                # filter names:
                # hltEle30WPTightGsfTrackIsoFilter
                trigger_bits=2**1,  # 1e (WPTight) (bit 1)
            ),
        ],
        aux={
            "channels": ["e", "ee", "emu", "mue", "mixed"],
            "data_stream": "data_egamma" if config.x.run == 3 else "data_e",
        },
        tags={"single_trigger", "single_e"},
    )
    di_e = Trigger(
        # CaloIdL_TrackIdL_IsoVL only for the second leg?
        name="HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
        id=202,
        legs=[
            TriggerLeg(
                pdg_id=11,
                min_pt=23.0,
                # filter names:
                # TODO
                trigger_bits=2**4 + 2**0,  # 2e (bit 4) + CaloIdL_TrackIdL_IsoVL (bit 0)
            ),
            TriggerLeg(
                pdg_id=11,
                min_pt=12.0,
                # filter names:
                # TODO
                trigger_bits=2**4 + 2**0,  # 2e (bit 4) + CaloIdL_TrackIdL_IsoVL (bit 0)
            ),
        ],
        aux={
            "channels": ["ee"],
            "data_stream": "data_egamma" if config.x.run == 3 else "data_e",
        },
        tags={"di_trigger", "di_e"},
    )
    single_e50_noniso = Trigger(
        name="HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165",
        id=203,
        legs=[
            TriggerLeg(
                pdg_id=11,
                min_pt=50.0,
                # filter names: TODO
                trigger_bits=2**11 + 2**12,  # CaloIdVT_GsfTrkIdT (bit 11) + PFJet (bit 12)
            ),
        ],
        aux={
            "channels": ["e", "ee", "emu", "mue", "mixed"],
            "data_stream": "data_egamma" if config.x.run == 3 else "data_e",
        },
    )
    di_e33_noniso = Trigger(
        name="HLT_DoubleEle33_CaloIdL_MW",
        id=204,
        legs=[
            TriggerLeg(
                pdg_id=11,
                min_pt=33.0,
                # filter names: TODO
                trigger_bits=2**4,  # 2e (bit 4) + CaloIdL_MW (no bit?)
            ),
            TriggerLeg(
                pdg_id=11,
                min_pt=33.0,
                # filter names: TODO
                trigger_bits=2**4,  # 2e (bit 4) + CaloIdL_MW (no bit?)
            ),
        ],
        aux={
            "channels": ["ee"],
            "data_stream": "data_egamma" if config.x.run == 3 else "data_e",
        },
    )
    mixed_mue = Trigger(
        name="HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
        id=301,
        legs=[
            TriggerLeg(
                pdg_id=13,
                min_pt=23.0,
                # filter names:
                # TODO
                trigger_bits=2**5 + 2**0,  # 1e-1mu (bit 5) + TrkIsoVVL (bit 0)
            ),
            TriggerLeg(
                pdg_id=11,
                min_pt=12.0,
                # filter names:
                # TODO
                trigger_bits=2**5 + 2**0,  # 1mu-1e (bit 5) + CaloIdL_TrackIdL_IsoVL (bit 0)
            ),
        ],
        aux={
            "channels": ["mue", "mixed"],
            "data_stream": "data_muoneg",
        },
        tags={"mixed_trigger", "mixed_mue"},
    )
    mixed_emu = Trigger(
        name="HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        id=401,
        legs=[
            TriggerLeg(
                pdg_id=13,
                min_pt=8.0,
                # filter names:
                # TODO
                trigger_bits=2**5 + 2**0,  # 1mu-1e (bit 5) + TrkIsoVVL (bit 0)
            ),
            TriggerLeg(
                pdg_id=11,
                min_pt=23.0,
                # filter names:
                # TODO
                trigger_bits=2**5 + 2**0,  # 1mu-1e (bit 5) + CaloIdL_TrackIdL_IsoVL (bit 0)
            ),
        ],
        aux={
            "channels": ["emu", "mixed"],
            "data_stream": "data_muoneg",
        },
        tags={"mixed_trigger", "mixed_emu"},
    )

    # add triggers to the config
    if config.has_tag("is_dl"):
        config.x.triggers = od.UniqueObjectIndex(Trigger, [
            single_e,
            single_e50_noniso,
            single_mu,
            di_e,
            di_e33_noniso,
            di_mu,
            mixed_mue,
            mixed_emu,
        ])
    elif config.has_tag("is_sl"):
        config.x.triggers = od.UniqueObjectIndex(Trigger, [
            single_e,
            single_mu,
        ])
    else:
        raise ValueError("Analysis, please set the 'is_dl' or 'is_sl' tag")
