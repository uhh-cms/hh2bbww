# coding: utf-8

"""
Helpers shared by the hbw tests: toy order objects, synthetic events and standalone calls of
TaskArrayFunctions (producers, selectors, categorizers) outside of law tasks.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import law
import order as od

from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")


def ensure_campaign_summary(config_name: str) -> None:
    """
    Makes sure that the pickled campaign required by the lazy config factory of *config_name*
    exists (see ``hbw.analysis.create_analysis``), building it when missing. This only needs
    cmsdb, so it also works outside of the NAF.
    """
    from hbw.tasks.campaigns import BuildCampaignSummary

    # limited configs ("l...") share the campaign of their full twin ("c...")
    task = BuildCampaignSummary(config=config_name.replace("l", "c", 1))
    if task.complete():
        return

    for target in law.util.flatten(task.output()):
        target.parent.touch()
    task.run()


def load_analysis(analysis_name: str) -> od.Analysis:
    """
    Returns the analysis *analysis_name*, either the name of one of the hbw analyses (e.g. "hbw_dl")
    or the full path to an analysis instance (e.g. "tests.analyses.hbw_dl_ml").
    """
    if "." not in analysis_name:
        analysis_name = f"hbw.analysis.{analysis_name}.{analysis_name}"
    module_name, attr = analysis_name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr)


def load_config(analysis_name: str, config_name: str) -> od.Config:
    """
    Returns the config *config_name* of the analysis *analysis_name* (see :py:func:`load_analysis`).
    Prefer the session fixture ``get_config``, which caches configs.
    """
    analysis_inst = load_analysis(analysis_name)
    ensure_campaign_summary(config_name)
    return analysis_inst.get_config(config_name)


def toy_config(
    tags: set[str] | tuple[str] = ("is_dl", "is_hh", "is_nonresonant"),
    campaign_name: str = "run3_2024_nano_v15",
    year: int = 2024,
    run: int = 3,
    **aux,
) -> od.Config:
    """
    Creates a minimal :py:class:`od.Analysis` with a single :py:class:`od.Config` that contains a
    nominal shift, one MC process + dataset ("toy_mc") and one data process + dataset ("toy_data").
    Additional *aux* is set on the config, e.g. ``met_name="PuppiMET"``.
    """
    analysis_inst = od.Analysis("toy_analysis", 1)
    campaign_inst = od.Campaign(campaign_name, 1, ecm=13.6, aux={"year": year, "run": run})
    config_inst = analysis_inst.add_config(campaign_inst, name="toy_config", id=1, tags=set(tags))
    config_inst.add_shift(name="nominal", id=0)
    # also stored in the config aux by add_config
    config_inst.x.run = run
    config_inst.x.year = year

    mc_proc = config_inst.add_process(name="toy_proc", id=1, xsecs={13.6: 1.0})
    data_proc = config_inst.add_process(name="data", id=2, is_data=True)
    config_inst.add_dataset(name="toy_mc", id=1, processes=[mc_proc], keys=["/toy/mc/NANOAODSIM"])
    config_inst.add_dataset(name="toy_data", id=2, processes=[data_proc], keys=["/toy/data/NANOAOD"], is_data=True)

    for key, value in aux.items():
        config_inst.set_aux(key, value)

    return config_inst


def make_events(fields: dict, behavior: bool = False) -> ak.Array:
    """
    Creates an awkward array of events from a dict of (nested) fields, e.g.

    .. code-block:: python

        make_events({"Jet": {"pt": [[50.0, 30.0], []], "eta": [[0.1, 1.0], []]}})

    When *behavior* is *True*, the coffea nanoevents behavior is attached so that four-vector
    methods (``.mass``, ``+``, ``.delta_r``, ...) are available. This requires each collection to
    contain at least one object in total, so that its fields are known.
    """
    events = ak.Array(fields)
    if behavior:
        from columnflow.columnar_util import attach_coffea_behavior, default_coffea_collections
        collections = [name for name in default_coffea_collections if name in events.fields]
        empty = [name for name in collections if not events[name].fields]
        if empty:
            raise ValueError(f"cannot attach behavior to collections without any object: {empty}")
        events = attach_coffea_behavior(events, collections=collections)
    return events


def task_stub(config_inst: od.Config, dataset_inst: od.Dataset, **kwargs) -> SimpleNamespace:
    """
    Minimal stand-in for the *task* argument that TaskArrayFunction hooks receive.
    """
    attrs = {
        "analysis_inst": config_inst.analysis,
        "config_inst": config_inst,
        "dataset_inst": dataset_inst,
        "shift": "nominal",
        "local_shift": "nominal",
        "global_shift_inst": config_inst.get_shift("nominal"),
        "local_shift_inst": config_inst.get_shift("nominal"),
    }
    attrs.update(kwargs)
    return SimpleNamespace(**attrs)


def build_taf(taf_cls, config_inst: od.Config, dataset: str | od.Dataset = "toy_mc"):
    """
    Instantiates the TaskArrayFunction class *taf_cls* for *config_inst* and *dataset* the same way
    tasks do it (see ``get_array_function_dict`` in columnflow's task base).
    """
    dataset_inst = config_inst.get_dataset(dataset) if isinstance(dataset, str) else dataset
    return taf_cls(inst_dict={
        "analysis_inst": config_inst.analysis,
        "config_inst": config_inst,
        "config_insts": [config_inst],
        "dataset_inst": dataset_inst,
    })


def run_taf(
    taf_cls,
    events: ak.Array,
    config_inst: od.Config,
    dataset: str | od.Dataset = "toy_mc",
    setup: bool = False,
    task=None,
    **kwargs,
):
    """
    Builds *taf_cls* via :py:func:`build_taf` and calls it on *events*. When *setup* is *True*, the
    setup hooks are run first (only possible when they do not need external inputs). Additional
    *kwargs* are forwarded to the call, e.g. ``stats={}`` for selectors.

    Returns whatever the TaskArrayFunction returns: *events* for producers, *(events, results)*
    for selectors and *(events, mask)* for categorizers.
    """
    inst = build_taf(taf_cls, config_inst, dataset)
    if task is None:
        task = task_stub(config_inst, inst.dataset_inst)
    if setup:
        inst.run_setup(task=task)
    return inst(events, task=task, **kwargs)


def models_per_module(
    base_cls,
    modules: dict[str, tuple[str, str]],
    exclude: set[str] | None = None,
) -> dict[str, tuple[str, str]]:
    """
    Maps the names of all registered subclasses of *base_cls* (MLModels or InferenceModels) to the
    (analysis, config) pair given for the module in *modules* whose namespace contains the class.
    Modules are checked in the given order, so classes imported by later modules keep the first one.
    Names in *exclude* (e.g. abstract base classes) are skipped.

    The module namespace is used since classes created via ``derive`` do not reliably carry the
    module they were derived in.
    """
    from hbw.util import get_subclasses_deep

    registered = get_subclasses_deep(base_cls)
    exclude = {base_cls.__name__} | set(exclude or ())
    models = {}
    for module_name, case in modules.items():
        module = importlib.import_module(module_name)
        for obj in vars(module).values():
            name = getattr(obj, "__name__", None)
            if name in registered and registered[name] is obj and name not in exclude:
                models.setdefault(name, case)
    return dict(sorted(models.items()))


def to_list(array) -> list:
    """
    Converts awkward or numpy arrays to (nested) python lists for readable assertions.
    """
    if isinstance(array, ak.Array):
        return array.to_list()
    if isinstance(array, np.ndarray):
        return array.tolist()
    return law.util.make_list(array)
