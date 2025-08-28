# coding: utf-8

"""
Collection of helpers
"""

from __future__ import annotations

# NOTE: needs to be added to cf sandbox
import os
import re
import itertools
import time
from typing import Hashable, Iterable, Callable
from functools import wraps, reduce
import tracemalloc

import law
import order as od

from columnflow.types import Any
from columnflow.columnar_util import ArrayFunction, deferred_column, get_ak_routes
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")
psutil = maybe_import("psutil")

_logger = law.logger.get_logger(__name__)


def nevents_expected(
    config_inst: od.Config,
    processes: list = ["hh_ggf_hbb_hww2l2nu_kl1_kt1", "hh_vbf_hbb_hww2l2nu_kv1_k2v1_kl1"],
    lumis: dict = {
        # 13: 36310 + 41480 + 59830,
        13.6: 7980.4 + 26671.7 + 17794 + 9451,
    },
):
    nevents = 0
    for com, lumi in lumis.items():
        for process in law.util.make_list(processes):
            xs = config_inst.get_process(process).xsecs[com].nominal
            nevents += xs * lumi
    return nevents


def ak_any(masks: list[ak.Array], **kwargs) -> ak.Array:
    """
    Apparently, ak.any is very slow, so just do the "or" of all masks in a loop.
    This is more than 100x faster than doing `ak.any(masks, axis=0)`.

    param masks: list of masks to be combined via logical "or"
    return: ak.Array of logical "or" of all masks
    """
    if not masks:
        return False
    return reduce(lambda a, b: a | b, masks)


def ak_all(masks: list[ak.Array], **kwargs) -> ak.Array:
    """
    Apparently, ak.all is very slow, so just do the "and" of all masks in a loop.
    This is more than 100x faster than doing `ak.all(masks, axis=0)`.

    param masks: list of masks to be combined via logical "and"
    return: ak.Array of logical "and" of all masks
    """
    if not masks:
        return False
    return reduce(lambda a, b: a & b, masks)


def has_tag(tag, *container, operator: callable = any) -> bool:
    """
    Helper to check multiple container for a certain tag *tag*.
    Per default, booleans are combined with logical "or"

    :param tag: String of which tag to look for.
    :param container: Instances to check for tags.
    :param operator: Callable on how to combine tag existance values.
    :return: Boolean whether any (all) containter contains the requested tag.
    """
    values = [inst.has_tag(tag) for inst in container]
    return operator(values)


def print_law_config(sections: list | None = None, keys: list | None = None):
    """ Helper to print the currently used law config """
    law_sections = law.config.sections()
    for section in sections:
        if section not in law_sections:
            continue
        print(f"{'=' * 20} [{section}] {'=' * 20}")
        section_keys = law.config.options(section)
        for key in section_keys:
            if key in keys:
                value = law.config.get_expanded(section, key)
                print(f"{key}: {value}")


def get_subclasses_deep(*classes):
    """
    Helper that gets all subclasses from input classes based on the '_subclasses' attribute.
    """
    classes = {_cls.__name__: _cls for _cls in classes}
    all_classes = {}

    while classes:
        for key, _cls in classes.copy().items():
            classes.update(getattr(_cls, "_subclasses", {}))
            all_classes[key] = classes.pop(key)

    return all_classes


def build_param_product(params: dict[str, list], output_keys: Callable = lambda i: i):
    """
    Helper that builds the product of all *param* values and returns a dictionary of
    all the resulting parameter combinations.

    Example:

    .. code-block:: python
        build_param_product({"A": ["a", "b"], "B": [1, 2]})
        # -> {
            0: {"A": "a", "B": 1},
            1: {"A": "a", "B": 2},
            2: {"A": "b", "B": 1},
            3: {"A": "b", "B": 2},
        }
    """
    from itertools import product
    param_product = {}
    keys, values = zip(*params.items())
    for i, bundle in enumerate(product(*values)):
        d = dict(zip(keys, bundle))
        param_product[output_keys(i)] = d

    return param_product


def round_sig(
    value: int | float | np.number,
    sig: int = 4,
    convert: Callable | None = None,
) -> int | float | np.number:
    """
    Helper function to round number *value* on *sig* significant digits and
    optionally transform output to type *convert*
    """
    if not np.isfinite(value):
        # cannot round infinite
        _logger.warning("cannot round infinite number")
        return value

    from math import floor, log10

    def try_rounding(_value):
        try:
            n_digits = sig - int(floor(log10(abs(_value)))) - 1
            if convert in (int, np.int8, np.int16, np.int32, np.int64):
                # do not round on decimals when converting to integer
                n_digits = min(n_digits, 0)
            return round(_value, n_digits)
        except Exception:
            _logger.warning(f"Cannot round number {value} to {sig} significant digits. Number will not be rounded")
            return value

    # round first to not lose information from type conversion
    rounded_value = try_rounding(value)

    # convert number if "convert" is given
    if convert not in (None, False):
        try:
            rounded_value = convert(rounded_value)
        except Exception:
            _logger.warning(f"Cannot convert {rounded_value} to {convert.__name__}")
            return rounded_value

        # some types need rounding again after converting (e.g. np.float32 to float)
        rounded_value = try_rounding(rounded_value)

    return rounded_value


def get_memory_usage():
    """Get current memory usage in MB"""
    if not psutil:
        return "? (psutil not available)"
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory(
    message: str = "",
    unit: str = "MB",
    restart: bool = False,
    logger=None,
    prefer_psutil: bool = True,
):
    if logger is None:
        logger = _logger

    if psutil:
        current_memory = get_memory_usage()
        if unit == "GB":
            current_memory /= 1024
        logger.info(f"Memory after {message}: {current_memory:.3f}{unit}")
        return

    if restart or not tracemalloc.is_tracing():
        logger.info("Start tracing memory")
        tracemalloc.start()

    unit_transform = {
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
    }[unit]

    current, peak = [x / unit_transform for x in tracemalloc.get_traced_memory()]
    logger.info(f"Memory after {message}: {current:.3f}{unit} (peak: {peak:.3f}{unit})")


def debugger(msg: str = ""):
    """
    Small helper to give some more information when starting IPython.embed

    :param msg: Message to be printed when starting the debugger
    """

    # get the previous frame to get some infos
    frame = __import__("inspect").currentframe().f_back

    header = (
        f"{msg}\n"
        f"Line: {frame.f_lineno}, Function: {frame.f_code.co_name}, "
        f"File: {frame.f_code.co_filename}"
    )

    # get the namespace of the previous frame
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)

    # add helper function to namespace
    namespace.update({
        "q": __import__("functools").partial(__import__("os")._exit, 0),
    })

    # start the debugger
    __import__("IPython").embed(header=header, user_ns=namespace)


def traceback_function(depth: int = 1):
    """
    Helper function to trace back function call by up to *depth* frames.
    """
    frame = __import__("inspect").currentframe().f_back
    logger = law.logger.get_logger(frame.f_code.co_name)
    logger.info("starting traceback")
    for i in range(depth + 1):
        if not frame:
            logger.info("max depth reached")
            return
        logger = law.logger.get_logger(f"{frame.f_code.co_name} (depth {i})")
        logger.info(f"Line: {frame.f_lineno}, File: {frame.f_code.co_filename}")
        frame = frame.f_back


def make_dict_hashable(d: dict, deep: bool = True):
    """Small helper that converts dict into a hashable representation."""
    d_out = d.copy()
    for key, value in d.items():
        if isinstance(value, Hashable):
            # Skip values that are already hashable
            continue
        elif isinstance(value, dict):
            # Convert nested dictionaries to a hashable form
            if deep:
                value = make_dict_hashable(value)
            d_out[key] = tuple(value)
        else:
            # Convert other types to tuples
            d_out[key] = law.util.make_tuple(value)

    return d_out.items()


def dict_diff(dict1: dict, dict2: dict):
    """Return the differences between two dictionaries."""
    set1 = set(make_dict_hashable(dict1))
    set2 = set(make_dict_hashable(dict2))

    return set1 ^ set2


def filter_unchanged_keys(d1: dict, d2: dict):
    """Recursively remove unchanged keys from nested dictionaries and return modified values."""
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        return {"old": d1, "new": d2} if d1 != d2 else None

    filtered = {}
    all_keys = set(d1.keys()).union(set(d2.keys()))

    for key in all_keys:
        val1 = d1.get(key)
        val2 = d2.get(key)

        if isinstance(val1, dict) and isinstance(val2, dict):
            # Recur for nested dictionaries
            nested_diff = filter_unchanged_keys(val1, val2)
            if nested_diff:
                filtered[key] = nested_diff
        elif val1 != val2:
            # Value changed or key added/removed
            filtered[key] = {"old": val1, "new": val2}

    return filtered if filtered else None


def dict_diff_filtered(old_dict: dict, new_dict: dict):
    """Return the differences between two dictionaries with nested filtering of unchanged keys."""
    diff = {}

    # Check keys present in either dict
    all_keys = set(old_dict.keys()).union(set(new_dict.keys()))

    for key in all_keys:
        if key in old_dict and key in new_dict:
            if isinstance(old_dict[key], dict) and isinstance(new_dict[key], dict):
                # Recur for nested dictionaries and get filtered diff
                nested_diff = filter_unchanged_keys(old_dict[key], new_dict[key])
                if nested_diff:
                    diff[key] = nested_diff
            elif old_dict[key] != new_dict[key]:
                diff[key] = {"old": old_dict[key], "new": new_dict[key]}
        elif key in old_dict:
            diff[key] = {"old": old_dict[key], "new": None}
        else:
            diff[key] = {"old": None, "new": new_dict[key]}

    return diff


def gather_dict_diff(old_dict: dict, new_dict: dict) -> str:
    """Gather the differences between two dictionaries and return them as a formatted string."""
    diff = filter_unchanged_keys(old_dict, new_dict)
    lines = []

    if not diff:
        return "✅ No differences found."

    def process_diff(diff, indent=0):
        indentation = "    " * indent
        for key, value in diff.items():
            if isinstance(value, dict) and "old" in value and "new" in value:
                if value["old"] is None:
                    lines.append(f"{indentation}🔹 Added: {key}: {value['new']}")
                elif value["new"] is None:
                    lines.append(f"{indentation}🔻 Removed: {key}: {value['old']}")
                else:
                    lines.append(f"{indentation}🔄 Modified: {key}:")
                    lines.append(f"{indentation}    - Old: {value['old']}")
                    lines.append(f"{indentation}    - New: {value['new']}")
            elif isinstance(value, dict):
                lines.append(f"{indentation}🔄 Modified: {key}:")
                process_diff(value, indent + 1)

    process_diff(diff)
    return "\n".join(lines)


def four_vec(
    collections: str | Iterable[str],
    columns: str | Iterable[str] | None = None,
    skip_defaults: bool = False,
) -> set[str]:
    """
    Helper to quickly get a set of 4-vector component string for all collections in *collections*.
    Additional columns can be added wih the optional *columns* parameter.

    TODO: this function is not really needed anymore, since we can just pass
    uses={Jet.{pt,eta,phi,mass}} instead, so we should deprecate this function.

    Example:

    .. code-block:: python

    four_vec("Jet", "jetId")
    # -> {"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.jetId"}

    four_vec({"Electron", "Muon"})
    # -> {
            "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
            "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        }
    """
    # make sure *collections* is a set
    collections = law.util.make_set(collections)

    # transform *columns* to a set and add the default 4-vector components
    columns = law.util.make_set(columns) if columns else set()
    default_columns = {"pt", "eta", "phi", "mass"}
    if not skip_defaults:
        columns |= default_columns

    outp = set(
        f"{obj}.{var}"
        for obj in collections
        for var in columns
    )

    # manually remove MET eta and mass
    outp = outp.difference({"MET.eta", "MET.mass"})

    return outp


def bracket_expansion(inputs: list, clean_chars: str = "_") -> list:
    """
    Expands a list of strings with bracket notation into all possible combinations.

    Example:
    bracket_expansion(["{Jet,Muon}.{pt,eta}", "{Electron,Photon}.{phi}"]) -->
    {"Jet.pt", "Jet.eta", "Muon.pt", "Muon.eta", "Electron.phi", "Photon.phi"}

    :param inputs: list of strings with bracket notation
    :param clean_chars: characters to remove from the beginning and end of each string (default is "_")
    :return list: expanded and sorted list of strings

    NOTE: similar implementation might be somewhere in columnflow.
    """
    pattern = re.compile(r"\{([^{}]+)\}")
    outp = set()
    inputs = law.util.make_list(inputs)
    for inp in inputs:
        # Find all bracketed groups and extract options by splitting on ","
        matches = pattern.findall(inp)
        options = [match.split(",") for match in matches]

        # Replace each bracketed group with a placeholder "{}"
        template = pattern.sub("{}", inp)

        # Generate all possible combinations and add to the output set
        combinations = itertools.product(*options)
        expanded = (template.format(*combo) for combo in combinations)

        # Remove specified leading/trailing characters
        cleaned = (s.strip(clean_chars) for s in expanded)
        outp.update(cleaned)

    # remove empty string
    outp.discard("")

    return sorted(outp)


def has_four_vec(
    events: ak.Array,
    collection_name: str,
):
    """
    Check if the collection has the fields required for a 4-vector.
    """
    four_vec_cols = {"pt", "eta", "phi", "mass"}
    return collection_name in events.fields and four_vec_cols.issubset(events[collection_name].fields)


def call_once_on_config(func=None, *, include_hash=False):
    """
    Parametrized decorator to ensure that function *func* is only called once for the config *config*.
    Can be used with or without parentheses.
    """
    if func is None:
        # If func is None, it means the decorator was called with arguments.
        def wrapper(f):
            return call_once_on_config(f, include_hash=include_hash)
        return wrapper

    @wraps(func)
    def inner(config, *args, **kwargs):
        tag = f"{func.__name__}_called"
        if include_hash:
            tag += f"_{func.__hash__()}"

        if config.has_tag(tag):
            return

        config.add_tag(tag)
        return func(config, *args, **kwargs)

    return inner


def timeit(func):
    """
    Simple wrapper to measure execution time of a function.
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        _logger.info(f"Function '{func.__name__}' done; took {round_sig(total_time)} seconds")
        return result
    return timeit_wrapper


def timeit_multiple(func):
    """ Wrapper to measure the number of execution calls and the added execution time of a function """
    log_method = "info"
    log_func = getattr(_logger, log_method)

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        func.total_calls = getattr(func, "total_calls", 0) + 1
        _repr = func.__name__

        if len(args) >= 1 and hasattr(args[0], "__name__"):
            # some classmethod
            _repr = f"{args[0].__name__}.{_repr}"

            if len(args) >= 2 and isinstance(args[1], dict):
                params = args[1]
            elif len(args) >= 3 and isinstance(args[2], dict):
                params = args[2]
            else:
                params = {}

            for param in ("branch", "dataset"):
                if param in params:
                    _repr = f"{_repr} ({param} {params[param]})"

        elif len(args) >= 1 and hasattr(args[0], "cls_name"):
            # probably a CSP function
            inst = args[0]
            params = {}
            _repr = f"{inst.cls_name}.{_repr}"
            if hasattr(inst, "config_inst"):
                _repr = f"{_repr} ({inst.config_inst.name})"
            if hasattr(inst, "dataset_inst"):
                _repr = f"{_repr} ({inst.dataset_inst.name})"
            if hasattr(inst, "shift_inst"):
                _repr = f"{_repr} ({inst.shift_inst.name})"

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        func.total_time = getattr(func, "total_time", 0) + total_time
        log_func(f"{_repr} has been run {func.total_calls} times ({round_sig(func.total_time)} seconds)")
        return result

    return timeit_wrapper


def timeit_multiple_plain(func):
    """ Wrapper to measure the number of execution calls and the added execution time of a function """
    log_method = "info"
    log_func = getattr(_logger, log_method)

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        func.total_calls = getattr(func, "total_calls", 0) + 1
        _repr = func.__name__
        if len(args) >= 1 and hasattr(args[0], "__name__"):
            _repr = f"{args[0].__name__}.{_repr}"

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        func.total_time = getattr(func, "total_time", 0) + total_time
        log_func(f"{_repr} has been run {func.total_calls} times ({round_sig(func.total_time)} seconds)")
        return result

    return timeit_wrapper


def print_array_sums(events: ak.Array) -> None:
    """
    Helper to print the sum of all (nested) columns in an awkward array
    """
    routes = get_ak_routes(events)
    for route in routes:
        print(route.string_column, ak.sum(route.apply(events)))


def call_func_safe(func, *args, **kwargs) -> Any:
    """
    Small helper to make sure that our training does not fail due to plotting
    """

    # get the function name without the possibility of raising an error
    try:
        func_name = func.__name__
    except Exception:
        # default to empty name
        func_name = ""

    t0 = time.perf_counter()

    try:
        outp = func(*args, **kwargs)
        _logger.info(f"Function '{func_name}' done; took {(time.perf_counter() - t0):.2f} seconds")
    except Exception as e:
        _logger.warning(f"Function '{func_name}' failed due to {type(e)}: {e}")
        outp = None

    return outp


@deferred_column
def IF_NANO_V12(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    """
    Helper to check if the campaign of this particular dataset is nano v12.
    """
    cpn_name = func.dataset_inst.x("campaign", func.config_inst.campaign.name)
    version = int(cpn_name.split("v")[-1])
    return self.get() if version == 12 else None


@deferred_column
def IF_NANO_V14(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    """
    Helper to check if the campaign of this particular dataset is nano v14.
    """
    cpn_name = func.dataset_inst.x("campaign", func.config_inst.campaign.name)
    version = int(cpn_name.split("v")[-1])
    return self.get() if version == 14 else None


@deferred_column
def IF_NANO_geV13(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    """
    Helper to check if the campaign of this particular dataset is nano v13 or higher.
    """
    cpn_name = func.dataset_inst.x("campaign", func.config_inst.campaign.name)
    version = int(cpn_name.split("v")[-1])
    return self.get() if version >= 13 else None


@deferred_column
def IF_RUN_2(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.run == 2 else None


@deferred_column
def IF_RUN_3(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.run == 3 else None


@deferred_column
def IF_SL(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.has_tag("is_sl") else None
    # return self.get() if func.config_inst.x.lepton_tag == "sl" else None


@deferred_column
def IF_DL(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.has_tag("is_dl") else None
    # return self.get() if func.config_inst.x.lepton_tag == "dl" else None


@deferred_column
def BTAG_COLUMN(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    """
    This helper allows adding the correct btag column based on the b_tagger configuration.
    Requires the b_tagger aux to be set in the config. Example usecase:

    .. code-block:: python

        @producer(uses={BTAG_COLUMN("Jet")})
        def my_producer(self, events):
            btag_score = events.Jet[self.config_inst.x.btag_column]
            ...
            return events
    """
    btag_column = func.config_inst.x("btag_column", None)
    if not btag_column:
        raise Exception("the btag_column has not been configured")
    return f"{self.get()}.{btag_column}"


@deferred_column
def MET_COLUMN(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    """
    This helper allows adding the correct btag column based on the b_tagger configuration.
    Requires the b_tagger aux to be set in the config. Example usecase:

    .. code-block:: python

        @producer(uses={MET_COLUMN("pt")})
        def my_producer(self, events):
            met_pt = events[self.config_inst.x.met_name].pt
            ...
            return events
    """
    met_name = func.config_inst.x("met_name", None)
    if not met_name:
        raise Exception("the met_name has not been configured")
    return f"{met_name}.{self.get()}"


@deferred_column
def RAW_MET_COLUMN(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    """
    Similar to MET_COLUMN, see MET_COLUMN for more information.
    """
    raw_met_name = func.config_inst.x("raw_met_name", None)
    if not raw_met_name:
        raise Exception("the raw_met_name has not been configured")
    return f"{raw_met_name}.{self.get()}"


@deferred_column
def IF_DATASET_HAS_LHE_WEIGHTS(
    self: ArrayFunction.DeferredColumn,
    func: ArrayFunction,
) -> Any | set[Any]:
    if getattr(func, "dataset_inst", None) is None:
        return self.get()

    return None if func.dataset_inst.has_tag("no_lhe_weights") else self.get()


@deferred_column
def IF_MC(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if getattr(func, "dataset_inst", None) is None:
        return self.get()

    return self.get() if func.dataset_inst.is_mc else None


@deferred_column
def IF_DATA(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if getattr(func, "dataset_inst", None) is None:
        return self.get()

    return self.get() if func.dataset_inst.is_data else None


@deferred_column
def IF_VJETS(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if getattr(func, "dataset_inst", None) is None:
        return self.get()

    return self.get() if func.dataset_inst.has_tag("is_v_jets") else None


@deferred_column
def IF_DY(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if getattr(func, "dataset_inst", None) is None:
        return self.get()

    return self.get() if func.dataset_inst.has_tag("is_dy") else None


@deferred_column
def IF_TOP(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if getattr(func, "dataset_inst", None) is None:
        return self.get()

    return self.get() if func.dataset_inst.has_tag("has_top") else None


@deferred_column
def IF_TT(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if getattr(func, "dataset_inst", None) is None:
        return self.get()

    return self.get() if func.dataset_inst.has_tag("is_ttbar") else None


@deferred_column
def IF_HBV(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if getattr(func, "dataset_inst", None) is None:
        return self.get()

    return self.get() if func.dataset_inst.has_tag("is_hbv") else None
