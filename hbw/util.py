# coding: utf-8

"""
Collection of helpers
"""

from __future__ import annotations


import time
from typing import Hashable, Iterable, Callable
from functools import wraps
import tracemalloc

import law

from columnflow.util import maybe_import

np = maybe_import("numpy")

_logger = law.logger.get_logger(__name__)


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


def log_memory(
    message: str = "",
    unit: str = "MB",
    restart: bool = False,
    logger=None,
):
    if logger is None:
        logger = _logger

    if restart or not tracemalloc.is_tracing():
        logger.info("Start tracing memory")
        tracemalloc.start()

    unit_transform = {
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
    }[unit]

    current, peak = [x / unit_transform for x in tracemalloc.get_traced_memory()]
    logger.info(f"Memory after {message}: {current:.3f}{unit} (peak: {peak:.3f}{unit})")


def debugger():
    """
    Small helper to give some more information when starting IPython.embed
    """

    # get the previous frame to get some infos
    frame = __import__("inspect").currentframe().f_back

    header = (
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


def make_dict_hashable(d: dict, deep: bool = True):
    """ small helper that converts dict into hashable dict"""
    d_out = d.copy()
    for key, value in d.items():
        if isinstance(value, Hashable):
            # skip values that are already hashable
            continue
        elif isinstance(value, dict):
            # convert dictionary items to hashable and use items of resulting dict
            if deep:
                value = make_dict_hashable(value)
            d_out[key] = tuple(value)
        else:
            # hopefully, everything else can be cast to a tuple
            d_out[key] = law.util.make_tuple(value)

    return d_out.items()


def dict_diff(dict1: dict, dict2: dict):
    set1 = set(make_dict_hashable(dict1))
    set2 = set(make_dict_hashable(dict2))

    return set1 ^ set2


def four_vec(
    collections: str | Iterable[str],
    columns: str | Iterable[str] | None = None,
    skip_defaults: bool = False,
) -> set[str]:
    """
    Helper to quickly get a set of 4-vector component string for all collections in *collections*.
    Additional columns can be added wih the optional *columns* parameter.

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


def call_once_on_config(include_hash=False):
    """
    Parametrized decorator to ensure that function *func* is only called once for the config *config*
    """
    def outer(func):
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
    return outer


def timeit(func):
    """ Simple wrapper to measure execution time of a function """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        _logger.info(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result
    return timeit_wrapper


def timeit_multiple(func):
    """ Wrapper to measure the number of execution calls and the added execution time of a function """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        func.total_calls = getattr(func, "total_calls", 0) + 1
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        func.total_time = getattr(func, "total_time", 0) + total_time
        _logger.info(f"{func.__name__} has been run {func.total_calls} times ({func.total_time:.4f} seconds)")
        return result
    return timeit_wrapper
