# coding: utf-8

"""
Collection of helpers
"""

from __future__ import annotations

from typing import Hashable

import tracemalloc

import law

_logger = law.logger.get_logger(__name__)


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
