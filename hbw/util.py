# coding: utf-8

"""
Collection of helpers
"""

from __future__ import annotations

import tracemalloc

import law

logger = law.logger.get_logger(__name__)


def log_memory(message: str = "", unit: str = "MB", restart: bool = False):
    if restart or not tracemalloc.is_tracing():
        logger.info("Start tracing memory")
        tracemalloc.start()

    unit_transform = {
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
    }[unit]

    current, peak = [x / unit_transform for x in tracemalloc.get_traced_memory()]
    logger.info(f"Memory after {message}: {current:.3f}{unit} (peak: {peak:.3f}{unit})")


def memory_GB(y_true, y_pred):
    """
    Memory tracing as a keras metric
    """
    if not tracemalloc.is_tracing():
        tracemalloc.start()

    return tracemalloc.get_traced_memory()[0] / 1024 ** 3
