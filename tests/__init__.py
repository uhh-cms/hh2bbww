# coding: utf-8

"""
Test suite of the hbw analysis, run via pytest (see tests/README.md).
"""

import os
import sys

# adjust the path to import the package
base = os.path.normpath(os.path.join(os.path.abspath(__file__), "../.."))
if base not in sys.path:
    sys.path.insert(0, base)
