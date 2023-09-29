# coding: utf-8

"""
unittests for hbw.util
"""

import unittest

from columnflow.util import maybe_import

from hbw.util import dict_diff, four_vec, call_once_on_config

import order as od

np = maybe_import("numpy")
ak = maybe_import("awkward")


class HbwUtilTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config_inst = od.Config(name="test_config", id=123456)

    def test_dict_diff(self):
        d1 = {"A": 1, "B": 2}
        d2 = {"A": 1, "B": 3, "C": 4}
        diff = dict_diff(d1, d2)

        self.assertEqual(diff, {("B", 2), ("B", 3), ("C", 4)})

    def test_four_vec(self):
        self.assertEqual(four_vec("Jet"), {"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass"})
        self.assertEqual(four_vec("Jet", "px", skip_defaults=True), {"Jet.px"})
        self.assertEqual(
            four_vec(["Electron", "Muon"], ["dxy", "dz"]),
            {
                "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass", "Electron.dxy", "Electron.dz",
                "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass", "Muon.dxy", "Muon.dz",
            },
        )
        self.assertEqual(four_vec("MET"), {"MET.pt", "MET.phi"})

    def test_call_once_on_config(self):
        @call_once_on_config()
        def some_config_function(config: od.Config) -> str:
            # do something with config
            config.add_variable(name="dummy_variable")

            return "test_string"

        # on first call, function is called -> returns "test_string" and adds identifier tag
        self.assertEqual(some_config_function(self.config_inst), "test_string")
        self.assertTrue(self.config_inst.has_tag("some_config_function_called"))

        # on second call, function should not be called -> returns None
        self.assertEqual(some_config_function(self.config_inst), None)
