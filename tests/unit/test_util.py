# coding: utf-8

"""
unittests for hbw.util
"""

import unittest

from columnflow.util import maybe_import

from columnflow.production import producer

from hbw.util import (
    build_param_product, round_sig, dict_diff, four_vec, call_once_on_config,
    ak_any, ak_all, has_tag, bracket_expansion, make_dict_hashable, filter_unchanged_keys,
    dict_diff_filtered, gather_dict_diff, has_four_vec, get_subclasses_deep, call_func_safe,
    IF_NANO_V14, IF_NANO_V15, IF_NANO_geV14, IF_RUN_2, IF_RUN_3, IF_SL, IF_DL, IF_HH, IF_HHH,
    IF_2024, IF_NOT_2024, BTAG_COLUMN, MET_COLUMN, IF_MC, IF_DATA, IF_DY,
)

import order as od

from tests.helpers import toy_config, build_taf

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={
        IF_NANO_V14("nano_v14"), IF_NANO_V15("nano_v15"), IF_NANO_geV14("nano_ge_v14"),
        IF_RUN_2("run2"), IF_RUN_3("run3"), IF_2024("y2024"), IF_NOT_2024("not_y2024"),
        IF_SL("sl"), IF_DL("dl"), IF_HH("hh"), IF_HHH("hhh"),
        IF_MC("mc"), IF_DATA("data"), IF_DY("dy"),
        BTAG_COLUMN("Jet"), MET_COLUMN("pt"),
    },
)
def _test_deferred_columns(self, events, **kwargs):
    return events


class HbwUtilTest(
    unittest.TestCase,
):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config_inst = od.Config(name="test_config", id=123456)

    def test_build_param_product(self):
        params = {
            "A": ["a", "b"],
            "B": [1, 2],
            "C": [{"x": 1}, {"x": 2}],
        }
        param_product = build_param_product(params)

        expected_param_product = {
            0: {"A": "a", "B": 1, "C": {"x": 1}},
            1: {"A": "a", "B": 1, "C": {"x": 2}},
            2: {"A": "a", "B": 2, "C": {"x": 1}},
            3: {"A": "a", "B": 2, "C": {"x": 2}},
            4: {"A": "b", "B": 1, "C": {"x": 1}},
            5: {"A": "b", "B": 1, "C": {"x": 2}},
            6: {"A": "b", "B": 2, "C": {"x": 1}},
            7: {"A": "b", "B": 2, "C": {"x": 2}},
        }
        self.assertEqual(param_product, expected_param_product)

    def test_round_sig(self):
        self.assertEqual(round_sig(0.15, 1), 0.1)
        self.assertEqual(round_sig(0.15001, 1), 0.2)

        number = 1.23456789
        self.assertEqual(round_sig(number, 3), 1.23)
        self.assertEqual(round_sig(number, 4), 1.235)
        self.assertEqual(round_sig(number, 4, np.float32), np.float32(1.235))

        number = 1234.567
        self.assertEqual(round_sig(number, 2), 1200)
        self.assertEqual(round_sig(number, 5), 1234.6)

        # ugly edge case when rounding and transforming to integer
        number = 12.9
        self.assertEqual(round_sig(number, 2, int), 13)
        self.assertEqual(round_sig(number, 3, int), 13)

        number = np.float32(1.23456789)
        self.assertEqual(round_sig(number, 4, float), 1.235)

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

    def test_ak_any_all(self):
        a = ak.Array([True, False, False, True])
        b = ak.Array([False, False, True, True])
        self.assertEqual(ak_any([a, b]).to_list(), [True, False, True, True])
        self.assertEqual(ak_all([a, b]).to_list(), [False, False, False, True])
        self.assertEqual(ak_any([a]).to_list(), a.to_list())
        # empty inputs fall back to False
        self.assertIs(ak_any([]), False)
        self.assertIs(ak_all([]), False)

    def test_has_tag(self):
        p1 = od.Process("p1", id=1, tags={"a", "b"})
        p2 = od.Process("p2", id=2, tags={"a"})
        self.assertTrue(has_tag("a", p1, p2, operator=all))
        self.assertTrue(has_tag("b", p1, p2))
        self.assertFalse(has_tag("b", p1, p2, operator=all))
        self.assertFalse(has_tag("c", p1, p2))

    def test_bracket_expansion(self):
        self.assertEqual(
            bracket_expansion(["{Jet,Muon}.{pt,eta}", "{Electron,Photon}.{phi}"]),
            ["Electron.phi", "Jet.eta", "Jet.pt", "Muon.eta", "Muon.pt", "Photon.phi"],
        )
        # leading/trailing clean characters are stripped and empty strings removed
        self.assertEqual(bracket_expansion("{,mll_}{incl,sr}"), ["incl", "mll_incl", "mll_sr", "sr"])
        self.assertEqual(bracket_expansion("{_,}"), [])
        # strings without brackets are passed through
        self.assertEqual(bracket_expansion("n_jet"), ["n_jet"])

    def test_make_dict_hashable(self):
        d = {"A": 1, "B": [1, 2], "C": {"x": [3]}}
        hashable = set(make_dict_hashable(d))
        self.assertIn(("A", 1), hashable)
        self.assertIn(("B", (1, 2)), hashable)
        # nested dicts are converted to tuples of their (hashable) items
        self.assertIn(("C", (("x", (3,)),)), hashable)

    def test_filter_unchanged_keys(self):
        old = {"A": 1, "B": {"x": 1, "y": 2}, "C": 3}
        new = {"A": 1, "B": {"x": 1, "y": 5}, "D": 4}
        self.assertEqual(filter_unchanged_keys(old, new), {
            "B": {"y": {"old": 2, "new": 5}},
            "C": {"old": 3, "new": None},
            "D": {"old": None, "new": 4},
        })
        self.assertIsNone(filter_unchanged_keys(old, old))
        self.assertEqual(filter_unchanged_keys(1, 2), {"old": 1, "new": 2})

    def test_dict_diff_filtered(self):
        old = {"A": 1, "B": {"x": 1, "y": 2}, "C": 3}
        new = {"A": 2, "B": {"x": 1, "y": 2}, "D": 4}
        self.assertEqual(dict_diff_filtered(old, new), {
            "A": {"old": 1, "new": 2},
            "C": {"old": 3, "new": None},
            "D": {"old": None, "new": 4},
        })

    def test_gather_dict_diff(self):
        self.assertIn("No differences", gather_dict_diff({"A": 1}, {"A": 1}))
        diff = gather_dict_diff({"A": 1, "B": {"x": 1}}, {"A": 2, "B": {"x": 1}, "C": 3})
        self.assertIn("Modified: A", diff)
        self.assertIn("Added: C: 3", diff)
        self.assertNotIn("B", diff)

    def test_has_four_vec(self):
        events = ak.Array({
            "Jet": [[{"pt": 1.0, "eta": 0.0, "phi": 0.0, "mass": 0.0}]],
            "MET": [{"pt": 1.0, "phi": 0.0}],
        })
        self.assertTrue(has_four_vec(events, "Jet"))
        self.assertFalse(has_four_vec(events, "MET"))
        self.assertFalse(has_four_vec(events, "Muon"))

    def test_get_subclasses_deep(self):
        class A:
            _subclasses = {}

        class B(A):
            _subclasses = {}

        class C(B):
            pass

        A._subclasses = {"B": B}
        B._subclasses = {"C": C}
        self.assertEqual(get_subclasses_deep(A), {"A": A, "B": B, "C": C})

    def test_call_func_safe(self):
        self.assertEqual(call_func_safe(lambda x: 2 * x, 3), 6)

        def fail():
            raise RuntimeError("expected failure")

        self.assertIsNone(call_func_safe(fail))

    def test_deferred_columns(self):
        def used_columns(config_inst, dataset):
            inst = build_taf(_test_deferred_columns, config_inst, dataset)
            return {route.string_column for route in inst.used_columns}

        cfg = toy_config(tags={"is_dl", "is_hh"}, btag_column="btagUParTAK4B", met_name="PuppiMET")
        self.assertEqual(used_columns(cfg, "toy_mc"), {
            "nano_v15", "nano_ge_v14", "run3", "y2024", "dl", "hh", "mc", "Jet.btagUParTAK4B", "PuppiMET.pt",
        })
        self.assertEqual(used_columns(cfg, "toy_data"), {
            "nano_v15", "nano_ge_v14", "run3", "y2024", "dl", "hh", "data", "Jet.btagUParTAK4B", "PuppiMET.pt",
        })

        # the dataset campaign takes precedence over the config campaign for nano version checks
        cfg = toy_config(
            tags={"is_sl", "is_hhh"}, campaign_name="run2_2017_nano_v14", year=2017, run=2,
            btag_column="btagDeepFlavB", met_name="MET",
        )
        cfg.get_dataset("toy_mc").x.campaign = "run2_2017_nano_v15"
        cfg.get_dataset("toy_data").add_tag("is_dy")
        self.assertEqual(used_columns(cfg, "toy_mc"), {
            "nano_v15", "nano_ge_v14", "run2", "not_y2024", "sl", "hhh", "mc", "Jet.btagDeepFlavB", "MET.pt",
        })
        self.assertEqual(used_columns(cfg, "toy_data"), {
            "nano_v14", "nano_ge_v14", "run2", "not_y2024", "sl", "hhh", "data", "dy", "Jet.btagDeepFlavB",
            "MET.pt",
        })

    def test_deferred_columns_missing_aux(self):
        # BTAG_COLUMN and MET_COLUMN require their aux entries in the config
        cfg = toy_config()
        with self.assertRaises(Exception):
            build_taf(_test_deferred_columns, cfg, "toy_mc").used_columns


if __name__ == "__main__":
    unittest.main()
