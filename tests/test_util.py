# coding: utf-8

"""
unittests for hbw.util
"""

import unittest

from columnflow.util import maybe_import

from hbw.util import build_param_product, round_sig, dict_diff, four_vec, call_once_on_config, gather_dict_diff

import order as od

np = maybe_import("numpy")
ak = maybe_import("awkward")


class TestDictDiff(unittest.TestCase):
    def test_no_difference(self):
        dict1 = {"name": "Alice", "age": 25}
        dict2 = {"name": "Alice", "age": 25}
        result = gather_dict_diff(dict1, dict2)
        self.assertEqual(result, "✅ No differences found.")

    def test_simple_modification(self):
        dict1 = {"name": "Alice", "age": 25}
        dict2 = {"name": "Alice", "age": 26}
        result = gather_dict_diff(dict1, dict2)
        expected_output = (
            "🔄 Modified: age:\n"
            "    - Old: 25\n"
            "    - New: 26"
        )
        self.assertEqual(result, expected_output)

    def test_addition(self):
        dict1 = {"name": "Alice"}
        dict2 = {"name": "Alice", "hobby": "cycling"}
        result = gather_dict_diff(dict1, dict2)
        expected_output = "🔹 Added: hobby: cycling"
        self.assertEqual(result, expected_output)

    def test_removal(self):
        dict1 = {"name": "Alice", "hobby": "cycling"}
        dict2 = {"name": "Alice"}
        result = gather_dict_diff(dict1, dict2)
        expected_output = "🔻 Removed: hobby: cycling"
        self.assertEqual(result, expected_output)

    def test_nested_modification(self):
        dict1 = {
            "name": "Alice",
            "skills": {
                "python": "intermediate",
                "sql": "beginner",
            },
        }
        dict2 = {
            "name": "Alice",
            "skills": {
                "python": "advanced",
                "sql": "beginner",
            },
        }
        result = gather_dict_diff(dict1, dict2)
        expected_output = (
            "🔄 Modified: skills:\n"
            "    🔄 Modified: python:\n"
            "        - Old: intermediate\n"
            "        - New: advanced"
        )
        self.assertEqual(result, expected_output)

    def test_nested_addition(self):
        dict1 = {
            "name": "Alice",
            "skills": {
                "python": "intermediate",
            },
        }
        dict2 = {
            "name": "Alice",
            "skills": {
                "python": "intermediate",
                "docker": "beginner",
            },
        }
        result = gather_dict_diff(dict1, dict2)
        expected_output = (
            "🔄 Modified: skills:\n"
            "    🔹 Added: docker: beginner"
        )
        self.assertEqual(result, expected_output)

    def test_complex_diff(self):
        dict1 = {
            "name": "Alice",
            "age": 25,
            "skills": {
                "python": "intermediate",
                "sql": "beginner",
            },
        }
        dict2 = {
            "name": "Alice",
            "age": 26,
            "skills": {
                "python": "advanced",
                "sql": "beginner",
                "docker": "beginner",
            },
            "hobby": "cycling",
        }
        result = gather_dict_diff(dict1, dict2)
        expected_output = (
            "🔄 Modified: skills:\n"
            "    🔹 Added: docker: beginner\n"
            "    🔄 Modified: python:\n"
            "        - Old: intermediate\n"
            "        - New: advanced\n"
            "🔹 Added: hobby: cycling"
            "🔄 Modified: age:\n"
            "    - Old: 25\n"
            "    - New: 26\n"
        )
        self.assertEqual(result, expected_output)


class HbwUtilTest(
    TestDictDiff,
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


if __name__ == "__main__":
    unittest.main()
