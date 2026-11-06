# coding: utf-8

"""
unittests for hbw.plotting.unrolled and hbw.plotting.multi
"""

import unittest

from columnflow.util import maybe_import

from hbw.plotting.multi import multi_cat_plot, multi_var_plot
from hbw.plotting.unrolled import (
    multi_cat_unrolled_plot, multi_var_unrolled_plot,
    unroll_hists, build_unroll_labels, get_level_spans, order_categories, get_unrolled_xticks, make_unroll_blocks,
)

import order as od

np = maybe_import("numpy")
hist = maybe_import("hist")
sn = maybe_import("scinum")
matplotlib = maybe_import("matplotlib")


def make_hist(values, shifts=("nominal",), var_name="x"):
    h = (
        hist.Hist.new
        .StrCat(list(shifts), name="shift")
        .Regular(len(values), 0, 1, name=var_name)
        .Weight()
    )
    for i in range(len(shifts)):
        h.view()[i]["value"] = values
        h.view()[i]["variance"] = 2 * np.asarray(values, dtype=float)
    return h


class HbwPlottingMultiTest(
    unittest.TestCase,
):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config_inst = od.Config(name="test_config", id=123456)
        self.tt = od.Process(name="tt", id=1)
        self.dy = od.Process(name="dy", id=2)

        # root categories and their combinations, built like in hbw.config.categories
        root_cats = {
            "main": ["sr"],
            "lep": ["2e", "2mu"],
            "jet": ["resolved", "boosted"],
            "b": ["1b", "2b"],
        }
        for i, names in enumerate(root_cats.values()):
            for j, name in enumerate(names):
                self.config_inst.add_category(name=name, id=10**i * (j + 1), label=f"label_{name}")
        self.config_inst.get_category("boosted").label_short = "Boost"

        # per lepton channel: resolved 1b, resolved 2b, boosted 2b
        combinations = [
            (lep, jet, b)
            for lep in root_cats["lep"]
            for jet, b in [("resolved", "1b"), ("resolved", "2b"), ("boosted", "2b")]
        ]
        self.category_insts = [
            od.Category(
                name=f"sr__{lep}__{jet}__{b}",
                id=1000 + i,
                aux={"root_cats": {"main": "sr", "lep": lep, "jet": jet, "b": b}},
            )
            for i, (lep, jet, b) in enumerate(combinations)
        ]

    def test_unroll_hists(self):
        hists = {
            "cat_a": {self.tt: make_hist([1, 2]), self.dy: make_hist([3, 4])},
            "cat_b": {self.tt: make_hist([5, 6, 7])},
        }
        unrolled, n_bins = unroll_hists(hists, "x", axis_name="x_unrolled")

        self.assertEqual(n_bins, [2, 3])
        self.assertEqual(list(unrolled), [self.tt, self.dy])

        h_tt = unrolled[self.tt]
        self.assertEqual(h_tt.axes.name, ("shift", "x_unrolled"))
        self.assertEqual(list(h_tt.axes["x_unrolled"].edges), [0, 1, 2, 3, 4, 5])
        self.assertEqual(list(h_tt.axes["shift"]), ["nominal"])
        np.testing.assert_array_equal(h_tt.view()["value"][0], [1, 2, 5, 6, 7])
        np.testing.assert_array_equal(h_tt.view()["variance"][0], [2, 4, 10, 12, 14])

        # dy is missing in cat_b and filled with zeros there
        np.testing.assert_array_equal(unrolled[self.dy].view()["value"][0], [3, 4, 0, 0, 0])
        np.testing.assert_array_equal(unrolled[self.dy].view()["variance"][0], [6, 8, 0, 0, 0])

    def test_unroll_hists_multiple_shifts(self):
        shifts = ("nominal", "up")
        hists = {
            "cat_a": {self.tt: make_hist([1, 2], shifts=shifts)},
            "cat_b": {self.tt: make_hist([3], shifts=shifts)},
        }
        unrolled, _ = unroll_hists(hists, "x")

        h_tt = unrolled[self.tt]
        self.assertEqual(list(h_tt.axes["shift"]), list(shifts))
        np.testing.assert_array_equal(h_tt.view()["value"], [[1, 2, 3], [1, 2, 3]])

    def test_unroll_hists_mismatching_axes(self):
        hists = {
            "cat_a": {self.tt: make_hist([1, 2])},
            "cat_b": {self.tt: make_hist([3], shifts=("up",))},
        }
        with self.assertRaises(ValueError):
            unroll_hists(hists, "x")

    def test_unroll_hists_different_variables(self):
        # one block per variable, with different axis names and numbers of bins
        hists = {
            "x": {self.tt: make_hist([1, 2], var_name="x"), self.dy: make_hist([3, 4], var_name="x")},
            "y": {self.tt: make_hist([5, 6, 7], var_name="y")},
        }
        unrolled, n_bins = unroll_hists(hists, ["x", "y"], axis_name="vars_unrolled")

        self.assertEqual(n_bins, [2, 3])
        h_tt = unrolled[self.tt]
        self.assertEqual(h_tt.axes.name, ("shift", "vars_unrolled"))
        self.assertEqual(list(h_tt.axes["vars_unrolled"].edges), [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(h_tt.view()["value"][0], [1, 2, 5, 6, 7])
        np.testing.assert_array_equal(unrolled[self.dy].view()["value"][0], [3, 4, 0, 0, 0])

        # the axis name defaults to the first variable
        unrolled, _ = unroll_hists(hists, ["x", "y"])
        self.assertEqual(unrolled[self.tt].axes.name, ("shift", "x"))

        # other axes must still agree, and one variable name is needed per block
        hists["y"][self.tt] = make_hist([5], shifts=("up",), var_name="y")
        with self.assertRaises(ValueError):
            unroll_hists(hists, ["x", "y"])
        with self.assertRaises(ValueError):
            unroll_hists(hists, ["x"])

    def test_make_unroll_blocks(self):
        blocks = make_unroll_blocks(
            paths=[("x",), ("y",)],
            labels=[["label x"], ["label y"]],
            n_bins=[2, 3],
        )
        self.assertEqual([block.start for block in blocks], [0, 2])
        self.assertEqual([block.n_bins for block in blocks], [2, 3])
        self.assertEqual([block.path for block in blocks], [("x",), ("y",)])
        self.assertEqual([block.labels for block in blocks], [["label x"], ["label y"]])

    def test_build_unroll_labels(self):
        n_bins = [2, 2, 1, 2, 2, 1]
        common_label, blocks = build_unroll_labels(self.config_inst, self.category_insts, n_bins)

        self.assertEqual(common_label, "label_sr")
        self.assertEqual([block.start for block in blocks], [0, 2, 4, 5, 7, 9])
        self.assertEqual([block.n_bins for block in blocks], n_bins)
        # lep changes least often (outermost level), jet and b change equally often and keep their order
        self.assertEqual(
            [block.labels for block in blocks],
            [
                [lep, jet, b]
                for lep in ["label_2e", "label_2mu"]
                for jet, b in [("label_resolved", "label_1b"), ("label_resolved", "label_2b"), ("Boost", "label_2b")]
            ],
        )
        self.assertEqual(blocks[2].path, ("2e", "boosted", "2b"))

    def test_build_unroll_labels_missing_root_cats(self):
        # alphabetically sorted, as passed by the task; boosted categories have no b-tag root category
        names = ["sr__2e__boosted", "sr__2e__resolved__1b", "sr__2mu__boosted", "sr__2mu__resolved__1b"]
        category_insts = [
            od.Category(
                name=name,
                id=3000 + i,
                aux={"root_cats": dict(zip(("main", "lep", "jet", "b"), name.split("__")))},
            )
            for i, name in enumerate(names)
        ]
        _, blocks = build_unroll_labels(self.config_inst, category_insts, [1, 1, 1, 1])

        # missing root categories get an empty label
        self.assertEqual(
            [block.labels for block in blocks],
            [
                ["label_2e", "Boost", ""], ["label_2e", "label_resolved", "label_1b"],
                ["label_2mu", "Boost", ""], ["label_2mu", "label_resolved", "label_1b"],
            ],
        )

    def test_build_unroll_labels_from_names(self):
        # without root_cats aux, categories are split at "__" and unknown tokens are used as labels
        category_insts = [
            od.Category(name=name, id=2000 + i)
            for i, name in enumerate(["sr__2e__resolved__1b", "sr__2e__resolved__2b", "sr__2e__boosted"])
        ]
        common_label, blocks = build_unroll_labels(self.config_inst, category_insts, [3, 4, 2])

        self.assertEqual(common_label, "label_sr\nlabel_2e")
        self.assertEqual(
            [block.labels for block in blocks],
            [["label_resolved", "label_1b"], ["label_resolved", "label_2b"], ["Boost", ""]],
        )

    def test_get_level_spans(self):
        # 2 lepton channels with resolved 1b, resolved 2b and boosted 2b each
        _, blocks = build_unroll_labels(self.config_inst, self.category_insts, [2, 2, 1, 2, 2, 1])

        self.assertEqual(get_level_spans(blocks, 0), [(0, 5, "label_2e"), (5, 10, "label_2mu")])
        # consecutive resolved blocks are merged, but not across lepton channels
        self.assertEqual(
            get_level_spans(blocks, 1),
            [(0, 4, "label_resolved"), (4, 5, "Boost"), (5, 9, "label_resolved"), (9, 10, "Boost")],
        )
        self.assertEqual(
            get_level_spans(blocks, 2),
            [
                (0, 2, "label_1b"), (2, 4, "label_2b"), (4, 5, "label_2b"),
                (5, 7, "label_1b"), (7, 9, "label_2b"), (9, 10, "label_2b"),
            ],
        )

    def make_categories(self, names):
        return [
            od.Category(
                name=name,
                id=4000 + i,
                aux={"root_cats": dict(zip(("main", "lep", "jet", "b"), name.split("__")))},
            )
            for i, name in enumerate(names)
        ]

    def test_order_categories(self):
        # alphabetical order, as passed by the task
        category_insts = self.make_categories(sorted(
            f"sr__{lep}__{jet}"
            for lep in ["2e", "2mu", "emu"]
            for jet in ["resolved__1b", "resolved__2b", "boosted"]
        ))
        get_names = lambda cat_insts: [cat_inst.name.replace("sr__", "") for cat_inst in cat_insts]

        # default: levels in name order, values in order of appearance, missing root categories last
        ordered, level_keys = order_categories(category_insts)
        self.assertEqual(level_keys, ["lep", "jet", "b"])
        self.assertEqual(get_names(ordered)[:3], ["2e__boosted", "2e__resolved__1b", "2e__resolved__2b"])

        # configured order of root category names
        ordered, _ = order_categories(category_insts, category_order=("sr", "resolved", "boosted"))
        self.assertEqual(get_names(ordered)[:3], ["2e__resolved__1b", "2e__resolved__2b", "2e__boosted"])

        # configured level order, remaining levels in name order
        ordered, level_keys = order_categories(
            category_insts,
            category_order=("resolved", "boosted"),
            level_order="jet",
        )
        self.assertEqual(level_keys, ["jet", "lep", "b"])
        self.assertEqual(
            get_names(ordered),
            [
                "2e__resolved__1b", "2e__resolved__2b", "2mu__resolved__1b", "2mu__resolved__2b",
                "emu__resolved__1b", "emu__resolved__2b", "2e__boosted", "2mu__boosted", "emu__boosted",
            ],
        )

    def test_get_unrolled_xticks(self):
        # two categories with 3 and 2 bins
        block_starts = [0, 3]
        edges = [[0, 50, 100, 200], [0, 0.5, 1]]

        # bin edges, the upper edge is only shown for the last category
        positions, labels = get_unrolled_xticks(block_starts, edges)
        self.assertEqual(positions, [0, 1, 2, 3, 4, 5])
        self.assertEqual(labels, ["0", "50", "100", "0", "0.5", "1"])

        # every second tick of each category is labeled, but not next to the first tick of the next category
        _, labels = get_unrolled_xticks(block_starts, edges, step=2)
        self.assertEqual(labels, ["0", "", "", "0", "", "1"])
        _, labels = get_unrolled_xticks([0, 4], [[0, 1, 2, 3, 4], [0, 1, 2]], step=2)
        self.assertEqual(labels, ["0", "", "2", "", "0", "", "2"])

    def test_get_unrolled_xticks_discrete(self):
        # integer bins centered at 2, 3, 4 (e.g. number of jets)
        block_starts = [0, 3]
        edges = [[1.5, 2.5, 3.5, 4.5]] * 2

        positions, labels = get_unrolled_xticks(block_starts, edges, discrete_x=True)
        self.assertEqual(positions, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        self.assertEqual(labels, ["2", "3", "4"] * 2)

        # custom bin labels are used when given for each bin
        _, labels = get_unrolled_xticks(block_starts, edges, discrete_x=True, x_labels=[["two", "three", "four"]] * 2)
        self.assertEqual(labels, ["two", "three", "four"] * 2)

    def test_get_unrolled_xticks_mixed(self):
        # a continuous variable with 2 bins followed by a discrete one with 3 bins and custom labels
        block_starts = [0, 2]
        edges = [[0, 50, 100], [-0.5, 0.5, 1.5, 2.5]]

        positions, labels = get_unrolled_xticks(
            block_starts,
            edges,
            discrete_x=[False, True],
            x_labels=[None, ["zero", "one", "two"]],
        )
        self.assertEqual(positions, [0, 1, 2.5, 3.5, 4.5])
        self.assertEqual(labels, ["0", "50", "zero", "one", "two"])

    def test_order_categories_unknown_level(self):
        # keys that are unknown or shared by all categories are ignored in the level order
        category_insts = self.make_categories(["sr__2mu__resolved", "sr__2e__resolved"])
        ordered, level_keys = order_categories(category_insts, category_order="2e", level_order=("main", "foo"))
        self.assertEqual(level_keys, ["lep"])
        self.assertEqual([cat_inst.name for cat_inst in ordered], ["sr__2e__resolved", "sr__2mu__resolved"])


class MinimalConfigMixin(object):
    """
    Mixin setting up a minimal config with two processes, categories and variables to draw plots.
    """

    def setUp(self):
        matplotlib.use("Agg")

        campaign = od.Campaign(name="test_campaign", id=1, ecm=13.6)
        self.config_inst = od.Config(name="test_config", id=2, campaign=campaign, aux={"luminosity": sn.Number(1000)})
        self.tt = self.config_inst.add_process(name="tt", id=1, label="tt", color1="#3f90da")
        self.dy = self.config_inst.add_process(name="dy", id=2, label="DY", color1="#ffa90e")
        for proc_inst in (self.tt, self.dy):
            proc_inst.is_mc = True
        self.category_inst = self.config_inst.add_category(name="sr", id=10, label="SR")
        self.config_inst.add_category(name="2e", id=11, label="ee")
        self.config_inst.add_category(name="2mu", id=12, label="mumu")
        self.mll = self.config_inst.add_variable(name="mll", binning=[0, 50, 100, 200], x_title="mll")
        self.n_jet = self.config_inst.add_variable(
            name="n_jet", binning=(3, 1.5, 4.5), x_title="N jets", discrete_x=True,
        )
        self.shift_insts = [od.Shift(name="nominal", id=0)]

    def make_hist(self, var_inst, values):
        h = hist.Hist(
            hist.axis.StrCategory(["nominal"], name="shift"),
            hist.axis.Variable(var_inst.bin_edges, name=var_inst.name),
            storage=hist.storage.Weight(),
        )
        h.view()["value"][0] = values
        h.view()["variance"][0] = values
        return h


class HbwPlottingUnrolledPlotTest(
    MinimalConfigMixin,
    unittest.TestCase,
):
    """
    Smoke tests drawing the unrolled multi-category and multi-variable plots on a minimal config.
    """

    def test_multi_var_unrolled_plot(self):
        hists = {
            "mll": {self.tt: self.make_hist(self.mll, [10, 20, 5]), self.dy: self.make_hist(self.mll, [3, 1, 2])},
            "n_jet": {self.tt: self.make_hist(self.n_jet, [15, 12, 8])},
        }
        fig, axs = multi_var_unrolled_plot(
            hists, self.config_inst, self.category_inst, {"mll": [self.mll], "n_jet": [self.n_jet]},
            self.shift_insts, skip_ratio=True,
        )
        self.assertEqual(axs[0].get_xlim(), (0, 6))
        # by default, edges of the continuous variable and centers of the discrete one, without an x-axis title
        self.assertEqual([t.get_text() for t in axs[-1].get_xticklabels()], ["0", "50", "100", "2", "3", "4"])
        self.assertEqual(axs[-1].get_xlabel(), "")
        # one label per variable
        self.assertTrue({"mll", "N jets"} <= {text.get_text() for text in axs[0].texts})

    def test_multi_cat_unrolled_plot(self):
        category_insts = [
            od.Category(name=f"sr__{lep}", id=20 + i, aux={"root_cats": {"main": "sr", "lep": lep}})
            for i, lep in enumerate(["2mu", "2e"])
        ]
        hists = {
            cat_inst.name: {self.tt: self.make_hist(self.mll, [10, 20, 5])}
            for cat_inst in category_insts
        }
        fig, axs = multi_cat_unrolled_plot(
            hists, self.config_inst, category_insts, [self.mll], self.shift_insts,
            unroll_category_order=["2e", "2mu"], skip_ratio=True,
        )
        self.assertEqual(axs[0].get_xlim(), (0, 6))
        # categories re-sorted, labels from the root categories
        texts = [text.get_text() for text in axs[0].texts]
        self.assertLess(texts.index("ee"), texts.index("mumu"))


class HbwPlottingMultiPlotTest(
    MinimalConfigMixin,
    unittest.TestCase,
):
    """
    Tests drawing the multi-category and multi-variable overlay plots on a minimal config.
    """

    def setUp(self):
        super().setUp()
        self.category_insts = [
            od.Category(name=f"sr__{lep}", id=20 + i, aux={"root_cats": {"main": "sr", "lep": lep}})
            for i, lep in enumerate(["2e", "2mu"])
        ]

    def make_cat_hists(self, procs=None):
        # per category i and process j: [10 * (i + 1), 20, 5 * (j + 1)]
        procs = procs or (self.tt, self.dy)
        return {
            cat_inst.name: {
                proc_inst: self.make_hist(self.mll, [10 * (i + 1), 20, 5 * (j + 1)])
                for j, proc_inst in enumerate(procs)
            }
            for i, cat_inst in enumerate(self.category_insts)
        }

    def plot_cats(self, hists=None, **kwargs):
        return multi_cat_plot(
            self.make_cat_hists() if hists is None else hists,
            self.config_inst, self.category_insts, [self.mll], self.shift_insts,
            **kwargs,
        )

    @staticmethod
    def get_steps(ax):
        # drawn histogram lines (step patches), in drawing order
        from matplotlib.patches import StepPatch
        return [patch for patch in ax.patches if isinstance(patch, StepPatch)]

    @staticmethod
    def get_legend_texts(ax):
        return [text.get_text() for text in ax.get_legend().get_texts()]

    def test_multi_cat_plot(self):
        fig, axs = self.plot_cats()

        # ratio panel shown by default
        self.assertEqual(len(axs), 2)
        # common root category in the top left, varying ones in the legend
        self.assertIn("SR", [text.get_text() for text in axs[0].texts])
        self.assertEqual(self.get_legend_texts(axs[0]), ["tt: ee", "DY: ee", "tt: mumu", "DY: mumu"])
        # one line per category and process, colored per category and styled per process
        steps = self.get_steps(axs[0])
        self.assertEqual(len(steps), 4)
        self.assertEqual(steps[0].get_edgecolor(), steps[1].get_edgecolor())
        self.assertNotEqual(steps[0].get_edgecolor(), steps[2].get_edgecolor())
        self.assertNotEqual(steps[0].get_linestyle(), steps[1].get_linestyle())
        # ratio to the same process in the first category
        ratio_steps = self.get_steps(axs[1])
        self.assertEqual(len(ratio_steps), 4)
        np.testing.assert_allclose(ratio_steps[0].get_data().values, [1, 1, 1])
        np.testing.assert_allclose(ratio_steps[2].get_data().values, [2, 1, 1])
        np.testing.assert_allclose(ratio_steps[3].get_data().values, [2, 1, 1])

        fig, axs = self.plot_cats(skip_ratio=True)
        self.assertEqual(len(axs), 1)

    def test_multi_cat_plot_single_process(self):
        # the process is shown as legend title
        fig, axs = self.plot_cats(self.make_cat_hists(procs=[self.tt]))
        self.assertEqual(self.get_legend_texts(axs[0]), ["ee", "mumu"])
        self.assertEqual(axs[0].get_legend().get_title().get_text(), "tt")

    def test_multi_cat_plot_single_category(self):
        # single category input, processes are colored with their process colors
        fig, axs = multi_cat_plot(
            self.make_cat_hists()[self.category_insts[0].name],
            self.config_inst, self.category_insts[0], [self.mll], self.shift_insts,
            process_settings={"dy": {"color1": "#000000", "label": "Drell-Yan"}},
        )
        self.assertEqual(self.get_legend_texts(axs[0]), ["tt", "Drell-Yan"])
        steps = self.get_steps(axs[0])
        self.assertEqual(matplotlib.colors.to_hex(steps[0].get_edgecolor()), "#3f90da")
        self.assertEqual(matplotlib.colors.to_hex(steps[1].get_edgecolor()), "#000000")

    def test_multi_cat_plot_process_scaling(self):
        # the scale factor is applied in all categories, but added to the process label only once
        fig, axs = self.plot_cats(process_settings={"tt": {"scale": 10}})
        self.assertEqual(
            self.get_legend_texts(axs[0]),
            ["tt$\\times$10: ee", "DY: ee", "tt$\\times$10: mumu", "DY: mumu"],
        )
        steps = self.get_steps(axs[0])
        np.testing.assert_allclose(steps[0].get_data().values, [100, 200, 50])
        np.testing.assert_allclose(steps[2].get_data().values, [200, 200, 50])

    def test_multi_cat_plot_shape_norm(self):
        fig, axs = self.plot_cats(shape_norm=True, hide_stat_errors=True)
        self.assertEqual(axs[0].get_ylabel(), "Normalized entries")
        for step in self.get_steps(axs[0]):
            self.assertAlmostEqual(step.get_data().values.sum(), 1)
        # the ratio compares shapes
        tt_mumu = np.array([20, 20, 5]) / 45
        tt_ee = np.array([10, 20, 5]) / 35
        np.testing.assert_allclose(self.get_steps(axs[1])[2].get_data().values, tt_mumu / tt_ee)
        # no error bars drawn
        self.assertEqual(len(axs[0].collections), 0)

    def test_multi_cat_plot_density(self):
        fig, axs = self.plot_cats(self.make_cat_hists(procs=[self.tt]), density=True)
        np.testing.assert_allclose(self.get_steps(axs[0])[0].get_data().values, [10 / 50, 20 / 50, 5 / 100])

    def test_multi_cat_plot_variable_settings(self):
        # overflow is added to the last bin
        hists = self.make_cat_hists(procs=[self.tt])
        for proc_hists in hists.values():
            proc_hists[self.tt].view(flow=True)["value"][0, -1] = 100
        fig, axs = self.plot_cats(hists, variable_settings={"mll": {"overflow": True}})
        np.testing.assert_allclose(self.get_steps(axs[0])[0].get_data().values, [10, 20, 105])

    def test_multi_cat_plot_style_config(self):
        # passed style config overwrites the defaults
        fig, axs = self.plot_cats(style_config={"legend_cfg": {"loc": "upper left"}, "rax_cfg": {"ylim": (0, 3)}})
        self.assertEqual(axs[0].get_legend()._loc, 2)
        self.assertEqual(axs[1].get_ylim(), (0, 3))

    def test_multi_var_plot(self):
        mll_2 = self.config_inst.add_variable(name="mll_2", binning=[0, 50, 100, 200], x_title="mll 2")
        hists = {
            "mll": {self.tt: self.make_hist(self.mll, [10, 20, 5])},
            "mll_2": {self.tt: self.make_hist(mll_2, [20, 20, 5])},
        }
        fig, axs = multi_var_plot(
            hists, self.config_inst, self.category_inst, {"mll": [self.mll], "mll_2": [mll_2]}, self.shift_insts,
        )
        self.assertEqual(len(axs), 2)
        self.assertEqual(self.get_legend_texts(axs[0]), ["mll", "mll 2"])
        np.testing.assert_allclose(self.get_steps(axs[1])[1].get_data().values, [2, 1, 1])

    def test_multi_var_plot_different_binnings(self):
        hists = {
            "mll": {self.tt: self.make_hist(self.mll, [10, 20, 5])},
            "n_jet": {self.tt: self.make_hist(self.n_jet, [15, 12, 8])},
        }
        with self.assertLogs("hbw.plotting.multi", level="WARNING"):
            fig, axs = multi_var_plot(
                hists, self.config_inst, self.category_inst, {"mll": [self.mll], "n_jet": [self.n_jet]},
                self.shift_insts,
            )
        # no ratio, x-range covering both variables
        self.assertEqual(len(axs), 1)
        self.assertEqual(axs[0].get_xlim(), (0, 200))
