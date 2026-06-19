# coding: utf-8

from __future__ import annotations

import functools

# import tabulate
import gc
import law
import order as od

from hbw.util import round_sig, timeit
from columnflow.ml import MLModel
from columnflow.util import maybe_import, DotDict
from columnflow.plotting.plot_util import get_position
from columnflow.types import TYPE_CHECKING


np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
mticker = maybe_import("matplotlib.ticker")
shap = maybe_import("shap")

if TYPE_CHECKING:
    hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


cms_label_kwargs = {
    "data": False,
    "llabel": "Private work (CMS simulation)",
    # "llabel": "Simulation Work in progress",
    # "llabel": "Simulation Preliminary",
    # "llabel": "Simulation Supplementary",
    "lumi": "62",  # NOTE: hard-coded, to be updated if needed
    # "exp": "",
}
if "CMS" in cms_label_kwargs["llabel"]:
    cms_label_kwargs["exp"] = ""


def barplot_from_multidict(
        dict_of_rankings: dict[str, dict],
        normalize_weights: bool = True,
        title: str | None = None,
):
    """
    :param dict_of_rankings: dictionary of multiple dictionaries of rankings of variables. The keys of this
    dictionary are interpreted as labels for different types of variable rankings. The keys of the sub-dictionaries
    correspond to the names of the variables to be ranked and they should be identical for each sub-dictionary.
    The first sub-directory is used for the sorting of variables.
    :param normalize_weights: whether to normalize the sum of weights per ranking to 1.
    """
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 10))

    num_dicts = len(dict_of_rankings.keys())
    num_labels = len(dict_of_rankings[list(dict_of_rankings.keys())[0]].keys())
    labels = list(dict_of_rankings[list(dict_of_rankings.keys())[0]].keys())[::-1]

    bar_width = 0.9 / num_dicts
    index = np.arange(num_labels)

    for idx, (d_label, d) in enumerate(dict_of_rankings.items()):
        # always get labels in the same order
        weights = [d[label] for label in labels]
        if normalize_weights:
            weights = weights / np.sum(weights)

        # Offset to separate bars from different dictionaries
        offset = idx * bar_width

        ax.barh(index - offset, weights, bar_width, label=d_label)

    xlabel = f"Contribution to {title}" if title else "Contribution"
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel("Input feature", fontsize=20)
    ax.set_yticks(index - (bar_width * (num_dicts - 1)) / 2)
    ax.set_yticklabels(labels)  # fontsize = 14?
    ax.legend(fontsize=18, loc="lower right")
    # ax.legend(title=title, fontsize=18, title_fontsize=20, loc="lower right")

    mplhep.cms.label(ax=ax, **cms_label_kwargs, com=13.6, fontsize=16)
    plt.tight_layout()

    return fig, ax


def _sample_shap_background_per_class(
    features,
    labels,
    num_events_per_class: int,
    random_state: int = 42,
):
    sampled = []
    rng = np.random.default_rng(random_state)
    sampled_counts = {}
    for cls in np.unique(labels):
        class_features = features[labels == cls]
        if len(class_features) == 0:
            continue
        n_sample = min(num_events_per_class, len(class_features))
        sampled.append(shap.utils.sample(class_features, n_sample, random_state=random_state))
        sampled_counts[int(cls)] = int(n_sample)

    if not sampled:
        logger.warning("No class-wise background samples found, using fallback sample over all events")
        return shap.utils.sample(features, min(num_events_per_class, len(features)), random_state=random_state)

    background = np.concatenate(sampled, axis=0)
    shuffle_idx = rng.permutation(len(background))
    logger.info(
        "SHAP background sampled: n_events=%d, n_classes=%d, per_class=%s",
        len(background),
        len(sampled_counts),
        sampled_counts,
    )
    return background[shuffle_idx]


def _sample_shap_explain_values(
    features,
    labels,
    max_events: int = 100,
    random_state: int = 42,
):
    # Sample explain events per class to keep SHAP slices separable by process.
    sampled_features = []
    sampled_labels = []
    rng = np.random.default_rng(random_state)
    sampled_counts = {}

    for cls in np.unique(labels):
        class_features = features[labels == cls]
        if len(class_features) == 0:
            continue
        n_sample = min(max_events, len(class_features))
        cls_sample = shap.utils.sample(class_features, n_sample, random_state=random_state)
        sampled_features.append(cls_sample)
        sampled_labels.append(np.full(len(cls_sample), cls, dtype=labels.dtype))
        sampled_counts[int(cls)] = int(n_sample)

    if not sampled_features:
        logger.warning("No class-wise explain samples found, using fallback sample over all events")
        sampled_features = [shap.utils.sample(features, min(max_events, len(features)), random_state=random_state)]
        sampled_labels = [np.full(len(sampled_features[0]), -1, dtype=labels.dtype)]

    explain_inputs = np.concatenate(sampled_features, axis=0)
    explain_labels = np.concatenate(sampled_labels, axis=0)
    shuffle_idx = rng.permutation(len(explain_inputs))
    logger.info(
        "SHAP explain sampled: n_events=%d, n_classes=%d, per_class=%s",
        len(explain_inputs),
        len(sampled_counts),
        sampled_counts,
    )
    return explain_inputs[shuffle_idx], explain_labels[shuffle_idx]


def _calculate_shap_values(
    model: MLModel,
    background,
    explain_inputs,
    input_features: list | None,
):
    predict = functools.partial(model.trained_model.predict, verbose=0)
    explainer = shap.Explainer(predict, background)
    logger.info(
        "Computing SHAP values with background shape %s and explain shape %s",
        getattr(background, "shape", None),
        getattr(explain_inputs, "shape", None),
    )
    shap_values = explainer(explain_inputs)
    if input_features is not None:
        shap_values.feature_names = list(input_features)
    return shap_values


def _shap_ranking_for_output_node(shap_values, output_node: int) -> dict:
    ranking = dict(zip(shap_values.feature_names, shap_values[:, :, output_node].abs.mean(axis=0).values))
    return dict(sorted(ranking.items(), key=lambda x: abs(x[1]), reverse=True))


def _plot_shap_scatter_plots(
    model: MLModel,
    output: law.FileSystemDirectoryTarget,
    shap_values,
    explain_labels,
    class_label_map: dict[int, str],
    output_node: int,
    postfix: str,
    cmap: str | None = None,
    input_features: list | None = None,  # HOTFIX: used only to set output file name
) -> None:
    plt.style.use("seaborn-v0_8")
    feature_labels = list(shap_values.feature_names)
    if input_features and len(input_features) != len(feature_labels):
        logger.warning(
            "Provided input_features list has length %d but shap_values has %d features, ignoring input_features",
            len(input_features),
            len(feature_labels),
        )
        input_features = None
    cmap = plt.get_cmap("tab10")  # for now, overwrite any custom cmap
    class_ids = [int(cls) for cls in np.unique(explain_labels)]
    output_class = class_label_map.get(output_node, str(output_node))
    logger.info(
        "Creating SHAP scatter plots for class %s with %d features across classes %s",
        output_class,
        len(feature_labels),
        class_ids,
    )

    for feature_idx, feature_label in enumerate(feature_labels):
        try:
            if input_features:
                feature_name = input_features[feature_idx]
            logger.debug("Scatter SHAP plot: feature='%s', node=%d", feature_label, output_node)

            sub_shap_values = shap_values[:, feature_label, output_node]
            explain_class_labels = np.array(
                [class_label_map.get(int(lbl), str(lbl)) for lbl in explain_labels],
                dtype=object,
            )

            process_labels = shap.Explanation(values=sub_shap_values.values, data=explain_labels)
            process_labels.display_data = explain_class_labels

            shap.plots.scatter(
                sub_shap_values,
                show=False,
                color=process_labels,
                xmin=sub_shap_values.percentile(1),
                xmax=sub_shap_values.percentile(99),
                dot_size=8,
                alpha=1.0,
                # TODO: for some reason the last color legend does not match the scatter points,
                # to be investigated (maybe a bug in shap.plots.scatter with categorical coloring?
                cmap=cmap,
            )

            fig = plt.gcf()
            ax = fig.axes[0]

            x_title = (
                feature_label if isinstance(feature_label, str)
                else model.config_inst.get_variable(feature_label).x_title
            )
            ax.set_xlabel(x_title, fontsize=16)
            ax.set_ylabel(f"SHAP value ({output_class} node)", fontsize=16)
            mplhep.cms.label(ax=ax, **cms_label_kwargs, com=13.6, fontsize=14)
            plt.tight_layout()
            output.child(f"shap_scatter_{feature_name}_node{output_node}{postfix}.pdf", type="f").dump(
                fig,
                formatter="mpl",
            )
            plt.close(fig)
        except Exception:
            logger.exception(
                f"Failed to produce SHAP scatter plot for feature '{feature_name}' "
                f"(label: '{feature_label}') and output node {output_node}",
            )


def _plot_shap_waterfall_plots(
    model: MLModel,
    output: law.FileSystemDirectoryTarget,
    shap_values,
    explain_inputs,
    output_node: int,
    postfix: str,
) -> None:
    """
    Note: this plot function fails (or takes forever) when the mplhep.style.CMS style is used, to be investigated.
    """
    try:
        plt.style.use("seaborn-v0_8")
        predicted_values = model.trained_model.predict(explain_inputs, verbose=0)[:, output_node]
        sorted_idxs = np.argsort(predicted_values)
        signal_idx = sorted_idxs[0]
        bkg_idx = sorted_idxs[-1]

        shap.plots.waterfall(shap_values[signal_idx, :, output_node], max_display=10, show=True)
        fig = plt.gcf()
        for ax in fig.axes:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
        output.child(f"shap_waterfall_sig_node{output_node}{postfix}.pdf", type="f").dump(fig, formatter="mpl")
        plt.close(fig)

        shap.plots.waterfall(shap_values[bkg_idx, :, output_node], max_display=10, show=True)
        fig = plt.gcf()
        for ax in fig.axes:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
        output.child(f"shap_waterfall_bkg_node{output_node}{postfix}.pdf", type="f").dump(fig, formatter="mpl")
        plt.close(fig)
    except Exception:
        logger.exception(f"Failed to produce SHAP waterfall plots for output node {output_node}")


def _plot_shap_rankings(
    model: MLModel,
    output: law.FileSystemDirectoryTarget,
    explain_inputs,
    shap_ranking_dict: dict,
    output_node: int,
    input_features: list | None,
    postfix: str,
):
    from hbw.ml.introspection import sensitivity_analysis, gradient_times_input

    rankings = {
        "SHAP": shap_ranking_dict,
        "Sensitivity Analysis": sensitivity_analysis(
            model.trained_model,
            explain_inputs,
            output_node,
            input_features,
        ),
        "Gradient * Input": gradient_times_input(
            model.trained_model,
            explain_inputs,
            output_node,
            input_features,
        ),
    }
    title = {
        "ggfv3": r"$NN_{ggF}$" + (" (bkg node)" if output_node == 1 else ""),
        "vbfv3": r"$NN_{VBF}$" + (" (bkg node)" if output_node == 1 else ""),
        "vbfv3_tag": r"$NN_{VBF}$" + (" (bkg node)" if output_node == 1 else ""),
        "multiclassv3": r"$NN_{mult}$" + {
            0: r" ($HH_{ggF}$ node)",
            1: r" ($HH_{VBF}$ node)",
            2: r" ($t\bar{t}$ node)",
            3: r" (single $t/\bar{t}$ node)",
            4: r" (DY node)",
            5: r" (H node)",
        }[output_node],
    }.get(model.cls_name, model.cls_name)
    fig, ax = barplot_from_multidict(rankings, title=title)
    logger.info("Saving SHAP ranking comparison plot for node %d", output_node)
    output.child(f"rankings_node{output_node}{postfix}.pdf", type="f").dump(fig, formatter="mpl")

    reduced_rankings = rankings.copy()
    reduced_rankings.pop("Gradient * Input")
    fig, ax = barplot_from_multidict(reduced_rankings, title=title)
    output.child(f"rankings_node{output_node}{postfix}_reduced.pdf", type="f").dump(fig, formatter="mpl")
    plt.close(fig)
    return fig, ax


@timeit
def plot_introspection(
    model: MLModel,
    output: law.FileSystemDirectoryTarget,
    inputs,
    postfix: str = "",
    input_features: list | None = None,
    stats: dict | None = None,
    store_shap_values: bool = False,
):
    input_labels = None
    if input_features:
        input_labels = [
            model.config_inst.get_variable(feature).x_title if isinstance(feature, str) else feature.x_title
            for feature in input_features
        ]

    class_label_map = {
        node_config["ml_id"]: proc
        for proc, node_config in model.train_nodes.items()
    }
    class_colors = {
        _id: model.config_inst.get_process(proc).color for _id, proc in class_label_map.items()
    }
    from matplotlib.colors import LinearSegmentedColormap
    # NOTE: this colormap does not really work as I want to... use ListedColormap instead?
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", [class_colors[_id] for _id in sorted(class_colors.keys())], N=len(class_colors),
    )

    class_label_map = {_id: model.config_inst.get_process(proc).label for _id, proc in class_label_map.items()}
    missing_labels = [int(lbl) for lbl in np.unique(inputs.labels) if int(lbl) not in class_label_map]
    if missing_labels:
        raise ValueError(
            "SHAP introspection requires that all classes in the inputs "
            "have a corresponding process in the model's train nodes. "
            f"Missing classes: {missing_labels}",
        )

    features = inputs.features  # numpy array with multiple features per event
    labels = inputs.labels  # np.array integers
    num_bkg_events_per_class = 1000
    num_explain_events_per_class = 100
    logger.info(
        "Starting plot_introspection: features_shape=%s, unique_labels=%s, postfix='%s'",
        getattr(features, "shape", None),
        [int(x) for x in np.unique(labels)],
        postfix,
    )

    # first, sample a single background distribution for SHAP with up to N events per class
    background = _sample_shap_background_per_class(features, labels, num_bkg_events_per_class)
    explain_inputs, explain_labels = _sample_shap_explain_values(
        features,
        labels,
        num_explain_events_per_class,
    )
    shap_target = output.parent.child(f"shap_values{postfix}.pkl", type="f")
    if shap_target.exists():
        logger.info("Found existing SHAP values at %s, loading from file", shap_target.path)
        shap_payload = shap_target.load(formatter="pickle")
        shap_values = shap.Explanation(
            values=shap_payload["values"],
            base_values=shap_payload["base_values"],
            data=shap_payload["data"],
            # feature_names=shap_payload["feature_names"],
            feature_names=input_labels if input_labels else shap_payload["feature_names"],
        )
        if not np.array_equal(shap_values.data, explain_inputs):
            raise ValueError("SHAP explain inputs do not match loaded SHAP values data, cannot reuse SHAP values")
        if not np.array_equal(shap_payload["labels"], explain_labels):
            raise ValueError("SHAP explain labels do not match loaded SHAP payload labels, cannot reuse SHAP values")
        explain_labels = shap_payload["labels"]
        logger.info(
            "Loaded SHAP payload: values_shape=%s, labels_shape=%s, output_nodes=%s",
            getattr(shap_values.values, "shape", None),
            getattr(explain_labels, "shape", None),
            shap_payload.get("output_nodes", []),
        )
    else:
        shap_values = _calculate_shap_values(model, background, explain_inputs, input_labels)
        logger.info("SHAP values computed with shape %s", getattr(shap_values.values, "shape", None))
        if store_shap_values:
            shap_payload = {
                "values": np.asarray(shap_values.values),
                "base_values": np.asarray(shap_values.base_values),
                "data": np.asarray(shap_values.data),
                "labels": np.asarray(explain_labels),
                "feature_names": list(shap_values.feature_names),
                # "output_nodes": [int(i) for i in output_nodes],
            }
            shap_target.dump(shap_payload, formatter="pickle")
            logger.info(
                "Stored SHAP payload: values_shape=%s, labels_shape=%s, output_nodes=%s",
                getattr(shap_payload["values"], "shape", None),
                getattr(shap_payload["labels"], "shape", None),
                shap_payload.get("output_nodes", []),
            )

    output_nodes = sorted(np.unique(labels))
    fig = ax = None
    for output_node in output_nodes:
        logger.info("Generating SHAP plots for output node %d", int(output_node))
        shap_ranking_dict = _shap_ranking_for_output_node(shap_values, output_node)

        _plot_shap_scatter_plots(
            model, output, shap_values, explain_labels, class_label_map, output_node, postfix, cmap=cmap,
            input_features=input_features,
        )
        # _plot_shap_waterfall_plots(model, output, shap_values, explain_inputs, output_node, postfix)
        fig, ax = _plot_shap_rankings(
            model,
            output,
            explain_inputs,
            shap_ranking_dict,
            output_node,
            input_labels,
            postfix,
        )

    return fig, ax


@timeit
def plot_introspection_old(
    model: MLModel,
    output: law.FileSystemDirectoryTarget,
    inputs,
    output_node: int = 0,
    postfix: str = "",
    input_features: list | None = None,
    stats: dict | None = None,
    store_shap_values: bool = False,
):
    from hbw.ml.introspection import sensitivity_analysis, gradient_times_input, shap_ranking

    # get only target-node events and subsample for expensive SHAP computations
    inputs = inputs.features[inputs.labels == output_node]

    shap_ranking_dict, shap_values = shap_ranking(model.trained_model, inputs, output_node, input_features)

    # get signal-like & bkg-like events
    import shap
    plt.style.use("seaborn-v0_8")
    predicted_values = model.trained_model.predict(inputs[:100], verbose=0)[:, output_node]
    sorted_idxs = np.argsort(predicted_values)
    signal_idx = sorted_idxs[0]
    bkg_idx = sorted_idxs[-1]
    for input_feature in shap_values.feature_names:
        try:
            shap.plots.scatter(shap_values[:, input_feature, output_node], show=False)
            fig = plt.gcf()
            for ax in fig.axes:
                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
            output.child(f"shap_scatter_{input_feature}{postfix}.pdf", type="f").dump(fig, formatter="mpl")
            plt.close(fig)
        except Exception:
            logger.exception(f"Failed to produce SHAP scatter plot for feature '{input_feature}'")
    try:
        shap.plots.waterfall(shap_values[signal_idx, :, output_node], max_display=10, show=True)
        fig = plt.gcf()
        for ax in fig.axes:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
        output.child(f"shap_waterfall_sig{postfix}.pdf", type="f").dump(fig, formatter="mpl")
        plt.close(fig)

        shap.plots.waterfall(shap_values[bkg_idx, :, output_node], max_display=10, show=True)
        fig = plt.gcf()
        for ax in fig.axes:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
        output.child(f"shap_waterfall_bkg{postfix}.pdf", type="f").dump(fig, formatter="mpl")
        plt.close(fig)
    except Exception:
        logger.exception("Failed to produce SHAP waterfall plots")

    if store_shap_values:
        shap_payload = {
            "values": np.asarray(shap_values.values),
            "base_values": np.asarray(shap_values.base_values),
            "data": np.asarray(shap_values.data),
            "feature_names": list(shap_values.feature_names),
            "output_node": int(output_node),
        }
        output.child(f"shap_values{postfix}.pkl", type="f").dump(shap_payload, formatter="pickle")

    rankings = {
        "SHAP": shap_ranking_dict,
        "Sensitivity Analysis": sensitivity_analysis(model.trained_model, inputs, output_node, input_features),
        "Gradient * Input": gradient_times_input(model.trained_model, inputs, output_node, input_features),
    }
    # TODO: dump rankings in stats json (need to convert float32 into str for json compatibility)
    # if stats:
    #     stats["rankings"] = rankings
    fig, ax = barplot_from_multidict(rankings)

    output.child(f"rankings{postfix}.pdf", type="f").dump(fig, formatter="mpl")
    return fig, ax


@timeit
def plot_history(
    history,
    output: law.FileSystemDirectoryTarget,
    metric: str = "loss",
    ylabel: str | None = None,
    yscale: str = "linear",
    output_name: str | None = None,
):
    """
    Simple function to create and store a plot from history data
    """
    # set default parameters if not assigned
    ylabel = ylabel or metric
    output_name = (output_name or ylabel).replace(" ", "")

    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    fig, ax = plt.subplots()
    ax.plot(history[metric])
    ax.plot(history[f"val_{metric}"])
    ax.set(**{
        "ylabel": ylabel,
        "yscale": yscale,
        "xlabel": "Epoch",
    })
    ax.legend(["train", "validation"], loc="best")
    mplhep.cms.label(ax=ax, **cms_label_kwargs, com=13.6)

    plt.tight_layout()
    output.child(f"{output_name}.pdf", type="f").dump(fig, formatter="mpl")


plot_loss = functools.partial(plot_history, metric="loss", ylabel="Loss")
plot_accuracy = functools.partial(plot_history, metric="categorical_accuracy", ylabel="Accuracy")


def gather_confusion_stats(
        confusion: np.array,
        process_insts: tuple[od.Process],
        input_type: str,
        stats: dict,
) -> None:
    from math import sqrt
    print(len(confusion))
    for i in range(len(confusion)):
        # labels must be in the same order as the confusion matrix
        proc_name = process_insts[i].name

        # diagonal events are True Positives (TP) or Signal (S)
        TP = S = confusion[i, i]

        # offdiagonal entries are either False Positives (FP or B) or False Negatives (FN) based on axis
        FP = B = np.sum(confusion[:, i]) - S
        FN = np.sum(confusion[i]) - S

        stats[f"precision_{input_type}_{proc_name}"] = round_sig(TP / (TP + FP), 4, float)
        stats[f"recall_{input_type}_{proc_name}"] = round_sig(TP / (TP + FN), 4, float)
        stats[f"S_over_B_{input_type}_{proc_name}"] = round_sig(S / B, 4, float)
        stats[f"S_over_sqrtB_{input_type}_{proc_name}"] = round_sig(S / sqrt(B), 4, float)


@timeit
def plot_confusion(
        model: MLModel,
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
        stats: dict | None = None,
        true_process_insts: tuple[od.Process] | None = None,
        normalize: str = "columns",
        with_title: bool = False,
        plot_postfix: str = "",
) -> None:
    """
    Simple function to create and store a confusion matrix plot
    """
    # NOTE: process_insts should always be the train_node_process_insts, otherwise y_pred is not mapped correctly
    process_insts = model.train_node_process_insts
    # use CMS plotting style but with non-quadratic figsize to avoid stretching the colorbar
    plt.style.use(mplhep.style.CMS)
    width_factor = 1.16 if with_title else 1.18
    len_factor = 1
    if true_process_insts is not None and len(true_process_insts) != len(process_insts):
        # make figures larger based on the number of processes evaluated
        # width_factor = width_factor * (len(true_process_insts) / len(process_insts))
        len_factor = len(true_process_insts) / len(process_insts)
        width_factor = width_factor * (1 + 0.16 * (len(true_process_insts) / len(process_insts) - 1))

    plt.rcParams["figure.figsize"] = (width_factor * 10, len_factor * 10)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # gather process labels
    labels = (
        [proc_inst.x("ml_label", proc_inst.label) for proc_inst in process_insts]
        if process_insts else None
    )

    num_classes = len(np.unique(inputs.labels))
    # Create confusion matrix and normalizes it over predicted (columns)
    if not true_process_insts:
        y_true = inputs.labels
        y_labels = labels
    else:
        y_true = np.ones(len(inputs.labels), dtype=int) * -1
        y_labels = []
        for i, proc_inst in enumerate(true_process_insts):
            y_labels.append(proc_inst.x("ml_label", proc_inst.label))
            mask = inputs.process_labels == proc_inst.name
            if np.any(mask):
                y_true[mask] = i
            elif proc_inst.x.ml_id != -1:
                logger.warning(
                    f"No events found for process {proc_inst.name} with ml_id "
                    f"{proc_inst.x.ml_id} in confusion matrix inputs.")
                mask = inputs.labels == proc_inst.x.ml_id
                if np.any(mask):
                    y_true[mask] = i
        y_true[y_true == -1] = len(true_process_insts)  # assign unknown processes to a separate class

    y_pred = np.argmax(inputs.prediction, axis=1)

    confusion = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=inputs.equal_weights,
    )
    if true_process_insts:
        # make confusion non-square if there are more evaluation processes than classes to predict
        confusion = confusion[:, :num_classes]

    if isinstance(stats, dict):
        gather_confusion_stats(confusion, process_insts, input_type, stats)

    # normalize confusion matrix (axis=1: over columns (predicted), axis=0: over rows (truth))
    if normalize == "columns":
        # normalize over columns (predicted)
        confusion = confusion / confusion.sum(axis=1, keepdims=True)
    elif normalize == "rows":
        # normalize over rows (truth)
        confusion = confusion / confusion.sum(axis=0, keepdims=True)
    elif normalize == "total":
        # normalize over all entries
        confusion = confusion / confusion.sum()
    else:
        logger.info(f"Confusion will not be normalized with normalize={normalize}")

    # Create a plot of the confusion matrix
    fig, ax = plt.subplots()
    if confusion.shape[0] == confusion.shape[1]:
        disp = ConfusionMatrixDisplay(confusion, display_labels=labels)
        disp.plot(ax=ax)
    else:
        im = ax.imshow(confusion, vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        for (i, j), value in np.ndenumerate(confusion):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="w")
        if labels:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels)

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=90,
        va="center",
    )

    # Add title and CMS label
    if with_title:
        ax.set_title(f"Confusion matrix for {input_type} set, rows normalized", fontsize=24, pad=+24 * 2)
    mplhep.cms.label(ax=ax, fontsize=24, loc=0, **cms_label_kwargs, com=model.config_inst.campaign.ecm)

    plt.tight_layout()
    output.child(f"Confusion_{input_type}_{plot_postfix}.pdf", type="f").dump(fig, formatter="mpl")


@timeit
def plot_roc_ovr(
        model: MLModel,
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
        stats: dict | None = None,
        weighting: str = "equal_weights",
) -> None:
    """
    Simple function to create and store some ROC plots;
    mode: OvR (one versus rest)

    NOTE: seems to be using a lot of memory, to be optimized!
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    from sklearn.metrics import roc_curve, roc_auc_score

    auc_scores = []
    n_classes = len(inputs.target[0])

    # load weights and remove negative values
    weights = np.copy(getattr(inputs, weighting))
    weights[weights < 0] = 0

    fig, ax = plt.subplots()
    for i in range(n_classes):
        y_true = (inputs.labels == i)
        fpr, tpr, thresholds = roc_curve(
            y_true=y_true,
            y_score=inputs.prediction[:, i],
            sample_weight=weights,
        )

        # to calculate the AUC score, we reduce the problem of multi-classification to a binary classification
        auc_scores.append(roc_auc_score(
            y_true=y_true,
            y_score=inputs.prediction[:, i],
            average="macro",
            multi_class="ovo",
            sample_weight=weights,
        ))

        # we could also switch to "ovr" (one versus rest) strategy, as shown in the block below
        # auc_scores.append(roc_auc_score(
        #     y_true=inputs.target,
        #     y_score=inputs.prediction,
        #     average="micro",
        #     multi_class="ovr",
        #     sample_weight=weights,
        # ))

        # create the plot
        ax.plot(fpr, tpr)

    ax.set_xlabel("Background selection efficiency")
    ax.set_ylabel("Signal selection efficiency")

    # legend
    labels = (
        [proc_inst.x("ml_label", proc_inst.label) for proc_inst in process_insts]
        if process_insts else range(n_classes)
    )
    ax.legend(
        [f"{labels[i]} (AUC: {auc_score:.2f})" for i, auc_score in enumerate(auc_scores)],
        # title=f"ROC (process vs rest), {input_type} set",
        title="ROC (process vs rest)",
        loc="lower right",
    )
    mplhep.cms.label(ax=ax, fontsize=24, loc=0, **cms_label_kwargs, com=model.config_inst.campaign.ecm)

    output.child(f"ROC_ovr_{input_type}.pdf", type="f").dump(fig, formatter="mpl")
    plt.close(fig)
    gc.collect()

    if isinstance(stats, dict):
        # append AUC scores to stats dict
        for i, auc_score in enumerate(auc_scores):
            stats[f"AUC_{input_type}_{process_insts[i].name}"] = round_sig(auc_score, 4, float)


@timeit
def plot_roc_ovo(
        model: MLModel,
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
        stats: dict | None = None,
        weighting: str = "equal_weights",
) -> None:
    """
    Simple function to create and store some ROC plots;
    mode: OvO (one versus one)

    NOTE: seems to be using a lot of memory (more than OvR), to be optimized!
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    from sklearn.metrics import roc_curve, roc_auc_score

    n_classes = len(inputs.target[0])

    labels = {
        proc_inst.x.ml_id: proc_inst.x("ml_label", proc_inst.label)
        for proc_inst in process_insts
    }

    # load weights and remove negative values
    weights = np.copy(getattr(inputs, weighting))
    weights[weights < 0] = 0

    # loop over all classes, considering each as signal for one OvO ROC curve
    for i in range(n_classes):
        auc_scores = {}
        fig, ax = plt.subplots()

        for j in range(n_classes):
            if i == j:
                continue

            event_mask = (inputs.labels == i) | (inputs.labels == j)
            y_true = (inputs.labels[event_mask] == i)
            y_score = inputs.prediction[event_mask, i]

            fpr, tpr, thresholds = roc_curve(
                y_true=y_true,
                y_score=y_score,
                sample_weight=weights[event_mask],
            )

            auc_scores[j] = roc_auc_score(
                y_true, y_score,
                average="macro", multi_class="ovo",
                sample_weight=weights[event_mask],
            )

            # create the plot
            ax.plot(fpr, tpr)

        if isinstance(stats, dict):
            # append AUC scores to stats dict
            for j, auc_score in auc_scores.items():
                auc_score = round_sig(auc_score, 4, float)
                stats[f"AUC_{input_type}_{process_insts[i].name}_vs_{process_insts[j].name}"] = auc_score

        ax.set_xlabel("Background selection efficiency (FPR)")
        ax.set_ylabel(f"{labels[i]} selection efficiency (TPR)")

        # legend
        ax.legend(
            [f"Background: {labels[j]} (AUC: {auc_score:.4f})" for j, auc_score in auc_scores.items()],
            title=f"ROC OvO, {input_type} set",
            loc="lower right",
        )
        mplhep.cms.label(ax=ax, loc=0, **cms_label_kwargs, com=model.config_inst.campaign.ecm)

        output.child(f"ROC_ovo_{process_insts[i].name}_{input_type}.pdf", type="f").dump(fig, formatter="mpl")
        plt.close(fig)
        gc.collect()


@timeit
def plot_output_nodes(
        model: MLModel,
        data: DotDict[DotDict],
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        plot_process_insts: tuple[od.Process] | None = None,
        shape_norm: bool = True,
        y_log: bool = True,
        postfix: str = "",
) -> None:
    """
    Function that creates a plot for each ML output node,
    displaying all processes per plot.
    """
    if not plot_process_insts:
        plot_process_insts = process_insts

    import hist
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    n_classes = len(list(data.values())[0].target[0])

    for i in range(n_classes):
        fig, ax = plt.subplots()

        var_title = f"{process_insts[i].x('ml_label', process_insts[i].label)} output node"

        h = (
            hist.Hist.new
            .StrCat(list(data.keys()), name="type")
            .IntCat([], name="process", growth=True, label="")
            .Reg(20, 0, 1, name=var_title)
            .Weight()
        )

        for input_type, inputs in data.items():
            for j, proc_inst in enumerate(plot_process_insts):
                mask = (inputs.process_labels == proc_inst.name)
                if proc_inst.name == "dy":
                    # hard-coded process mapping for DY
                    # (TODO: we should use process ids instead and improve process mapping as done in columnflow plots)
                    mask = mask | (inputs.process_labels == "dy_m10to50") | (inputs.process_labels == "dy_m50toinf")
                if not np.any(mask):
                    logger.warning(
                        f"No events found for process {proc_inst.name} in {input_type} set with process_labels, "
                        f"falling back to labels for masking.",
                    )
                    mask = (inputs.labels == proc_inst.x.ml_id)
                    if not np.any(mask):
                        logger.warning(
                            f"No events found for process {proc_inst.name} in {input_type} set with labels either, "
                            f"skipping this process for output node plotting.",
                        )
                        continue
                # mask = (np.argmax(inputs.target, axis=1) == j)
                fill_kwargs = {
                    "type": input_type,
                    "process": j,
                    var_title: inputs.prediction[:, i][mask],
                    "weight": inputs.weights[mask],
                }
                h.fill(**fill_kwargs)

        label = [proc_inst.label for proc_inst in plot_process_insts]
        plot_kwargs = {
            "ax": ax,
            "color": [proc_inst.color for proc_inst in plot_process_insts],
        }

        labels = {
            "train": ("Training", "solid"),
            "val": ("Validation", "dotted"),
            "test": ("Test", "dashed"),
        }

        # dummy legend entries
        for input_type in data.keys():
            plt.hist([], histtype="step", label=labels[input_type][0], linestyle=labels[input_type][1], color="black")

        # get the correct normalization factors
        scale_factors = {}
        for input_type, inputs in data.items():
            scale_factors[input_type] = np.array([
                # hist.loc(i) ?
                h[{"type": input_type, "process": i}].sum().value for i in range(len(plot_process_insts))
            ])[:, np.newaxis]
        keys = list(scale_factors.keys())
        if not shape_norm:
            base_factor = scale_factors[keys[0]]
            scale_factors[keys[0]] = 1
            for key in keys[1:]:
                scale_factors[key] = base_factor / scale_factors[key]

        # plot "first" dataset
        (h[{"type": keys[0]}] / scale_factors[keys[0]]).plot1d(**plot_kwargs, label=label, linestyle=labels[keys[0]][1])

        # axis styling
        ax_kwargs = {
            "ylabel": r"$\Delta N/N$" if shape_norm else "Entries",
            "xlim": (0, 1),
            "yscale": "log" if y_log else "linear",
        }
        # set y_lim to appropriate ranges based on the yscale
        magnitudes = 4
        whitespace_fraction = 0.3
        ax_ymin = ax.get_ylim()[1] / 10**magnitudes if y_log else 0.0000001
        ax_ymax = get_position(ax_ymin, ax.get_ylim()[1], factor=1 / (1 - whitespace_fraction), logscale=y_log)

        ax_kwargs["ylim"] = (ax_ymin, ax_ymax)

        ax.set(**ax_kwargs)

        # plot validation scores, scaled to train dataset
        for key in keys[1:]:
            (h[{"type": key}] / scale_factors[key]).plot1d(
                **plot_kwargs,
                linestyle=labels[key][1],
                label="_nolegend_",
            )

        # legend
        from math import ceil
        ncols = 2
        num_entries_per_col = ceil(len(plot_process_insts) + 3 / ncols)
        handles, labels = ax.get_legend_handles_labels()
        empty_handle = ax.plot([], label="", linestyle="None")[0]
        if num_entries_per_col > 3:
            for _ in range(num_entries_per_col - 4):
                handles.insert(3, empty_handle)
                labels.insert(3, "")
        ax.legend(handles=handles, labels=labels, loc="best", ncols=2, title="")

        mplhep.cms.label(ax=ax, loc=0, **cms_label_kwargs, com=model.config_inst.campaign.ecm, fontsize=24)
        output.child(f"Node_{process_insts[i].name}{postfix}.pdf", type="f").dump(fig, formatter="mpl")


@timeit
def plot_input_features(
        model: MLModel,
        train: DotDict,
        validation: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        shape_norm: bool = True,
        y_log: bool = True,
):
    """
    Function that creates a plot for each ML input feature, displaying all processes per plot.
    """
    import hist

    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    n_processes = len(process_insts)
    input_features = model.input_features_ordered

    for i, feature_name in enumerate(input_features):
        fig, ax = plt.subplots()

        variable_inst = model.config_inst.get_variable(feature_name, default=None)
        if not variable_inst:
            logger.warning(f"Could not get variable instance for {feature_name}, skipping")
            continue

        h = (
            hist.Hist.new
            .StrCat(["train", "validation"], name="type")
            .IntCat([], name="process", growth=True, label="")
            .Var(variable_inst.bin_edges, name=feature_name, label=variable_inst.get_full_x_title())
            .Weight()
        )

        for input_type, inputs in (("train", train), ("validation", validation)):
            for j in range(n_processes):
                mask = (inputs.labels == j)
                fill_kwargs = {
                    "type": input_type,
                    "process": j,
                    feature_name: inputs.features[:, i][mask],
                    "weight": inputs.weights[mask],
                }
                h.fill(**fill_kwargs)

        label = [proc_inst.label for proc_inst in process_insts]
        plot_kwargs = {
            "ax": ax,
            "color": [proc_inst.color for proc_inst in process_insts],
        }

        # dummy legend entries
        plt.hist([], histtype="step", label="Training", color="black")
        plt.hist([], histtype="step", label="Validation", linestyle="dotted", color="black")

        # get the correct normalization factors
        if shape_norm:
            scale_train = np.array([
                h[{"type": "train", "process": i}].sum().value for i in range(n_processes)
            ])[:, np.newaxis]
            scale_val = np.array([
                h[{"type": "validation", "process": i}].sum().value for i in range(n_processes)
            ])[:, np.newaxis]
        else:
            scale_train = 1
            scale_val = h[{"type": "train"}].sum().value / h[{"type": "validation"}].sum().value

        # plot training scores
        (h[{"type": "train"}] / scale_train).plot1d(**plot_kwargs, label=label)

        # axis styling
        ax_kwargs = {
            "ylabel": r"$\Delta N/N$" if shape_norm else "Entries",
            "xlim": (variable_inst.x_min, variable_inst.x_max),
            "yscale": "log" if y_log else "linear",
        }
        # set y_lim to appropriate ranges based on the yscale
        y_max = ax.get_ylim()[1]
        if y_log:
            ax_kwargs["ylim"] = (y_max * 1e-4, y_max * 2)
        else:
            ax_kwargs["ylim"] = (0.00001, y_max)

        ax.set(**ax_kwargs)

        # plot validation scores, scaled to train dataset
        (h[{"type": "validation"}] / scale_val).plot1d(
            **plot_kwargs,
            linestyle="dotted",
            label="_nolegend_",
        )

        # legend
        ax.legend(loc="best", title="")

        mplhep.cms.label(ax=ax, loc=0, **cms_label_kwargs, com=model.config_inst.campaign.ecm)
        try:
            output.child(f"Input_{feature_name}.pdf", type="f").dump(fig, formatter="mpl")
        except Exception:
            logger.warning(f"Feature {feature_name} plot does not like to be stored for some reason?")


def get_input_weights(model, output, input_features: list | None = None):
    """
    Get weights of input layer and sort them by weight sum
    """
    if not input_features:
        input_features = tuple(
            output.sibling("", type="d").child("input_features.pkl", type="f").load(formatter="pickle"),
        )

    # get the weights from the first dense layer
    for layer in model.layers:
        if "Dense" in str(type(layer)):
            weights = layer.get_weights()[0]
            break

    # check that the input shape is correct
    if weights.shape[0] != len(input_features):
        raise Exception(
            f"The number of weights {weights.shape[0]} in the first denes layer should be equivalent "
            f"to the numberof input features {len(input_features)}",
        )

    # sum weights per variable and round
    my_dict = {}
    for out_weights, variable in zip(weights, input_features):
        w_sum = np.sum(np.abs(out_weights))
        my_dict[variable] = round(float(w_sum), ndigits=3)

    # sort variables based on importance and print + dump
    variable_importance_sorted = dict(sorted(my_dict.items(), key=lambda item: item[1], reverse=True))
    for var_name, score in variable_importance_sorted.items():
        print(f"{var_name}: {score}")

    output.child("weights_first_layer.yaml", type="f").dump(
        variable_importance_sorted, formatter="yaml", sort_keys=False,
    )
