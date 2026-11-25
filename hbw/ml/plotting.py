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

if TYPE_CHECKING:
    hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


cms_label_kwargs = {
    "data": False,
    "llabel": "Private work (CMS simulation)",
    # "llabel": "Simulation work in progress",
    # "llabel": "Simulation preliminary",
    "lumi": "109.8",  # NOTE: hard-coded, to be updated if needed
    # "exp": "",
}
if "CMS" in cms_label_kwargs["llabel"]:
    cms_label_kwargs["exp"] = ""


def get_cms_label_kwargs(model: MLModel) -> dict:
    label_kwargs = cms_label_kwargs.copy()
    label_kwargs["lumi"] = round(
        0.001 * sum(
            config_inst.x.luminosity.get("nominal")
            for config_inst in model.config_insts
        ),
        1,
    )
    return label_kwargs


def barplot_from_multidict(dict_of_rankings: dict[str, dict], normalize_weights: bool = True):
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

    ax.set_xlabel("Contribution")
    ax.set_ylabel("Input features")
    ax.set_yticks(index - (bar_width * (num_dicts - 1)) / 2)
    ax.set_yticklabels(labels)
    ax.legend()

    plt.tight_layout()

    return fig, ax


@timeit
def plot_introspection(
    model: MLModel,
    output: law.FileSystemDirectoryTarget,
    inputs,
    output_node: int = 0,
    input_features: list | None = None,
    stats: dict | None = None,
):
    from hbw.ml.introspection import sensitivity_analysis, gradient_times_input, shap_ranking

    # get only signal events for now
    inputs = inputs.features[inputs.labels == 0]

    shap_ranking_dict, shap_values = shap_ranking(model.trained_models[0], inputs, output_node, input_features)

    rankings = {
        "SHAP": shap_ranking_dict,
        "Sensitivity Analysis": sensitivity_analysis(model.trained_models[0], inputs, output_node, input_features),
        "Gradient * Input": gradient_times_input(model.trained_models[0], inputs, output_node, input_features),
    }
    # TODO: dump rankings in stats json (need to convert float32 into str for json compatibility)
    # if stats:
    #     stats["rankings"] = rankings
    fig, ax = barplot_from_multidict(rankings)

    output.child("rankings.pdf", type="f").dump(fig, formatter="mpl")
    return fig, ax


@timeit
def plot_history(
    history,
    output: law.FileSystemDirectoryTarget,
    metric: str = "loss",
    ylabel: str | None = None,
    yscale: str = "linear",
    output_name: str | None = None,
    config_insts=None,
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
    label_kwargs = cms_label_kwargs.copy()
    if config_insts is not None:
        label_kwargs["lumi"] = round(
            0.001 * sum(config_inst.x.luminosity.get("nominal") for config_inst in config_insts),
            1,
        )
    mplhep.cms.label(ax=ax, fontsize=22, **label_kwargs, com=13.6)

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
        normalize: str = "columns",
        with_title: bool = False,
        weighting: str = "equal_weights",
) -> None:
    """
    Simple function to create and store a confusion matrix plot
    """
    # use CMS plotting style but with non-quadratic figsize to avoid stretching the colorbar
    plt.style.use(mplhep.style.CMS)
    plt.rcParams["figure.figsize"] = (11.8, 10)
    if with_title:
        plt.rcParams["figure.figsize"] = (11.6, 10)  # with title

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Create confusion matrix and normalizes it over predicted (columns)
    weights = getattr(inputs, weighting)
    if weights is None:
        raise ValueError(f"Unknown weighting option: {weighting}")

    predictions = inputs.prediction
    if predictions.ndim == 2:
        predictions_by_member = predictions[:, :, np.newaxis]
    else:
        predictions_by_member = predictions

    n_members = predictions_by_member.shape[-1]
    confusion_matrices = []

    for member in range(n_members):
        confusion = confusion_matrix(
            y_true=inputs.labels,
            y_pred=np.argmax(predictions_by_member[:, :, member], axis=1),
            sample_weight=weights,
        )
        if member == 0:
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

        confusion_matrices.append(confusion)

    confusion_mean = np.mean(confusion_matrices, axis=0)
    if len(confusion_matrices) > 1:
        confusion_std = np.std(confusion_matrices, axis=0)
    else:
        confusion_std = None
    # gather process labels
    labels = (
        [proc_inst.x("ml_label", proc_inst.label) for proc_inst in process_insts]
        if process_insts else None
    )

    # Create a plot of the confusion matrix
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_mean, display_labels=labels)
    disp.plot(ax=ax)

    if confusion_std is not None:
        for index, text in np.ndenumerate(disp.text_):
            mean_value = confusion_mean[index]
            std_value = confusion_std[index]
            text.set_text(f"{mean_value:.2f}\n$\\pm$ {std_value:.2f}")

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=90,
        va="center",
    )

    # Add title and CMS label
    if with_title:
        ax.set_title(f"Confusion matrix for {input_type} set, rows normalized", fontsize=24, pad=+24 * 2)
    mplhep.cms.label(ax=ax, fontsize=24, loc=0, **get_cms_label_kwargs(model), com=model.config_inst.campaign.ecm)

    plt.tight_layout()
    output.child(f"Confusion_{input_type}.pdf", type="f").dump(fig, formatter="mpl")


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
    auc_stds = []
    n_classes = len(inputs.target[0])

    # Shape:
    # normal:   (n_events, n_classes)
    # ensemble: (n_events, n_classes, n_members)
    predictions = inputs.prediction
    if predictions.ndim == 2:
        predictions = predictions[:, :, np.newaxis]

    n_members = predictions.shape[-1]

    # load weights and remove negative values
    weights = np.copy(getattr(inputs, weighting))
    weights[weights < 0] = 0

    fig, ax = plt.subplots()
    fpr_grid = np.linspace(0.0, 1.0, 1001)

    for i in range(n_classes):
        y_true = (inputs.labels == i)
        member_tprs = []
        member_aucs = []

        for member in range(n_members):
            member_predictions = predictions[:, i, member]

            fpr, tpr, thresholds = roc_curve(
                y_true=y_true,
                y_score=member_predictions,
                sample_weight=weights,
            )
            member_aucs.append(roc_auc_score(
                y_true=y_true,
                y_score=member_predictions,
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

            # Interpolate this member's ROC curve on the common FPR grid.
            interpolated_tpr = np.interp(fpr_grid, fpr, tpr)
            interpolated_tpr[0] = 0.0
            interpolated_tpr[-1] = 1.0
            member_tprs.append(interpolated_tpr)

        member_tprs = np.asarray(member_tprs)
        member_aucs = np.asarray(member_aucs)

        mean_tpr = np.mean(member_tprs, axis=0)
        mean_auc = float(np.mean(member_aucs))

        if n_members > 1:
            std_tpr = np.std(member_tprs, axis=0)
            std_auc = float(np.std(member_aucs, ddof=0))
        else:
            std_tpr = None
            std_auc = 0.0

        auc_scores.append(mean_auc)
        auc_stds.append(std_auc)

        # create the plot
        ax.plot(fpr_grid, mean_tpr)
        if n_members > 1:
            ax.fill_between(fpr_grid, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2, label="_nolegend_")

    ax.set_xlabel("Background selection efficiency")
    ax.set_ylabel("Signal selection efficiency")

    # legend
    labels = (
        [proc_inst.x("ml_label", proc_inst.label) for proc_inst in process_insts]
        if process_insts else range(n_classes)
    )
    if n_members == 1:
        ax.legend(
            [f"{labels[i]} (AUC: {auc_score:.4f})" for i, auc_score in enumerate(auc_scores)],
            # title=f"ROC (process vs rest), {input_type} set",
            title="ROC (process vs rest)",
            loc="lower right",
        )
    else:
        ax.legend(
            [
                f"{labels[i]} (AUC: {auc_score:.4f} ± {auc_std:.4f})"
                for i, (auc_score, auc_std) in enumerate(zip(auc_scores, auc_stds))
            ],
            # title=f"ROC (process vs rest), {input_type} set",
            title="ROC (process vs rest)",
            loc="lower right",
        )
    mplhep.cms.label(ax=ax, loc=0, fontsize=22, **get_cms_label_kwargs(model), com=model.config_inst.campaign.ecm)

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

    predictions = inputs.prediction
    if predictions.ndim == 2:
        predictions = predictions[:, :, np.newaxis]

    n_members = predictions.shape[-1]

    # load weights and remove negative values
    weights = np.copy(getattr(inputs, weighting))
    weights[weights < 0] = 0

    fpr_grid = np.linspace(0.0, 1.0, 1001)

    # loop over all classes, considering each as signal for one OvO ROC curve
    for i in range(n_classes):
        auc_scores = {}
        auc_stds = {}

        fig, ax = plt.subplots()

        for j in range(n_classes):
            if i == j:
                continue

            event_mask = (inputs.labels == i) | (inputs.labels == j)
            y_true = (inputs.labels[event_mask] == i)
            member_tprs = []
            member_aucs = []

            for member in range(n_members):
                y_score = predictions[event_mask, i, member]

                fpr, tpr, thresholds = roc_curve(
                    y_true=y_true,
                    y_score=y_score,
                    sample_weight=weights[event_mask],
                )

                member_aucs.append(roc_auc_score(
                    y_true, y_score,
                    average="macro", multi_class="ovo",
                    sample_weight=weights[event_mask],
                ))

                interpolated_tpr = np.interp(
                    fpr_grid,
                    fpr,
                    tpr,
                )
                interpolated_tpr[0] = 0.0
                interpolated_tpr[-1] = 1.0
                member_tprs.append(interpolated_tpr)

            member_tprs = np.asarray(member_tprs)
            member_aucs = np.asarray(member_aucs)

            mean_tpr = np.mean(member_tprs, axis=0)
            mean_auc = float(np.mean(member_aucs))

            if n_members > 1:
                std_tpr = np.std(member_tprs, axis=0)
                std_auc = float(np.std(member_aucs))

            auc_scores[j] = mean_auc
            auc_stds[j] = std_auc

            # create the plot
            ax.plot(fpr_grid, mean_tpr)
            if n_members > 1:
                ax.fill_between(fpr_grid, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2, label="_nolegend_")

        if isinstance(stats, dict):
            # append AUC scores to stats dict
            for j, auc_score in auc_scores.items():
                auc_score = round_sig(auc_score, 4, float)
                stats[f"AUC_{input_type}_{process_insts[i].name}_vs_{process_insts[j].name}"] = auc_score

        ax.set_xlabel("Background selection efficiency (FPR)")
        ax.set_ylabel(f"{labels[i]} selection efficiency (TPR)")

        # legend
        if n_members == 1:
            ax.legend(
                [f"Background: {labels[j]} (AUC: {auc_score:.4f})" for j, auc_score in auc_scores.items()],
                title=f"ROC OvO, {input_type} set",
                loc="lower right",
            )
        else:
            ax.legend(
                [
                    f"Background: {labels[j]} (AUC: {auc_scores[j]:.4f} ± {auc_stds[j]:.4f})"
                    for j, auc_score in auc_scores.items()
                ],
                title=f"ROC OvO, {input_type} set",
                loc="lower right",
            )
        mplhep.cms.label(ax=ax, loc=0, fontsize=22, **get_cms_label_kwargs(model), com=model.config_inst.campaign.ecm)

        output.child(f"ROC_ovo_{process_insts[i].name}_{input_type}.pdf", type="f").dump(fig, formatter="mpl")
        plt.close(fig)
        gc.collect()


@timeit
def plot_output_nodes(
        model: MLModel,
        data: DotDict[DotDict],
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        shape_norm: bool = True,
        y_log: bool = True,
) -> None:
    """
    Function that creates a plot for each ML output node,
    displaying all processes per plot.
    """
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
            predictions = inputs.prediction if inputs.prediction.ndim == 2 else inputs.prediction[:, :, 0]
            for j in range(n_classes):
                mask = (inputs.labels == j)
                # mask = (np.argmax(inputs.target, axis=1) == j)
                fill_kwargs = {
                    "type": input_type,
                    "process": j,
                    var_title: predictions[:, i][mask],
                    "weight": inputs.weights[mask],
                }
                h.fill(**fill_kwargs)

        label = [proc_inst.label for proc_inst in process_insts]
        plot_kwargs = {
            "ax": ax,
            "color": [proc_inst.color for proc_inst in process_insts],
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
                h[{"type": input_type, "process": i}].sum().value for i in range(n_classes)
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
        ax.legend(loc="best", ncols=2, title="")

        mplhep.cms.label(ax=ax, loc=0, fontsize=22, **get_cms_label_kwargs(model), com=model.config_inst.campaign.ecm)
        output.child(f"Node_{process_insts[i].name}.pdf", type="f").dump(fig, formatter="mpl")


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

        mplhep.cms.label(ax=ax, loc=0, fontsize=22, **get_cms_label_kwargs(model), com=model.config_inst.campaign.ecm)
        try:
            output.child(f"Input_{feature_name}.pdf", type="f").dump(fig, formatter="mpl")
        except Exception:
            logger.warning(f"Feature {feature_name} plot does not like to be stored for some reason?")
        plt.close(fig)


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
