# coding: utf-8

from __future__ import annotations

import functools

# import tabulate
import law
import order as od

from columnflow.util import maybe_import, DotDict


np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
hist = maybe_import("hist")
tf = maybe_import("tensorflow")


def plot_history(
    history,
    output: law.FileSystemDirectoryTarget,
    metric: str = "loss",
    ylabel: str | None = None,
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
        "xlabel": "Epoch",
    })
    ax.legend(["train", "validation"], loc="best")
    mplhep.cms.label(ax=ax, llabel="Simulation Work in progress", data=False)

    plt.tight_layout()
    output.child(f"{output_name}.pdf", type="f").dump(fig, formatter="mpl")


plot_loss = functools.partial(plot_history, metric="loss", ylabel="Loss")
plot_accuracy = functools.partial(plot_history, metric="categorical_accuracy", ylabel="Accuracy")


def plot_confusion(
        model: tf.keras.models.Model,
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
) -> None:
    """
    Simple function to create and store a confusion matrix plot
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Create confusion matrix and normalizes it over predicted (columns)
    confusion = confusion_matrix(
        y_true=inputs.label,
        y_pred=np.argmax(inputs.prediction, axis=1),
        sample_weight=inputs.weights,
        normalize="true",
    )
    # legend
    labels = (
        [proc_inst.x("ml_label", proc_inst.label) for proc_inst in process_insts]
        if process_insts else None
    )

    # Create a plot of the confusion matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion, display_labels=labels).plot(ax=ax)

    ax.set_title(f"Confusion matrix for {input_type} set, rows normalized", fontsize=20, pad=+40)
    mplhep.cms.label(ax=ax, llabel="Simulation Work in progress", data=False, loc=0)

    plt.tight_layout()
    output.child(f"Confusion_{input_type}.pdf", type="f").dump(fig, formatter="mpl")


def plot_roc_ovr(
        model: tf.keras.models.Model,
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
) -> None:
    """
    Simple function to create and store some ROC plots;
    mode: OvR (one versus rest)
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    auc_scores = []
    n_classes = len(inputs.target[0])

    fig, ax = plt.subplots()
    for i in range(n_classes):
        y_true = (inputs.label == i)
        fpr, tpr, thresholds = roc_curve(
            y_true=y_true,
            # y_true=inputs.target[:, i],
            y_score=inputs.prediction[:, i],
            sample_weight=inputs.weights,
        )

        auc_scores.append(roc_auc_score(
            y_true, inputs.prediction[:, i],
            average="macro", multi_class="ovr",
        ))

        # create the plot
        ax.plot(fpr, tpr)

    ax.set_title(f"ROC OvR, {input_type} set")
    ax.set_xlabel("Background selection efficiency (FPR)")
    ax.set_ylabel("Signal selection efficiency (TPR)")

    # legend
    labels = (
        [proc_inst.x("ml_label", proc_inst.label) for proc_inst in process_insts]
        if process_insts else range(n_classes)
    )
    ax.legend(
        [f"Signal: {labels[i]} (AUC: {auc_score:.4f})" for i, auc_score in enumerate(auc_scores)],
        loc="lower right",
    )
    mplhep.cms.label(ax=ax, llabel="Simulation\nWork in progress", data=False, loc=2)

    output.child(f"ROC_ovr_{input_type}.pdf", type="f").dump(fig, formatter="mpl")


def plot_roc_ovo(
        model: tf.keras.models.Model,
        inputs: DotDict,
        output: law.FileSystemDirectoryTarget,
        input_type: str,
        process_insts: tuple[od.Process],
) -> None:
    """
    Simple function to create and store some ROC plots;
    mode: OvO (one versus one)
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    n_classes = len(inputs.target[0])

    labels = {
        proc_inst.x.ml_id: proc_inst.x("ml_label", proc_inst.label)
        for proc_inst in process_insts
    }

    # loop over all classes, considering each as signal for one OvO ROC curve
    for i in range(n_classes):
        auc_scores = {}
        fig, ax = plt.subplots()

        for j in range(n_classes):
            if i == j:
                continue

            event_mask = (inputs.label == i) | (inputs.label == j)
            y_true = (inputs.label[event_mask] == i)
            y_score = inputs.prediction[event_mask, i]

            fpr, tpr, thresholds = roc_curve(
                y_true=y_true,
                y_score=y_score,
                sample_weight=inputs.weights[event_mask],
            )

            auc_scores[j] = roc_auc_score(
                y_true, y_score,
                average="macro", multi_class="ovo",
            )

            # create the plot
            ax.plot(fpr, tpr)

        ax.set_title(f"ROC OvO, {input_type} set")
        ax.set_xlabel("Background selection efficiency (FPR)")
        ax.set_ylabel(f"{labels[i]} selection efficiency (TPR)")

        # legend
        ax.legend(
            [f"Background: {labels[j]} (AUC: {auc_score:.4f})" for j, auc_score in auc_scores.items()],
            loc="lower right",
        )
        mplhep.cms.label(ax=ax, llabel="Simulation\nWork in progress", data=False, loc=2)

        output.child(f"ROC_ovo_{process_insts[i].name}_{input_type}.pdf", type="f").dump(fig, formatter="mpl")


def plot_output_nodes(
        model: tf.keras.models.Model,
        train: DotDict,
        validation: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
        shape_norm: bool = True,
        y_log: bool = False,
) -> None:
    """
    Function that creates a plot for each ML output node,
    displaying all processes per plot.
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    n_classes = len(train.target[0])

    for i in range(n_classes):
        fig, ax = plt.subplots()

        var_title = f"{process_insts[i].x('ml_label', process_insts[i].label)} output node"

        h = (
            hist.Hist.new
            .StrCat(["train", "validation"], name="type")
            .IntCat([], name="process", growth=True)
            .Reg(20, 0, 1, name=var_title)
            .Weight()
        )

        for input_type, inputs in (("train", train), ("validation", validation)):
            for j in range(n_classes):
                mask = (inputs.label == j)
                # mask = (np.argmax(inputs.target, axis=1) == j)
                fill_kwargs = {
                    "type": input_type,
                    "process": j,
                    var_title: inputs.prediction[:, i][mask],
                    "weight": inputs.weights[mask],
                }
                h.fill(**fill_kwargs)

        plot_kwargs = {
            "ax": ax,
            "label": [proc_inst.label for proc_inst in process_insts],
            "color": [proc_inst.color for proc_inst in process_insts],
        }

        # dummy legend entries
        plt.hist([], histtype="step", label="Training", color="black")
        plt.hist([], histtype="step", label="Validation", linestyle="dotted", color="black")

        # get the correct normalization factors
        if shape_norm:
            scale_train = np.array([
                h[{"type": "train", "process": i}].sum().value for i in range(n_classes)
            ])[:, np.newaxis]
            scale_val = np.array([
                h[{"type": "validation", "process": i}].sum().value for i in range(n_classes)
            ])[:, np.newaxis]
        else:
            scale_train = 1
            scale_val = h[{"type": "train"}].sum().value / h[{"type": "validation"}].sum().value

        # plot training scores
        (h[{"type": "train"}] / scale_train).plot1d(**plot_kwargs)

        # legend
        ax.legend(loc="best")

        # axis styling
        ax_kwargs = {
            "ylabel": "Entries",
            "xlim": (0, 1),
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
        (h[{"type": "validation"}] / scale_val).plot1d(**plot_kwargs, linestyle="dotted")

        mplhep.cms.label(ax=ax, llabel="Simulation Work in progress", data=False, loc=0)
        output.child(f"Node_{process_insts[i].name}.pdf", type="f").dump(fig, formatter="mpl")


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
