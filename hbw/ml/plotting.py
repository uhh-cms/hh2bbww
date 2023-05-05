# coding: utf-8

from __future__ import annotations

import law
import order as od

from columnflow.util import maybe_import, DotDict


np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
hist = maybe_import("hist")
tf = maybe_import("tensorflow")


def plot_loss(history, output) -> None:
    """
    Simple function to create and store a loss plot
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    fig, ax = plt.subplots()
    ax.plot(history["loss"])
    ax.plot(history["val_loss"])
    ax.set(**{
        "ylabel": "Loss",
        "xlabel": "Epoch",
    })
    ax.legend(["train", "validation"], loc="best")
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False)

    output.child("Loss.pdf", type="f").dump(fig, formatter="mpl")


def plot_accuracy(history, output) -> None:
    """
    Simple function to create and store an accuracy plot
    """
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    fig, ax = plt.subplots()
    ax.plot(history["categorical_accuracy"])
    ax.plot(history["val_categorical_accuracy"])
    ax.set(**{
        "ylabel": "Accuracy",
        "xlabel": "Epoch",
    })
    ax.legend(["train", "validation"], loc="best")
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False)

    output.child("Accuracy.pdf", type="f").dump(fig, formatter="mpl")


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
        y_true=np.argmax(inputs.target, axis=1),
        y_pred=np.argmax(inputs.prediction, axis=1),
        sample_weight=inputs.weights,
        normalize="true",
    )

    labels = [proc_inst.label for proc_inst in process_insts] if process_insts else None

    # Create a plot of the confusion matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion, display_labels=labels).plot(ax=ax)

    ax.set_title(f"Confusion matrix for {input_type} set, rows normalized", fontsize=20)
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=2)

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
        fpr, tpr, thresholds = roc_curve(
            y_true=inputs.target[:, i],
            y_score=inputs.prediction[:, i],
            sample_weight=inputs.weights,
        )

        auc_scores.append(roc_auc_score(
            inputs.target[:, i], inputs.prediction[:, i],
            average="macro", multi_class="ovr",
        ))

        # create the plot
        ax.plot(fpr, tpr)

    ax.set_title(f"ROC OvR, {input_type} set")
    ax.set_xlabel("Background selection efficiency (FPR)")
    ax.set_ylabel("Signal selection efficiency (TPR)")

    # legend
    labels = [proc_inst.label for proc_inst in process_insts] if process_insts else range(n_classes)
    ax.legend(
        [f"Signal: {labels[i]} (AUC: {auc_score:.4f})" for i, auc_score in enumerate(auc_scores)],
        loc="best",
    )
    mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=2)

    output.child(f"ROC_ovr_{input_type}.pdf", type="f").dump(fig, formatter="mpl")


def plot_output_nodes(
        model: tf.keras.models.Model,
        train: DotDict,
        validation: DotDict,
        output: law.FileSystemDirectoryTarget,
        process_insts: tuple[od.Process],
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

        var_title = f"Output node {process_insts[i].label}"

        h = (
            hist.Hist.new
            .StrCat(["train", "validation"], name="type")
            .IntCat([], name="process", growth=True)
            .Reg(20, 0, 1, name=var_title)
            .Weight()
        )

        for input_type, inputs in (("train", train), ("validation", validation)):
            for j in range(n_classes):
                mask = (np.argmax(inputs.target, axis=1) == j)
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
        plt.hist([], histtype="step", label="Validation (scaled)", linestyle="dotted", color="black")

        # plot training scores
        h[{"type": "train"}].plot1d(**plot_kwargs)

        # legend
        ax.legend(loc="best")

        ax.set(**{
            "ylabel": "Entries",
            "ylim": (0.00001, ax.get_ylim()[1]),
            "xlim": (0, 1),
        })

        # plot validation scores, scaled to train dataset
        scale = h[{"type": "train"}].sum().value / h[{"type": "validation"}].sum().value
        (h[{"type": "validation"}] * scale).plot1d(**plot_kwargs, linestyle="dotted")

        mplhep.cms.label(ax=ax, llabel="Work in progress", data=False, loc=0)
        output.child(f"Node_{process_insts[i].name}.pdf", type="f").dump(fig, formatter="mpl")
