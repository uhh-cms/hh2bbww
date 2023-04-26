# coding: utf-8

from columnflow.util import maybe_import


plt = maybe_import("matplotlib.pyplot")


def plot_loss(history, output) -> None:
    """
    Simple function to plot and store a loss plot
    """

    fig, ax = plt.subplots()
    ax.plot(history["loss"])
    ax.plot(history["val_loss"])
    ax.set(**{
        "ylabel": "Loss",
        "xlabel": "Epoch",
    })
    ax.legend(["train", "validation"], loc="best")

    output.child("Loss.pdf", type="f").dump(fig, formatter="mpl")


def plot_accuracy(history, output) -> None:
    """
    Simple function to plot and store an accuracy plot
    """

    fig, ax = plt.subplots()
    ax.plot(history["categorical_accuracy"])
    ax.plot(history["val_categorical_accuracy"])
    ax.set(**{
        "ylabel": "Loss",
        "xlabel": "Epoch",
    })
    ax.legend(["train", "validation"], loc="best")

    output.child("Accuracy.pdf", type="f").dump(fig, formatter="mpl")
