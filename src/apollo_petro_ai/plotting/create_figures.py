"""Scripts to create figures for the paper."""

import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve


def make_graphs_from_csv(
    plot_data: List[Any], descriptors: List[str], ft_start: int = -1
) -> None:
    """Plotting some graphs.

    Args:
        plot_data: list of what to plot
        descriptors: descriptor of what to plot
        ft_start: where fine-tuning started

    Returns:
        plot of training and validation curves
    """
    prefix = "../plots"
    for i, info in enumerate(descriptors):
        plot_paths = plot_data[i]
        train_path = os.path.join(prefix, plot_paths[0])
        val_path = os.path.join(prefix, plot_paths[1])

        train_gr = pd.read_csv(train_path, delimiter=";")
        val_gr = pd.read_csv(val_path, delimiter=";")
        loc = 1
        if info == "Accuracy":
            loc = 2

        plt.plot(
            train_gr.loc[:, "Epoch"],
            train_gr.loc[:, "Value"],
            color="#0736c4",
            label="Train",
        )
        plt.plot(
            val_gr.loc[:, "Epoch"],
            val_gr.loc[:, "Value"],
            color="#e0270b",
            label="Validation",
        )
        if ft_start > 0:
            plt.axvline(ft_start, color="#328758", label="Start Fine Tuning")
        plt.xlabel("Epoch")
        plt.ylabel(f"{info}")
        plt.title(f"{info} over epochs")
        plt.legend(loc=loc)
        plt.show()


def plot_confusion(
    c_m: np.array, title: str = "Confusion Matrix of Rock Type Classification"
) -> None:
    """Plots confusion matrix c_m.

    Args:
        c_m: confusion matrix to plot
        title: title of the confusion matrix

    Returns:
        show confusion matrix
    """
    # Select Confusion Matrix Size
    plt.figure(figsize=(6, 6), dpi=300)

    # Set names to show in boxes
    classes = [
        "Correctly predicted as Basalt\n (TP)",
        "Incorrectly predicted as Breccia\n (FP)",
        "Incorrectly predicted as Basalt\n (FN)",
        "Correctly predicted as Breccia\n (TN)",
    ]

    # Set values format
    values = ["{0:0.0f}".format(x) for x in c_m.flatten()]

    # Combine classes and values to show
    combined = [f"{i}\n{j}" for i, j in zip(classes, values)]
    combined = np.asarray(combined).reshape(2, 2)

    # Create Confusion Matrix
    b = sns.heatmap(c_m, annot=combined, fmt="", cmap="rocket", cbar=False)
    b.set(title=title)
    b.set(xticklabels=["Basalt", "Breccia"], yticklabels=["Basalt", "Breccia"])
    b.set(xlabel="Predicted", ylabel="Actual")

    plt.show()


def draw_PR_curve(
    target: np.array, predictions: np.array, title: str = "PR Curve"
) -> None:
    """Draws a precision recall curve.

    Args:
        target: the prediction targets
        predictions: the predictions made by the network
        title: title to give to the plot

    Returns:
        Does not return, but plots graph instead
    """
    precision, recall, _ = precision_recall_curve(target, predictions)
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    make_graphs_from_csv(
        [
            ["tl_train_loss.csv", "tl_val_loss.csv"],
            ["tl_train_ac.csv", "tl_val_ac.csv"],
        ],
        ["Loss", "Accuracy"],
        20,
    )
    # make_graphs_from_csv([['simple_cnn_train.csv', 'simple_cnn_val.csv']],
    # ['Accuracy'])
