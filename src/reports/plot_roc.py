# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 01:17:02 2021

@author: Artem
"""
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

from sklearn.metrics import auc


import warnings

warnings.filterwarnings("ignore")

SEED = 42


def plot_roc_curve(fprs, tprs):

    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(15, 15))

    # Построение графика ROC для каждого сгиба и вычисление score AUC
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs), 1):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label="ROC Fold {} (AUC = {:.3f})".format(i, roc_auc),
        )

    # Plotting ROC for random guessing
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=2,
        color="r",
        alpha=0.8,
        label="Random Guessing",
    )

    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plotting the mean ROC
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label="Mean ROC (AUC = {:.3f} $\pm$ {:.3f})".format(mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    # Plotting the standard deviation around the mean ROC Curve
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label="$\pm$ 1 std. dev.",
    )

    ax.set_xlabel("False Positive Rate", size=15, labelpad=20)
    ax.set_ylabel("True Positive Rate", size=15, labelpad=20)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    ax.set_title("ROC Curves of Folds", size=20, y=1.02)
    ax.legend(loc="lower right", prop={"size": 13})

    plt.show()
