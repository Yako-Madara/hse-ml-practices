import numpy as np
import pandas as pd

import seaborn as sns

sns.set(style="darkgrid")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import warnings

warnings.filterwarnings("ignore")

SEED = 42


def modRandFor(df_all, df_train, df_test, drop_cols):
    X_train = StandardScaler().fit_transform(df_train.drop(columns=drop_cols))
    y_train = df_train["Survived"].values
    X_test = StandardScaler().fit_transform(df_test.drop(columns=drop_cols))

    print("X_train shape: {}".format(X_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("X_test shape: {}".format(X_test.shape))

    # 3.1 Случайный лес
    single_best_model = RandomForestClassifier(
        criterion="gini",
        n_estimators=1100,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=5,
        max_features="auto",
        oob_score=True,
        random_state=SEED,
        n_jobs=-1,
        verbose=1,
    )

    leaderboard_model = RandomForestClassifier(
        criterion="gini",
        n_estimators=1750,
        max_depth=7,
        min_samples_split=6,
        min_samples_leaf=6,
        max_features="auto",
        oob_score=True,
        random_state=SEED,
        n_jobs=-1,
        verbose=1,
    )
    N = 5
    oob = 0
    probs = pd.DataFrame(
        np.zeros((len(X_test), N * 2)),
        columns=[
            "Fold_{}_Prob_{}".format(i, j) for i in range(1, N + 1) for j in range(2)
        ],
    )
    importances = pd.DataFrame(
        np.zeros((X_train.shape[1], N)),
        columns=["Fold_{}".format(i) for i in range(1, N + 1)],
        index=df_all.columns,
    )
    fprs, tprs, scores = [], [], []

    skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print("Fold {}\n".format(fold))

        # Fitting the model
        leaderboard_model.fit(X_train[trn_idx], y_train[trn_idx])

        # Computing Train AUC score
        trn_fpr, trn_tpr, trn_thresholds = roc_curve(
            y_train[trn_idx], leaderboard_model.predict_proba(X_train[trn_idx])[:, 1]
        )
        trn_auc_score = auc(trn_fpr, trn_tpr)
        # Computing Validation AUC score
        val_fpr, val_tpr, val_thresholds = roc_curve(
            y_train[val_idx], leaderboard_model.predict_proba(X_train[val_idx])[:, 1]
        )
        val_auc_score = auc(val_fpr, val_tpr)

        scores.append((trn_auc_score, val_auc_score))
        fprs.append(val_fpr)
        tprs.append(val_tpr)

        # X_test probabilities
        probs.loc[:, "Fold_{}_Prob_0".format(fold)] = leaderboard_model.predict_proba(
            X_test
        )[:, 0]
        probs.loc[:, "Fold_{}_Prob_1".format(fold)] = leaderboard_model.predict_proba(
            X_test
        )[:, 1]
        importances.iloc[:, fold - 1] = leaderboard_model.feature_importances_

        oob += leaderboard_model.oob_score_ / N
        print("Fold {} OOB Score: {}\n".format(fold, leaderboard_model.oob_score_))

    print("Average OOB Score: {}".format(oob))
    return importances, fprs, tprs, probs, N
