# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 01:36:11 2021

@author: Artem
"""

import numpy as np
import pandas as pd

import seaborn as sns

sns.set(style="darkgrid")

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import string
import warnings

warnings.filterwarnings("ignore")

SEED = 42


def feature_engineering(df_all, concat_df):
    # Feature Engineering
    # 2.1 Binning Continuous Features

    df_all["Fare"] = pd.qcut(df_all["Fare"], 13)
    df_all["Age"] = pd.qcut(df_all["Age"], 10)
    # 2.2 Частотное кодирование
    df_all["Family_Size"] = df_all["SibSp"] + df_all["Parch"] + 1
    df_all["Ticket_Frequency"] = df_all.groupby("Ticket")["Ticket"].transform("count")

    # 2.3 Title & Is Married
    df_all["Title"] = (
        df_all["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    )
    df_all["Is_Married"] = 0
    df_all["Is_Married"].loc[df_all["Title"] == "Mrs"] = 1

    # 2.4 Target Encoding
    def extract_surname(data):

        families = []

        for i in range(len(data)):
            name = data.iloc[i]

            if "(" in name:
                name_no_bracket = name.split("(")[0]
            else:
                name_no_bracket = name

            family = name_no_bracket.split(",")[0]

            for c in string.punctuation:
                family = family.replace(c, "").strip()

            families.append(family)

        return families

    df_all["Family"] = extract_surname(df_all["Name"])
    df_train = df_all.loc[:890]
    df_test = df_all.loc[891:]
    dfs = [df_train, df_test]

    # Создание списка семей и билетов, которые встречаются как в обучающем, так и в тестовом наборе
    non_unique_families = [
        x for x in df_train["Family"].unique() if x in df_test["Family"].unique()
    ]
    non_unique_tickets = [
        x for x in df_train["Ticket"].unique() if x in df_test["Ticket"].unique()
    ]

    df_family_survival_rate = df_train.groupby("Family")[
        "Survived", "Family", "Family_Size"
    ].median()
    df_ticket_survival_rate = df_train.groupby("Ticket")[
        "Survived", "Ticket", "Ticket_Frequency"
    ].median()

    family_rates = {}
    ticket_rates = {}

    for i in range(len(df_family_survival_rate)):
        # Проверка семьи существует как в обучающем, так и в тестовом наборе, и имеет членов более 1
        if (
            df_family_survival_rate.index[i] in non_unique_families
            and df_family_survival_rate.iloc[i, 1] > 1
        ):
            family_rates[
                df_family_survival_rate.index[i]
            ] = df_family_survival_rate.iloc[i, 0]

    for i in range(len(df_ticket_survival_rate)):
        # Проверка билета существует как в обучающем, так и в тестовом наборе, и в нем более 1 участника
        if (
            df_ticket_survival_rate.index[i] in non_unique_tickets
            and df_ticket_survival_rate.iloc[i, 1] > 1
        ):
            ticket_rates[
                df_ticket_survival_rate.index[i]
            ] = df_ticket_survival_rate.iloc[i, 0]

    mean_survival_rate = np.mean(df_train["Survived"])

    train_family_survival_rate = []
    train_family_survival_rate_NA = []
    test_family_survival_rate = []
    test_family_survival_rate_NA = []

    for i in range(len(df_train)):
        if df_train["Family"][i] in family_rates:
            train_family_survival_rate.append(family_rates[df_train["Family"][i]])
            train_family_survival_rate_NA.append(1)
        else:
            train_family_survival_rate.append(mean_survival_rate)
            train_family_survival_rate_NA.append(0)

        for i in range(len(df_test)):
            if df_test["Family"].iloc[i] in family_rates:
                test_family_survival_rate.append(
                    family_rates[df_test["Family"].iloc[i]]
                )
                test_family_survival_rate_NA.append(1)
            else:
                test_family_survival_rate.append(mean_survival_rate)
                test_family_survival_rate_NA.append(0)

    df_train["Family_Survival_Rate"] = train_family_survival_rate
    df_train["Family_Survival_Rate_NA"] = train_family_survival_rate_NA
    df_test["Family_Survival_Rate"] = test_family_survival_rate
    df_test["Family_Survival_Rate_NA"] = test_family_survival_rate_NA

    train_ticket_survival_rate = []
    train_ticket_survival_rate_NA = []
    test_ticket_survival_rate = []
    test_ticket_survival_rate_NA = []

    for i in range(len(df_train)):
        if df_train["Ticket"][i] in ticket_rates:
            train_ticket_survival_rate.append(ticket_rates[df_train["Ticket"][i]])
            train_ticket_survival_rate_NA.append(1)
        else:
            train_ticket_survival_rate.append(mean_survival_rate)
            train_ticket_survival_rate_NA.append(0)

        for i in range(len(df_test)):
            if df_test["Ticket"].iloc[i] in ticket_rates:
                test_ticket_survival_rate.append(
                    ticket_rates[df_test["Ticket"].iloc[i]]
                )
                test_ticket_survival_rate_NA.append(1)
            else:
                test_ticket_survival_rate.append(mean_survival_rate)
                test_ticket_survival_rate_NA.append(0)

    df_train["Ticket_Survival_Rate"] = train_ticket_survival_rate
    df_train["Ticket_Survival_Rate_NA"] = train_ticket_survival_rate_NA
    df_test["Ticket_Survival_Rate"] = test_ticket_survival_rate
    df_test["Ticket_Survival_Rate_NA"] = test_ticket_survival_rate_NA

    for df in [df_train, df_test]:
        df["Survival_Rate"] = (
            df["Ticket_Survival_Rate"] + df["Family_Survival_Rate"]
        ) / 2
        df["Survival_Rate_NA"] = (
            df["Ticket_Survival_Rate_NA"] + df["Family_Survival_Rate_NA"]
        ) / 2

    # 2.5 Feature Transformation
    # 2.5.1 Label Encoding Non-Numerical Features
    non_numeric_features = [
        "Embarked",
        "Sex",
        "Deck",
        "Title",
        "Family_Size_Grouped",
        "Age",
        "Fare",
    ]

    for df in dfs:
        for feature in non_numeric_features:
            df[feature] = LabelEncoder().fit_transform(df[feature])

    # 2.5.2 One-Hot Энкодинг категориальных признаков
    cat_features = ["Pclass", "Sex", "Deck", "Embarked", "Title", "Family_Size_Grouped"]
    encoded_features = []

    for df in dfs:
        for feature in cat_features:
            encoded_feat = (
                OneHotEncoder()
                .fit_transform(df[feature].values.reshape(-1, 1))
                .toarray()
            )
            n = df[feature].nunique()
            cols = ["{}_{}".format(feature, n) for n in range(1, n + 1)]
            encoded_df = pd.DataFrame(encoded_feat, columns=cols)
            encoded_df.index = df.index
            encoded_features.append(encoded_df)

    df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)
    df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)

    # 2.6 Заключение
    df_all = concat_df(df_train, df_test)
    drop_cols = [
        "Deck",
        "Embarked",
        "Family",
        "Family_Size",
        "Family_Size_Grouped",
        "Survived",
        "Name",
        "Parch",
        "PassengerId",
        "Pclass",
        "Sex",
        "SibSp",
        "Ticket",
        "Title",
        "Ticket_Survival_Rate",
        "Family_Survival_Rate",
        "Ticket_Survival_Rate_NA",
        "Family_Survival_Rate_NA",
    ]

    df_all.drop(columns=drop_cols, inplace=True)
    return df_all, drop_cols
