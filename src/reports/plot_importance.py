# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 01:20:17 2021

@author: Artem
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


def plot_importance(importances):
    importances['Mean_Importance'] = importances.mean(axis=1)
    importances.sort_values(by='Mean_Importance', inplace=True, ascending=False)

    plt.figure(figsize=(15, 20))
    sns.barplot(x='Mean_Importance', y=importances.index, data=importances)

    plt.xlabel('')
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.title('Random Forest Classifier Mean Feature Importance Between Folds', size=15)

    plt.show()