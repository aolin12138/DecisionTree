"""
This module contains functions to calculate information gain for a given dataset.
"""
import numpy as np


def map_features(df):
    """
    Map atrributes to their unique values.
    """
    column_feature_map = {}
    for column in df.columns:
        column_feature_map[column] = df[column].unique().tolist()
    return column_feature_map


def entropy(df):
    """
    Calculate the entropy of a label array.
    """
    counts = df['income_more50K'].value_counts()
    zero_count = counts.get(0, 0)
    one_count = counts.get(1, 0)
    total_count = zero_count + one_count
    if zero_count == 0 or one_count == 0:
        return 0
    # Adding a small constant to avoid log(0)
    entropy = -(
        (zero_count / total_count) * np.log2(zero_count / total_count + 1e-10) +
        (one_count / total_count) * np.log2(one_count / total_count + 1e-10)
    )
    return entropy


def information_gain(df, feature, target_column):
    """
    Calculate the information gain of a feature.
    """
    # Calculate the entropy of the target column
    initial_entropy = entropy(df)

    # Calculate the weighted entropy after splitting on the feature
    weighted_entropy = 0
    correct_subset = df[df[target_column] == feature]
    incorrect_subset = df[df[target_column] != feature]
    total_count = len(df)
    correct_count = len(correct_subset)
    incorrect_count = len(incorrect_subset)

    # Calculate the entropy of the correct and incorrect subsets
    correct_entropy = entropy(correct_subset)
    incorrect_entropy = entropy(incorrect_subset)
    # Calculate the weighted entropy
    weighted_entropy += (correct_count / total_count) * correct_entropy
    weighted_entropy += (incorrect_count / total_count) * incorrect_entropy

    # Information gain is the difference between the initial entropy and the weighted entropy
    return initial_entropy - weighted_entropy


def calculate_best_feature(df):
    """
    Find the best feature to split on.
    """
    column_feature_map = map_features(df)
    best_gain = -1
    best_feature = None
    best_feature_column = None

    for column, features in column_feature_map.items():
        if column == 'income_more50K':
            continue
        for feature in features:
            gain = information_gain(df, feature, column)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_feature_column = column

    return best_feature, best_feature_column
