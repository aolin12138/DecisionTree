"""
This module contains the TreeNode class, which represents a node in a decision tree.
"""

import pandas as pd
import numpy as np
from info_gain_calculation import calculate_best_feature


class TreeNode:
    def __init__(self,
                 feature: str = None,
                 column: str = None,
                 data: pd.DataFrame = None,
                 label: str = None,
                 index: int = None,
                 max_depth: int = None,
                 left=None,
                 right=None):
        """
        Initialize a tree node with the given parameters.
        :param data: Data to train the node.
        :param feature: Feature to split on.
        :param column: Column name of the feature.
        :param label: Label of the node output
        value: Value of the node.
        """
        self.data = data
        self.feature = feature
        self.column = column
        self.label = label
        self.index = index
        self.max_depth = max_depth
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.column) + ": " + str(self.feature)

    def train(self):
        """
        Train the tree node with the given data.
        :param data: Data to train the node.
        """
        # print(self.index)

        if self.data['income_more50K'].nunique() == 1:
            self.label = self.data['income_more50K'].values[0]
            return

        if self.max_depth is not None and self.index >= self.max_depth:
            self.label = self.data['income_more50K'].mode()[0]
            return

        self.feature, self.column = calculate_best_feature(self.data)
        right_data = self.data[self.data[self.column] == self.feature]
        left_data = self.data[self.data[self.column] != self.feature]

        if right_data.empty or left_data.empty:
            self.label = self.data['income_more50K'].mode()[0]
            return

        right_node = TreeNode(
            data=right_data, index=self.index + 1, max_depth=self.max_depth)
        left_node = TreeNode(
            data=left_data, index=self.index + 1, max_depth=self.max_depth)
        self.add_children(left_node, "left")
        self.add_children(right_node, "right")
        # Recursively train the left and right nodes

        left_node.train()
        right_node.train()

    def add_children(self, tree_node, direction: str):
        """
        Add children to the node.
        """

        if direction == "left":
            self.left = tree_node
        elif direction == "right":
            self.right = tree_node
        else:
            raise ValueError("Direction must be 'left' or 'right'")

    def print_tree(self, depth=0, direction=None):
        """ Recursively print the tree structure """

        if self.index > 8:
            return

        indent = "   " * depth
        if self.feature is not None:
            if direction == "left":
                print(
                    f"{indent}├── {self.index} Elseif: {self.column}.{self.feature}")
            elif direction == "right":
                print(
                    f"{indent}├── {self.index} If: {self.column}.{self.feature}")
        else:
            print(f"{indent}└── {self.index} output {self.label}")

        if self.right is not None:
            self.right.print_tree(depth=depth + 1, direction="right")
        if self.left is not None:
            self.left.print_tree(depth=depth + 1, direction="left")

    def predict(self, row):
        """
        Predict the label for the given row.
        """

        if self.label is not None:
            return int(self.label)

        if row[self.column] == self.feature:
            return self.right.predict(row)
        else:
            return self.left.predict(row)

    def predict_all(self, df):
        """
        Predict the labels for all rows in the given DataFrame.
        """
        predictions = []
        for _, row in df.iterrows():
            prediction = self.predict(row)
            predictions.append(prediction)
        return predictions

    def accuracy(self, df):
        """
        Calculate the accuracy of the decision tree on the given DataFrame.
        """
        predictions = self.predict_all(df)
        correct_predictions = sum(predictions == df['income_more50K'])
        accuracy = correct_predictions / len(df)
        return accuracy

    def recall(self, df):
        """
        Calculate the recall of the decision tree on the given DataFrame.
        """
        predictions = self.predict_all(df)
        true_positives = sum(
            (predictions == df['income_more50K']) & (df['income_more50K'] == 1))
        false_negatives = sum(
            (predictions != df['income_more50K']) & (df['income_more50K'] == 1))

        if true_positives + false_negatives == 0:
            return 0

        recall = true_positives / (true_positives + false_negatives)
        return recall

    def precision(self, df):
        """
        Calculate the precision of the decision tree on the given DataFrame.
        """
        predictions = self.predict_all(df)
        true_positives = sum(
            (predictions == df['income_more50K']) & (df['income_more50K'] == 1))
        false_positives = sum(
            (predictions != df['income_more50K']) & (df['income_more50K'] == 0))

        if true_positives + false_positives == 0:
            return 0

        precision = true_positives / (true_positives + false_positives)
        return precision

    def f1_score(self, df):
        """
        Calculate the F1 score of the decision tree on the given DataFrame.
        """
        recall = self.recall(df)
        precision = self.precision(df)
        if precision + recall == 0:
            return 0
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
