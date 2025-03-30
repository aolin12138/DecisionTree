"""
This module contains the TreeNode class, which represents a node in a decision tree.
"""

import pandas as pd
import numpy as np


class TreeNode:
    def __init__(self,
                 feature: str = None,
                 column: str = None,
                 data: pd.DataFrame = None,
                 label: str = None):
        """
        Initialize a tree node with left and right children and a value.
        :param left: Left child node.
        :param right: Right child node.
        :param
        value: Value of the node.
        """
        self.data = data
        self.feature = feature
        self.column = column
        self.label = label

    def __str__(self):
        return str(self.column) + ": " + str(self.feature)

    def train(self):
        """
        Train the tree node with the given data.
        :param data: Data to train the node.
        """
        import decision_tree as dt

        print(self.data)

        if self.data['income_more50K'].nunique() == 1:
            self.label = self.data['income_more50K'].values[0]
            return

        self.feature, self.column = dt.calculate_best_feature(self.data)
        right_data = self.data[self.data[self.column] == self.feature]
        left_data = self.data[self.data[self.column] != self.feature]

        if right_data.empty or left_data.empty:
            self.label = self.data['income_more50K'].mode()[0]
            return

        right_node = TreeNode(data=right_data)
        left_node = TreeNode(data=left_data)
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
