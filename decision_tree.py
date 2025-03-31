"""
This script is used to train a decision tree classifier on the adult dataset.

Author: Aolin Yang
"""

import pandas as pd
from tree_node import TreeNode


init_df = pd.read_csv('data/training/adult_train_data.csv')
test_df = pd.read_csv('data/testing/adult_test_data.csv')

my_decision_tree_2 = TreeNode(data=init_df, index=0, max_depth=2)
my_decision_tree_2.train()
my_decision_tree_3 = TreeNode(data=init_df, index=0, max_depth=3)
my_decision_tree_3.train()
my_decision_tree_4 = TreeNode(data=init_df, index=0, max_depth=4)
my_decision_tree_4.train()
my_decision_tree_10 = TreeNode(data=init_df, index=0, max_depth=10)
my_decision_tree_10.train()

my_decision_tree_15 = TreeNode(data=init_df, index=0, max_depth=15)
my_decision_tree_15.train()
# my_decision_tree_no_depth = TreeNode(data=init_df, index=0)
# my_decision_tree_no_depth.train()

print("Decision tree training completed.")

my_decision_tree_2.print_tree()
my_decision_tree_3.print_tree()
my_decision_tree_4.print_tree()


my_decision_tree_2.print_statistics(test_df)

my_decision_tree_3.print_statistics(test_df)

my_decision_tree_4.print_statistics(test_df)

my_decision_tree_10.print_statistics(test_df)

my_decision_tree_15.print_statistics(test_df)

# print("Decision tree with no max depth:")

# print(my_decision_tree_no_depth.accuracy(init_df))
# print(my_decision_tree_no_depth.accuracy(test_df))
# print(my_decision_tree_no_depth.recall(test_df))
# print(my_decision_tree_no_depth.precision(test_df))
# print(my_decision_tree_no_depth.f1_score(test_df))
