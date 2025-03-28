import pandas as pd
import numpy as np

init_df = pd.read_csv('data/training/adult_train_data.csv')

column_feature_map = {}
for column in init_df.columns:
    column_feature_map[column] = init_df[column].unique().tolist()
    
print(column_feature_map['workclass'])