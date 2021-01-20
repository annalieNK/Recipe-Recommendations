# x = 1 + 1

# print(x)

import numpy as np

# y = [i*2 for i in [1,2,3]]

# print(y) 

import pandas as pd

# df = pd.DataFrame({'column1': [1,2,3], 'column2': [4,5,6]})

df = pd.read_json('/Users/annalie/Dev/poetry-demo-2/data/train.json')#'./data/train.json')

print(df.head())

