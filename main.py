import numpy as np
import pandas as pd

#importing the dataset
df = pd.read_csv('Data.csv')
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

print(x)
print(y)