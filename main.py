import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

#importing the dataset
df = pd.read_csv('Data.csv')
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

#taking care of missing data
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

print(x[:,1:3])