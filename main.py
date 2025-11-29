import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.model_selection import train_test_split

#importing the dataset
df = pd.read_csv('Data.csv')
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

#taking care of missing data
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#encoding independent variable
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

#encoding dependent variable
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#creating training set and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)