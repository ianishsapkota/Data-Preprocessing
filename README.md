# Data Preprocessing Template

A comprehensive Python script for data preprocessing using scikit-learn, covering essential steps from handling missing data to feature scaling.

## Overview

This script demonstrates a complete data preprocessing pipeline for machine learning projects, including data cleaning, encoding, and standardization.

## Features

- **Missing Data Handling**: Imputes missing numerical values using mean strategy
- **Categorical Encoding**: Converts categorical variables to numerical format
- **Label Encoding**: Encodes target variable for classification tasks
- **Train-Test Split**: Separates data into training and testing sets
- **Feature Scaling**: Standardizes features for improved model performance

## Requirements

```bash
numpy
pandas
scikit-learn
```

Install dependencies:
```bash
pip install numpy pandas scikit-learn
```

## Dataset Format

The script expects a CSV file named `Data.csv` with the following structure:
- First column: Categorical feature (e.g., Country)
- Middle columns: Numerical features (e.g., Age, Salary)
- Last column: Target variable (e.g., Purchased: Yes/No)

Example:
```
Country,Age,Salary,Purchased
France,44,72000,No
Spain,27,48000,Yes
Germany,30,54000,No
```

## Preprocessing Steps

### 1. Data Import
```python
df = pd.read_csv('Data.csv')
x = df.iloc[:,:-1].values  # Features
y = df.iloc[:,-1].values   # Target
```

### 2. Missing Data Imputation
Replaces missing values in numerical columns (columns 1-2) with the mean of each column:
```python
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x[:,1:3] = imputer.fit_transform(x[:,1:3])
```

### 3. Encoding Categorical Features
Applies One-Hot Encoding to the first column (categorical data):
```python
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
```

### 4. Encoding Target Variable
Converts categorical target variable to numerical labels:
```python
le = LabelEncoder()
y = le.fit_transform(y)
```

### 5. Train-Test Split
Splits data into 80% training and 20% testing sets:
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
```

### 6. Feature Scaling
Standardizes numerical features (columns 3 onwards after encoding):
```python
sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])
x_test[:,3:] = sc.transform(x_test[:,3:])
```

## Usage

1. Place your dataset file (`Data.csv`) in the same directory as the script
2. Run the script:
```bash
python preprocessing.py
```

3. The script outputs:
   - Encoded target variable
   - Scaled training set
   - Scaled testing set

## Important Notes

- **Feature Scaling**: Only applied to columns starting from index 3 to avoid scaling the one-hot encoded categorical features
- **Random State**: Set to 1 for reproducible results
- **Mean Imputation**: Suitable for numerical data; consider other strategies for different distributions
- **One-Hot Encoding**: Creates binary columns for each category; may lead to high dimensionality with many categories

## Customization

Modify these parameters based on your dataset:
- `test_size`: Adjust train-test split ratio
- `strategy`: Change imputation method ('median', 'most_frequent', 'constant')
- `random_state`: Set different seed for different splits
- Column indices: Update based on your dataset structure

## Output

The script prints:
1. Encoded target variable array
2. Preprocessed and scaled training features
3. Preprocessed and scaled testing features
