
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



dirname = os.path.dirname(__file__)
file_path =  os.path.join(dirname, '../data/competition/train.csv')
X_full = pd.read_csv(file_path, index_col='Id')
 
dirname2 = os.path.dirname(__file__)
file_path2 =  os.path.join(dirname2, '../data/competition/test.csv')
X_test_full = pd.read_csv(file_path2, index_col='Id')

X_full.dropna(axis=0, inplace=True, subset=['SalePrice'])
y = X_full['SalePrice']
X_full.drop(['SalePrice'], axis=1, inplace=True)

X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2,train_size=0.8 ,random_state=0)

print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]
# Fill in the lines below: drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing,axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing,axis=1)

print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))