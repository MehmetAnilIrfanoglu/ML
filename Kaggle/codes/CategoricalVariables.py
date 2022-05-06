import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


dirname = os.path.dirname(__file__)
file_path =  os.path.join(dirname, '../data/competition/train.csv')
X = pd.read_csv(file_path, index_col='Id')
 
dirname2 = os.path.dirname(__file__)
file_path2 =  os.path.join(dirname2, '../data/competition/test.csv')
X_test = pd.read_csv(file_path2, index_col='Id')

X.dropna(axis=0, inplace=True, subset=['SalePrice'])
y = X['SalePrice']
X.drop(['SalePrice'], axis=1, inplace=True)

cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=0)

print(X_train.head())

drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())


