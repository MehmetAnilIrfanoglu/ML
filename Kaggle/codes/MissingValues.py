import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



dirname = os.path.dirname(__file__)
file_path =  os.path.join(dirname, '../data/melb_data.csv')
data = pd.read_csv(file_path) 

y = data.Price
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0,train_size=0.8,test_size=0.2)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
X_train_clean = X_train.drop(cols_with_missing, axis=1)
X_valid_clean = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(X_train_clean, X_valid_clean, y_train, y_valid))