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

y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = X_full[features].copy()
X_test = X_test_full[features].copy()

X_train , X_valid, y_train, y_valid = train_test_split(X, y, random_state=0,train_size=0.8,test_size=0.2)
print(X_train.head())

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]


def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))


my_model = model_3

my_model.fit(X_train, y_train)
preds_test = my_model.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)