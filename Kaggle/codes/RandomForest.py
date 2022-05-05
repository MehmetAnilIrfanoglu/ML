import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



dirname = os.path.dirname(__file__)
file_path =  os.path.join(dirname, '../data/train.csv')
home_data = pd.read_csv(file_path) 

print(home_data.describe())

y = home_data.SalePrice
train_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[train_features]

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_pred = rf_model.predict(test_X)
rf_val_mae = mean_absolute_error(test_y, rf_pred)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))