from pyexpat import features
import pandas as pd
import os

dirname = os.path.dirname(__file__)
file_path =  os.path.join(dirname, '../data/train.csv')
home_data = pd.read_csv(file_path) 

print(home_data.describe())

y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

print(X.describe())

from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X, y)

predictions = iowa_model.predict(X)
print(predictions)
print(y)

from sklearn.metrics import mean_absolute_error

val_mae =mean_absolute_error(y, predictions)
print("Validation MAE: {:,.0f}".format(val_mae))

