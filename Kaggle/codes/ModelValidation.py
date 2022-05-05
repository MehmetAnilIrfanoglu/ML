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

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)


from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)


from sklearn.metrics import mean_absolute_error

predicted_home_prices = iowa_model.predict(test_X)

val_mae =mean_absolute_error(test_y, predicted_home_prices)
print("Validation MAE: {:,.0f}".format(val_mae))