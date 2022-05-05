from pyexpat import features
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y,state):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=state)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


dirname = os.path.dirname(__file__)
file_path =  os.path.join(dirname, '../data/train.csv')
home_data = pd.read_csv(file_path) 

print(home_data.describe())

y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes

small = 100000000
small_tree =0
for tree_size in candidate_max_leaf_nodes:
    my_mae = get_mae(tree_size, train_X, test_X, train_y, test_y, 0)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(tree_size, my_mae))
    if my_mae < small :
        small_tree = tree_size
        small = my_mae

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = small_tree
print(small_tree)
print(small)

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X,y)
