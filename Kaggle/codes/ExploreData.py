import pandas as pd
import os

dirname = os.path.dirname(__file__)
file_path =  os.path.join(dirname, '../data/train.csv')
home_data = pd.read_csv(file_path) 

print(home_data.describe())

