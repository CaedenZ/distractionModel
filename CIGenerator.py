import numpy as np
import csv
import pandas as pd

files = ["data_center.csv", "data_downleft.csv",
         "data_downright.csv", "data_left.csv", "data_right.csv", "data_upperleft.csv", "data_uppright.csv"]


for i in range(len(files)):
    data = pd.read_csv(files[i], delimiter=" ").astype('float')

    data.dropna(inplace=True)
    print(files[i])
    print(data.describe())
