import pandas as pd
import numpy as np

def load_demographics(path):
    '''
    Load demographics data and clean common errors.
    Input: path e.g, "..\data\demographics_2021.csv"
    Output: DF
    '''
    data = pd.read_csv("demographics_2021.csv", sep=';')

    for col in data.iloc[:,2:].columns:
        if data[col].dtype == "object":
            data[col] = data[col].str.replace("\'","")
            data[col] = data[col].replace("X", np.NaN)
            data[col] = data[col].replace("*", 0)
            data[col] = data[col].astype(float)

    return data