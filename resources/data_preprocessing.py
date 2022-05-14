'''
Data preprocessing functions
'''

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def load_demographics(path):
    '''
    Load demographics data and clean common errors.
    Input: path e.g, "data/demographics_2021.csv"
    Output: DF
    '''

    data = pd.read_csv(path, sep=';')
    data = data.set_index('id')

    for col in data.iloc[:,1:].columns:
        if data[col].dtype == "object":
            data[col] = data[col].str.replace("\'","")
            data[col] = data[col].replace("X", np.NaN)
            data[col] = data[col].replace("*", 0)
            data[col] = data[col].astype(float)

    return data

def load_referendum(path):
    '''
    Load referendum data and clean common errors.
    Input: path e.g, "data/face_covering.csv"
    Output: DF
    '''

    data = pd.read_csv(path, sep=';')
    data = data.set_index('municipality_id')

    for col in data.iloc[:,3:].columns:
        if data[col].dtype == "object":
            data[col] = data[col].str.replace("\'","")
            data[col] = data[col].astype(float)

    return data

def merge_demographics_referendum(demographics, referendum):
    '''
    Merge datasets of demographics and face covering.
    Inputs:
        - demographics (DF)
        - referendum (DF)
    Output: DF
    '''

    # With the inner join we lose the votes from expats (7 rows in the face_covering file)
    merged_data = demographics.reset_index().merge(referendum,
        how='left',
        left_on='id',
        right_on='municipality_id',
        suffixes=('_dem', '_ref'))

    return merged_data



def handle_na(dataframe, fill = "mean", nn = 10):
    '''
    For a dataframe, review NAs and select missing data handling method (mean, median, drop NA, or KNN)
    '''
    if fill == "mean":
        dataframe.fillna(dataset.median(), inplace=True)
    elif fill == "median":
        dataframe.fillna(dataset.mean(), inplace=True)
    elif fill == "drop":
        dataframe.dropna()
    elif fill == 'KNN':
        imputer = KNNImputer(n_neighbors=nn)
        dataframe.iloc[:,1:]= imputer.fit_transform(dataframe.iloc[:,1:])
    else:
        assert('input correct method')

    return dataframe
