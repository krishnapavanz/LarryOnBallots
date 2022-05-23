'''
Data preprocessing functions
'''

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def load_demographics(path):
    '''
    Load demographics data and clean common errors as well as some feature engineering
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

    # Employemnt per-capita (per 5000)
    data['employment_total'] = data['employment_total'] / data['population'] * 5000
    data['employment_primary'] = data['employment_primary'] / data['population'] * 5000
    data['employment_secondary'] = data['employment_secondary'] / data['population'] * 5000
    data['employment_tertiary'] = data['employment_tertiary'] / data['population'] * 5000

    # Establishments per-capita (per 5000)
    data['establishments_total'] = data['establishments_total'] / data['population'] * 5000
    data['establishments_primary'] = data['establishments_primary'] / data['population'] * 5000
    data['establishments_secondary'] = data['establishments_secondary'] / data['population'] * 5000
    data['establishments_tertiary'] = data['establishments_tertiary'] / data['population'] * 5000

    # Empty housing per-capita (per 5000)
    data['empty_housing_units'] = data['empty_housing_units'] / data['population'] * 5000

    return data

def load_referendum(path):
    '''
    Load referendum data and clean common errors as well as some feature engineering
    Input: path e.g, "data/face_covering.csv"
    Output: DF
    '''

    data = pd.read_csv(path, sep=';')
    data = data.set_index('municipality_id')

    for col in data.iloc[:,3:].columns:
        if data[col].dtype == "object":
            data[col] = data[col].str.replace("\'","")
            data[col] = data[col].astype(float)

    data['yes'] = np.array([1 if yes_perc > 50 else 0 for yes_perc in data['yes_perc']])

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
        dataframe.fillna(dataframe.mean(), inplace=True)
    elif fill == "median":
        dataframe.fillna(dataframe.median(), inplace=True)
    elif fill == "drop":
        dataframe.dropna()
    elif fill == 'KNN':
        imputer = KNNImputer(n_neighbors=nn)
        dataframe.iloc[:,1:]= imputer.fit_transform(dataframe.iloc[:,1:])
    else:
        assert('input correct method')

    return dataframe

    