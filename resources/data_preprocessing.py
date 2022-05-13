'''
Data preprocessing functions
'''

import pandas as pd
import numpy as np

def load_demographics(path):
    '''
    Load demographics data and clean common errors.
    Input: path e.g, "\data\demographics_2021.csv"
    Output: DF
    '''

    data = pd.read_csv(path, sep=';')
    data.set_index('id')

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
    Input: path e.g, "\data\face_coverings.csv"
    Output: DF
    '''

    data = pd.read_csv(path, sep=';')
    data.set_index('municipality_id')

    for col in data.iloc[:,4:10].columns:
        if data[col].dtype == "object":
            data[col] = data[col].str.replace("\'","")
            data[col] = data[col].astype(float)
    
    return data

def merge_dem_face_coverings(demographics, face_covering):
    '''
    Merge datasets of demographics and face covering. 
    Inputs: 
        - demographics (DF)
        - face_covering (DF)
    Output: DF 
    '''

    # With the inner join we lose the votes from expats (7 rows in the face_covering file)
    merged_data = demographics.merge(
        face_covering,
        left_on='id', right_on='municipality_id',
        suffixes=('_dem', '_facecov'))
    
    return merged_data

