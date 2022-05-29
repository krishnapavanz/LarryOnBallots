'''
Data preprocessing functions
'''

import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import resources.dummies_colinearity as dc
import resources.split_normalization as sn

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

    # Private households per-capita (per 5000)
    data['private_households'] = data['private_households'] / data['population'] * 5000

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


def load_data():
    # File paths
    file_dir = os.path.abspath('')
    face_cov_filepath = os.path.join(file_dir,"data","face_covering.csv")
    demographics_filepath = os.path.join(file_dir,"data","demographics_2021.csv")

    # Load data
    demographics = load_demographics(demographics_filepath)
    face_covering = load_referendum(face_cov_filepath)

    # Merge data
    merged_data_raw = merge_demographics_referendum(demographics, face_covering)
    merged_data_raw = merged_data_raw[merged_data_raw['yes'].notna()]

    # Remove irrelevant attributes
    rm_attr = ["id", "canton_id", "municipality_ref", "age_percentage_less_20", "death_rate",
            "social_aid_perc", "employment_total", "establishments_total", "establishments_primary",
            "establishments_secondary", "establishments_tertiary",
            "registered_voters", "blank_votes", "invalid_votes", "valid_ballots", "yes_count", "no_count"]
    merged_data = merged_data_raw.drop(rm_attr, axis = 1)

    # Create dummy columns for categorical attributes
    dummy_cols = ["canton"]
    merged_data = dc.add_dummies(merged_data, dummy_cols)

    # Separate X and y
    X_attr = merged_data.columns.to_list()
    X_attr.remove('yes')
    X = merged_data.drop(["yes"], axis = 1)
    y_attr = "yes"
    y = merged_data["yes"]

    # Split data into train, development and test
    X_train_all, X_test_all, X_dev_all, y_train, y_test, y_dev = sn.split(X, y)
    print("Shapes:\nX_train :", X_train_all.shape, "\nX_dev: ", X_dev_all.shape, "\nX_test: ", X_test_all.shape)

    # Handle NAs
    for dataframe in [X_train_all, X_test_all, X_dev_all]:
        dataframe = handle_na(dataframe, fill = "KNN", nn=5)

    # Remove yes_perc (required later for evaluation)
    X_train = X_train_all.drop(["yes_perc", "municipality_dem"], axis = 1)
    X_test = X_test_all.drop(["yes_perc", "municipality_dem"], axis = 1)
    X_dev = X_dev_all.drop(["yes_perc", "municipality_dem"], axis = 1)

    # Save column names
    X_cols = X_train.columns

    # Scale attributes
    X_train, X_test, X_dev = sn.min_max_scaling(X_train, X_test, X_dev)

    # Add column names
    X_train.columns = X_cols
    X_test.columns = X_cols
    X_dev.columns = X_cols

    # Round to 6 decimal places
    X_train = X_train.round(6)
    X_test = X_test.round(6)
    X_dev = X_dev.round(6)

    return X_train, X_test, X_dev, y_train, y_test, y_dev, X_attr