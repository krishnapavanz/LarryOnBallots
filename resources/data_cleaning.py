'''
Data cleaning
'''

import pandas as pd

demographics_filepath = "data/demographics_2021.csv"
face_covering_filepath = "data/face_covering.csv"

demographics = pd.read_csv(demographics_filepath, sep=';') 
demographics.set_index('id')
face_covering = pd.read_csv(face_covering_filepath, sep=';') 
face_covering.set_index('municipality_id')


def merge_dem_face_coverings(demographics, face_covering):
    '''
    Merge datasets of demographics and face covering. 
    '''

    # With the inner join we lose the votes from expats (7 rows in the face_covering file)
    merged_data = demographics.merge(
        face_covering,
        left_on='id', right_on='municipality_id',
        suffixes=('_dem', '_facecov'))
    
    return merged_data


