'''
Data cleaning
'''

import pandas as pd
import csv

demographics_filepath = "data/demographics_2021.csv"
face_covering_filepath = "data/face_covering.csv"

demographics = pd.read_csv(demographics_filepath, sep=';') 
face_covering = pd.read_csv(face_covering_filepath, sep=';') 


with open(face_covering_filepath, encoding="utf8", errors='ignore') as f:
    face_covering = pd.read_csv(f, sep=';')





