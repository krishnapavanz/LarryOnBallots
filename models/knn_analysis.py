import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

file_dir = os.path.dirname(__file__)
face_cov_csv_path = os.path.join(file_dir,"..","data","facecovering.csv")
df_ref_face_ban = pd.read_csv(face_cov_csv_path)

print(df_ref_face_ban.head())