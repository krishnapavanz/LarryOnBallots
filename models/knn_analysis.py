import numpy as np
import matplotlib.pylab as plt
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


import sys
import os

sys.path.append(os.path.abspath('../resources'))
import data_preprocessing as dp
import split_normalization as sn

file_dir = os.path.dirname(__file__)

face_cov_filepath = os.path.join(file_dir,"..","data","face_covering.csv")
demographics_filepath = os.path.join(file_dir,"..","data","demographics_2021.csv")

# Loading data
demographics = dp.load_demographics(demographics_filepath)
face_covering = dp.load_referendum(face_cov_filepath)

# Merging data
merged_data = dp.merge_demographics_referendum(demographics, face_covering)

# 'names' can be edited to select a different set of columns
names = ['population_density', 'foreigner_percentage', \
    'age_percentage_between_20_64', 'agriculture_surface_perc', \
    'participation_rate', 'yes_perc']
df_filtered = merged_data.filter(names, axis=1)


# Filtering data
X = df_filtered.iloc[:, :-1].values
y = df_filtered.iloc[:, len(names)-1].values

# Converting continuous values to binary
# Assuming that a referendum is passed when yes_perc >= 51
y = np.array([1 if x >=51 else 0 for x in y])

# Splitting data into train, development and test
X_train, X_test, X_dev, y_train, y_test, y_dev = sn.split(X, y)

## Development data
print("\nDevelopment data: \n")
# Normalizing data
X_train_reg, X_dev_reg = sn.min_max_scaling(X_train, X_dev)

# Training KNN Classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train_reg, y_train)

# Testing with development data
y_dev_pred = classifier.predict(X_dev_reg)
print("Confusion matrix: \n", confusion_matrix(y_dev, y_dev_pred))
print("Classification report: \n",classification_report(y_dev, y_dev_pred))
print("Accuracy score: \n",accuracy_score(y_dev, y_dev_pred))

error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i_dev = knn.predict(X_dev)
    error.append(np.mean(pred_i_dev != y_dev))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value with Development data')
plt.xlabel('K Value')
plt.ylabel('Mean Error')



## Test data
print("\nTest data: \n")
# Normalizing data
X_train_reg, X_test_reg = sn.min_max_scaling(X_train, X_test)

# Training KNN Classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train_reg, y_train)

# Testing with test data
y_test_pred = classifier.predict(X_test_reg)
print("Confusion matrix: \n", confusion_matrix(y_test, y_test_pred))
print("Classification report: \n",classification_report(y_test, y_test_pred))
print("Accuracy score: \n",accuracy_score(y_test, y_test_pred))

error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i_test = knn.predict(X_test)
    error.append(np.mean(pred_i_test != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value with Test data')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
