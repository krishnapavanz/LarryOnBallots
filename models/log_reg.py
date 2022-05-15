import numpy as np
import matplotlib.pylab as plt
import pandas as pd


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


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
df_filtered = df_filtered.dropna()


# Filtering data
X = df_filtered.iloc[:, :-1].values
y = df_filtered.iloc[:, len(names)-1].values

# Converting continuous values to binary
# Assuming that a referendum is passed when yes_perc >= 51
y = np.array([1 if x >=51 else 0 for x in y])

# Splitting data into train, development and test
X_train, X_test, X_dev, y_train, y_test, y_dev = sn.split(X, y)


logreg = LogisticRegression()
# Instantiate the  model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)


## Development data
print("\nDevelopment data: \n")
# Testing with development data
y_dev_pred = logreg.predict(X_dev)
print("Confusion matrix: \n", confusion_matrix(y_dev, y_dev_pred))
print("Classification report: \n",classification_report(y_dev, y_dev_pred))
print("Accuracy score: \n",accuracy_score(y_dev, y_dev_pred))

y_dev_pred_proba = logreg.predict_proba(X_dev)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_dev,  y_dev_pred_proba)
auc = metrics.roc_auc_score(y_dev,  y_dev_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

## Test data
print("\nTest data: \n")
# Testing with development data
y_test_pred = logreg.predict(X_test)
print("Confusion matrix: \n", confusion_matrix(y_test, y_test_pred))
print("Classification report: \n",classification_report(y_test, y_test_pred))
print("Accuracy score: \n",accuracy_score(y_test, y_test_pred))

y_test_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_test_pred_proba)
auc = metrics.roc_auc_score(y_test,  y_test_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()