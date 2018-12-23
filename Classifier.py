from sklearn.pipeline import make_pipeline as MP
from sklearn.impute import SimpleImputer as Imp
from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np


#Import Data (using oriented data)
df2 = pd.read_csv('labeled feature matrix.csv')

features = ['Mean Reoriented Accelerometer X',
            'Mean Reoriented Accelerometer Z',
            'Variance Reoriented Accelerometer X',
            'Variance Reoriented Accelerometer Y',
            'Variance Reoriented Accelerometer Z',
            'Variance Reoriented Gyroscope Y',
            'Max Reoriented Accelerometer Y',
            'Max Reoriented Accelerometer Z',
            'Max Gyroscope X',
            'Min Reoriented Accelerometer X',
            'Min Reoriented Accelerometer Z',
            'Min Reoriented Gyroscope X',
            'Min Reoriented Gyroscope Y',
            'Min Reoriented Gyroscope Z']

feature_matrix = df2[features[:]]
print(feature_matrix)
print(df2['label'])

X = feature_matrix
y = df2['label']

my_pipeline = MP(Imp(), KN(n_neighbors=1))
scores = cross_val_score(my_pipeline, X, y, scoring='accuracy')
print(scores)






'''
X = df[features].iloc[15000:35000]
X = X.append(df[features].iloc[43000:61000])

y = df.label.iloc[15000:35000]
y = y.append(df.label.iloc[43000:61000])

print(X)
print(y)


my_pipeline = MP(Imp(),KN(n_neighbors=1),)
scores = cross_val_score(my_pipeline, X, y, scoring='accuracy')
print(scores)


#
'''

