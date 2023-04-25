# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 02:45:18 2023

@author: Nikhil
"""

# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# reading the data
data = pd.read_csv("dataset/african_crises.csv")

# encoding countries
le_country = LabelEncoder()
le_country.fit(data["country"])
enc_country = le_country.transform(data["country"])

onehot_country = OneHotEncoder(sparse=False)
enc_country = enc_country.reshape(len(enc_country), 1)
onehot_country = onehot_country.fit_transform(enc_country)

onehot_country_df = pd.DataFrame(onehot_country, columns=le_country.classes_)

# encoding result
le_crisis = LabelEncoder()
le_crisis.fit(data["banking_crisis"])
enc_crisis = le_crisis.transform(data["banking_crisis"])

# preparing dataset
data_train = data.iloc[:,3:13]

# concatenating the two dataframe
result = pd.concat([data_train, onehot_country_df], axis=1)

# train-test split in dataframe
X = result.values
y = enc_crisis

# handling the imbalancing of the class
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=0, 
                                                    stratify=y)

# Logistic Regression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Logistic Regression results 
print(classification_report(y_test, y_pred))

# Random Forest Classification
clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Random Forest results 
print(classification_report(y_test, y_pred))

# XGBoost Classification
model = xgb.XGBClassifier().fit(X_train, y_train)
y_pred = model.predict(X_test)

# XGBoost results 
print(classification_report(y_test, y_pred))



