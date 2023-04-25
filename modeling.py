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















