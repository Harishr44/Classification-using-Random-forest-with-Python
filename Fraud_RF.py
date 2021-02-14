# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:50:03 2020

@author: Harish
"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv("Fraud_check.csv")
data['Taxable.Income'].mean()

max(data['Taxable.Income'])#100000
min(data['Taxable.Income'])#10000
labels=["risky","good"]
bins=[0,30000,100000]
data['Taxable.Income']=pd.cut(data['Taxable.Income'],bins=bins,labels=labels)
data.isnull().sum()
from sklearn import preprocessing
string_columns=["Undergrad","Marital.Status","Urban"]
for i in string_columns:
    number=preprocessing.LabelEncoder()
    data[i]=number.fit_transform(data[i])
    
array = data.values


X = array[:,[0,1,3,4,5]]
Y = array[:,2]
num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean()) 
#74.67