# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:28:05 2020

@author: Harish
"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv("Company_Data.csv")
max(data['Sales'])
min(data['Sales'])
np.mean(data['Sales'])
labels=["bad","average","good"]
bins=[0,5,10,17]
data['Sales']=pd.cut(data['Sales'],bins=bins,labels=labels)

data.Sales.mode()[0]    
data.isnull().sum()
#sales column has one na value
data.Sales=data.Sales.fillna(data.Sales.mode()[0])
data.isnull().sum()

from sklearn import preprocessing
string_columns=["ShelveLoc","Urban","US"]
for i in string_columns:
    number=preprocessing.LabelEncoder()
    data[i]=number.fit_transform(data[i])
   

array = data.values


X = array[:,1:11]
Y = array[:,0]
num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())    
#67.75