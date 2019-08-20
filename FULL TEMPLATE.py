


# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:19:08 2019

@author: VAIBHAV
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#FOR MISSING VALUES
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#FOR CATEGORIES
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

#onehotencoder=OneHotEncoder(categorical_features=[0])
#x[:,0]=onehotencoder.fit_transform(x[:,0]).toarray()

ct = ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x=np.array(ct.fit_transform(x),dtype=np.float)
labelencoder = LabelEncoder()
y=np.array(labelencoder.fit_transform(y))

#SPLIT DATASET INTO TRAINING AND TEST SET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


#FEATURES SCALING
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train[]=sc_x.fit_transform(x_train[])
x_test[]=sc_x.transform(x_test[])



























