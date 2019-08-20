# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:25:52 2019

@author: VAIBHAV
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#SPLIT DATASET INTO TRAINING AND TEST SET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)


#FEATURES SCALING
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

#FITTINF THE SIMPLE LINEAR REGRESSION TO TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#PREDICTING THE TEST SET RESULT
y_pred = regressor.predict(x_test)

#PLOTTING THE TRAINING SET
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train))
plt.xlabel('Experiance')
plt.ylabel('Salary')
plt.title('Experiance vs Salary')
plt.show()

#PLOTTING FOR TEST SET
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train))
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Experiance vs Salary')
plt.show()







