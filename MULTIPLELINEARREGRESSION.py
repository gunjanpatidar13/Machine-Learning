
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#DUMMY VARIABLE
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(ct.fit_transform(x),dtype=np.float)

x=x[:,1:]

#SPLIT DATASET INTO TRAINING AND TEST SET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


#FEATURES SCALING
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train[:,2:] = sc_x.fit_transform(x_train[:,2:])
x_test = sc_x.transform(x_test)"""

#FITTING THE MULTIPLE L R MODEL
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


# BUILDING OPTIMAL MODAL USING BACKWARD ELIMINATION
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x,axis=1)

#FOR GENERAL
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#FOR X2(beacuse x2 is greater amoung all)
x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#FOR X1
x_opt = x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#FOR X4
x_opt = x[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#FOR X5
x_opt = x[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x1 = x_opt[:,1]


plt.scatter(x1,y,color='red')
