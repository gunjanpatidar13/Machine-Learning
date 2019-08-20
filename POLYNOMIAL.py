

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

"""
#SPLIT DATASET INTO TRAINING AND TEST SET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#FEATURES SCALING
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

#FITTING LINEAR REGRESSION TO DATASET

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)



#FITTING POLYNOMIAL REGRESSION TO DATASET

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#Plotting LR

plt.scatter(x,y, color='red')
plt.plot(x, lin_reg.predict(x),color="black")
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('Position vs Salary')
plt.show()


#plotting PR
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y, color='red')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)),color="black")
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('Position vs Salary')
plt.show()
