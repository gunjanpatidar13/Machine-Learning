
#Importing Lbraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Dataset
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


#FITTING DT REGRESSION TO DATASET

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#PREDICT NEW RESULT

y_pred=regressor.predict([[6.5]])

#PLOTTING THE DECISION TREE REGRESSION MODEL WITH HIGHER RESOLUTION

x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='black')
plt.title("Truth vs Bluff(DT)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
