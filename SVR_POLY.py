


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

"""
#SPLIT DATASET INTO TRAINING AND TEST SET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
"""

#FEATURES SCALING

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

"""
x1=sc_x.inverse_transform(x)
y1=sc_y.inverse_transform(y)
"""

#FITTING  REGRESSION TO DATASET

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)


#PREDICT NEW RESULT

y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

#PLOTTING THE REGRESSION MODEL

x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='black')
plt.title("Truth vs Bluff(SVR)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()






