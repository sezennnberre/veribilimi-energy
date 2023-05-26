# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:28:03 2023

@author: HP
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

energy = pd.read_csv("ENB2012_data.csv")

X = energy.iloc[:,:8].values
X = pd.DataFrame(data = X, columns=["X1","X2","X3","X4","X5","X6","X7","X8"])
Y = energy.iloc[:,8:].values
Y = pd.DataFrame(data = Y,columns=["Y1","Y2"])

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.5,random_state=0)


#burada çoklu doğrusal regression kullanıldı.
from sklearn.linear_model import LinearRegression
lr  = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)


#burada decision tree kullanıldı
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(x_train,y_train)

y_pred_dtr = dtr.predict(x_test)


#random forest regression kullanıldı.
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=10, n_jobs=-1)
rfr.fit(x_train,y_train)

y_pred_rfr = rfr.predict(x_test)



#r^2 kullanıldı.
from sklearn.metrics import r2_score
r2 = r2_score(y_train, rfr.predict(x_train))
print(r2)


