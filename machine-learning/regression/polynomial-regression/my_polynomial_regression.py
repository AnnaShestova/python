# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:05:04 2019

@author: it_ash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#polynomial regression - bluffing detector

#fitting Linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

#fitting Polynomial regression to the datase
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

#Visializing LinearRegresson results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regresson)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visializing PolynomialRegresson results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regresson)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predicting a new result with LinearRegression
lin_reg.predict(np.array([[6.5]]))

#predicting a new result with PolynomialRegresson
lin_reg_2.predict(poly_reg.fit_transform(np.array([[6.5]])))