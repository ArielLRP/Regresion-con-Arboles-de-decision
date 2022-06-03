# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:49:38 2022

@author: ariel
"""

# =============================================================================
# Arboles de decisiones
# =============================================================================

#Importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Ajustar la regresion con el dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state = 0)
regression.fit(X, y)

#Prediccion de nuestro modelo
y_pred = regression.predict([[6.5]])

#Visualizacion de los resultados del Modelo de regresion de arboles de decisiones
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regression.predict(X_grid), color = 'blue')
plt.title('Modelo de regresion')
plt.xlabel('Posicion del empleado')
plt.ylabel('Sueldo (en dolares)')
plt.show()