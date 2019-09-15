#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:53:03 2019

@author: aaranda
"""


# Libraries
import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

# Data
dataset = pd.read_csv('pulsar_stars.csv')

# Hay que calcular si es un pulsar o no, la última fila del dataset.
X = dataset.iloc[:, 0:8].values # No coge la 0 y coge hasta la 7.
y = dataset.iloc[:, 8].values # Coge la 8

# Separamos el set en train y test.
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

# Escalado de variables
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Construir la red neuronal artificial
classifier = Sequential()

# Entrada y primera capa oculta. Entran 8 variables a 3 neuronas.
classifier.add(Dense(units=6, kernel_initializer="uniform",
                     activation="relu", input_dim=8))

# Segunda capa. Las 6 neuronas se conectan con otras 6.
classifier.add(Dense(units=6, kernel_initializer="uniform",
                     activation="relu"))
# Última capa. Las 3 neuronas se conectan con la salida.
classifier.add(Dense(units=1, kernel_initializer="uniform",
                     activation="sigmoid"))

# Compilar la red neuronal artificial:
classifier.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=["accuracy"])

# Metemos el set de train a la red.
classifier.fit(X_train,y_train, batch_size = 10, epochs = 100)

# Evaluación de la red y cálculo de predicciones de pulsar.
y_pred = classifier.predict(X_test)
y_pred_bool = (y_pred > 0.3)
cm = confusion_matrix(y_test,y_pred_bool)

precision = (cm[0,0]+cm[1,1])/(17898*0.25)
sensibilidad = cm[0,0]/(cm[0,0]+cm[1,0]) # Tasa de acierto de no púlsares
Especifidad = cm[1,1]/(cm[0,1]+cm[1,1]) # Tasa de acierto de púlsares





































