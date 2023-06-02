# Trabajo-Final
Trabajo Final Especialización en IA

Debido a que el cliente no aportó un dataset, se debió recurrir a un dataset de Kaggle. Este es solar_wind.csv correspondiente a datos colectados de dos satélites de la SASA y NOAA. El proyecto Solar Wind busca estudiar los campos electromagnéticos del sol.
Hipótesis: El dataset no contiene datos anómalos.
En base a la hipotesis antes pleanteada, se decide generar registros con anomalidades en la variables "bt" que corresponde con la magnitud total del campo magnético interplanetario [nT].
Se agregó el feature "class" utilizandolo para indicar si el registro contiene cada valor anómalo. Los valores generados en bt se encuentran dentro de las magnitudes máximas y mínimas de la variable.

El dataset se dividió en Train-Test con una relación 80-20.

Se desarrollaron 3 modelos:
-Isolation Forest
-Restrictive Boltzmann Machine
-LSTM Autoencoder