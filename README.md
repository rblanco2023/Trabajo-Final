# Trabajo-Final
Trabajo Final Especialización en IA

Los satélites son equipos con un alto grado de complejidad. Utilizan múltiples sistemas y subsistemas para adquirir datos, procesarlos, realizar cálculos y tomar decisiones. La información entre los sistemas se transmite por medio de buses de comunicación. A su vez, existen distintos buses, con diferentes protocolos y diversas velocidades de comunicación. Esta complejidad, sumado al entorno hostil en el que estos equipos operan, ocasionan errores en los datos que se propagan por el satélite y generan resultados no esperados. En algunas ocasiones, dichos errores se pueden detectar fácilmente, pero en muchas otras pasan desapercibidos.

INVAP S.E. es referente a nivel internacional y la empresa más importante en Argentina que se dedica al diseño y construcción de sistemas tecnológicos complejos. La detección de las anomalías mencionadas anteriormente representa un desafío para INVAP S.E. [1] y todas las empresas que construyen sistemas satelitales.

Para la simulación del comportamiento de un satélite, se utiliza una representación virtual del mismo que se actualiza a partir de datos en tiempo real, mediante sensores conectados al sistema físico. Este sistema virtual se denomina gemelo digital. Estas representaciones virtuales se utilizan para ensayar, detectar, predecir y corregir problemas antes del lanzamiento del satélite al espacio. El uso de
gemelos digitales mejora la eficiencia y la seguridad del satélite durante su vida útil. También reduce los costos y el tiempo en testing del sistema físico.
En el presente trabajo se planteó utilizar modelos de inteligencia artificial para la predicción de anomalías en los buses de comunicación de los satélites. Se busca automatizar el análisis de los datos que estos producen, y minimizar la dependencia humana y los errores. La inteligencia artificial brinda herramientas para la detección de las anomalías, prediciendo si un dato debe o no ser analizado, reprocesado y/o descartado.

Para el desarrollo del presente trabajo se recurrió al dataset solar_wind.csv presente en Kaggle. Corresponde a datos colectados de dos satélites de la NASA y NOAA. El proyecto Solar Wind busca estudiar los campos electromagnéticos del sol.

Afirmación: El dataset no contiene datos anómalos.

De la información pleanteada, se decide generar registros con anomalidades en la variables "bt" que corresponde con la magnitud total del campo magnético interplanetario [nT].

Se dispone de tres archivos:
-dataset.ipynb
-Trabajo-final_train.ipynb
-Trabajo-final_test.ipynb

El primer archivo realiza el estudio del dataset, lo divide en train y test y le incorpora anomalías utilizando dos distribuciones. Por un lado se generan anomalías con distribución uniforme y por otro lado anomalías con distribución normal. Se generan 2 sets de datos con las anomalías generadas.

El segundo archivo realiza el preprocesamiento y el entrenamiento de los siguientes tres modelos:
-Isolation Forest
-Restrictive Boltzmann Machine
-LSTM Autoencoder

Cada modelo fue entrenado con ambos sets de datos (normal y uniforme)
También se estudia el entrenamiento, analizando matrices de confusión y métricas.

El ultimo archivo, se utiliza para simular el funsionamiento de los modelos antes entrenados. Para ello se utiliza la porción del set de datos resetvado para este fin. El análisis del desempeño de los 6 modelos (autoencoder_n y autoencoder_u, RBM_n y RBM_u, IF_n y IF_u) fue realizado con sus respectivos sets de datos.