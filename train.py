import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.ensemble import IsolationForest

from sklearn.svm import OneClassSVM
import joblib
from joblib import dump


print("Se está cargando el dataset. Aguarde un momento")
filePath = "C:/Users/raul.blanco/Documents/4 - Personales/UBA/Repositorios/IA/Trabajo-Final/solar_wind.csv"
dF = pd.read_csv(filepath_or_buffer=filePath, header=0, sep=",")

df_A = dF.loc[dF['period']=='train_a']
df_A = df_A.dropna()

train_data = df_A.sample(frac=0.8, random_state=0)
test_data = df_A.drop(train_data.index)

features = ['bt']
anomaly_factor = 0.17
# Crear una copia del DataFrame para no modificar el original
df_new = train_data.copy()

for a in features:
  num_replace = int(train_data[a].size * anomaly_factor)

  # Obtener una muestra aleatoria de índices de fila para reemplazar
  replace_idx = np.random.choice(train_data.index, size=num_replace, replace=False)

  # Reemplazar los valores seleccionados con valores aleatorios entre -90 y 1050
  #new_values = np.random.uniform(-90, 1050, size=num_replace)
  new_values = np.random.uniform(0, 75, size=num_replace)
  df_new.loc[replace_idx, a] = new_values

  # Crear la columna "class" con los valores correspondientes
  df_new.loc[train_data[a] != df_new[a], 'class'] = 1
  df_new.loc[train_data[a] == df_new[a], 'class'] = 0
  train_data = df_new

features = ['bt']
anomaly_factor = 0.012
# Crear una copia del DataFrame para no modificar el original
df_new_test = test_data.copy()

for a in features:
  num_replace = int(test_data[a].size * anomaly_factor)

  # Obtener una muestra aleatoria de índices de fila para reemplazar
  replace_idx = np.random.choice(test_data.index, size=num_replace, replace=False)

  # Reemplazar los valores seleccionados con valores aleatorios entre -90 y 1050
  new_values = np.random.uniform(0, 75, size=num_replace)
  df_new_test.loc[replace_idx, a] = new_values

  # Crear la columna "class" con los valores correspondientes
  df_new_test.loc[test_data[a] != df_new_test[a], 'class'] = 1
  df_new_test.loc[test_data[a] == df_new_test[a], 'class'] = 0
  test_data = df_new_test

train_timedelta = train_data[['timedelta']]
test_timedelta = test_data[['timedelta']]

y = test_data['class']
y_full = test_data
train_data = train_data.drop(["timedelta", "period", "source", 'class'], axis=1)
test_data = test_data.drop(["timedelta", "period", "source", 'class'], axis=1)
#y_full = y_full.to_numpy() #esto lo tengo que cargar en el archivo test

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

test_data_pd = pd.DataFrame(test_data, columns=['bx_gse', 'by_gse', 'bz_gse', 'theta_gse',
       'phi_gse', 'bx_gsm', 'by_gsm', 'bz_gsm', 'theta_gsm', 'phi_gsm', 'bt',
       'density', 'speed', 'temperature'])
test_data_pd.reset_index()

test_timedelta.reset_index(drop=True, inplace = True)
test_data_out = pd.concat([test_timedelta, test_data_pd], axis = 'columns')

select = int(input("Presione 1 para LSTM Autoencoder, 2 para Restricted Boltzmann Machine, 3 para Isolation Forest, 4 para Support Vector Machines: "))
if select == 1: ### LSTM AUTOENCODER ###
    train_data_LSTM = train_data.reshape(train_data.shape[0],1,train_data.shape[1])
    #test_data_LSTM = test_data.reshape(test_data.shape[0],1,test_data.shape[1])

    #Modelo
    input_dim = train_data_LSTM.shape[2]
    timesteps = 1
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(64, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(32, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(16, activation='relu', return_sequences=False)(encoded)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(16, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(input_dim, activation='linear', return_sequences=True)(decoded)
    model_autoencoder = Model(inputs, decoded)
    model_autoencoder.compile(optimizer='adam', loss='mse')

    #Entreno el modelo
    model_autoencoder.fit(train_data_LSTM, train_data_LSTM, epochs=20   , batch_size=320, validation_split=0.1)
    #Guardo el modelo.
    joblib.dump(model_autoencoder, 'model_autoencoder.joblib')

elif select == 2: ### RBM ###
    #Modelo
    model_rbm = Sequential()
    model_rbm.add(Dense(20, input_shape=(train_data.shape[1],), activation='sigmoid'))
    model_rbm.add(Dense(train_data.shape[1], activation='sigmoid'))
    model_rbm.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    #Entreno el modelo
    model_rbm.fit(train_data, train_data, epochs=10, batch_size=32, verbose=1, validation_split=0.1)
    #Guardo el modelo.    
    joblib.dump(model_rbm, 'model_rbm.joblib')

elif select == 3: ### Isolation Forest ###
    #Modelo
    model_if = IsolationForest(n_estimators=100, contamination='auto', random_state=0)
    #Entreno el modelo
    model_if.fit(train_data,train_data)
    #Guardo el modelo.
    joblib.dump(model_if, 'model_if.joblib')

elif select == 4: ### Support Vector Machines ###
    #Modelo
    model_svm = OneClassSVM(kernel='rbf', nu=0.01)
    #Entreno el modelo
    model_svm.fit(train_data)
    #Guardo el modelo.
    joblib.dump(model_svm, 'model_svm.joblib')
else:
    print('Debe seleccionar un modelo')

select2 = int(input("Presione 1 para guardar dataset de test, 2 para no guardar dataset: "))
if select2 == 1:
    test_data_out.to_csv('test_data.csv')
    y_full.to_csv('y_full.csv')
    print("Se guardó dataset de test con el nombre test_data_LSTM.csv")

elif select2 == 2:
   print('No se guardó dataset')
else: 
   print('No se guardó dataset')