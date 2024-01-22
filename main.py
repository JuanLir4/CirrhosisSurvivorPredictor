#Esse conjunto de dados me da como saida um numero de dias
#e algo que aconteceu apos esse dias, sendo C (censurado)/CL (censurado devido ao tx do fígado/D (óbito)
import keras
from keras.utils import to_categorical
import pandas as pd 
from sklearn.preprocessing import LabelEncoder #tranformar em numeros
import numpy as np
from keras.layers import Dense,Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

labelencoder = LabelEncoder()

DadosTotais  = pd.read_csv('cirrhosis.csv').dropna()

#preciso modificar alguns dados de entrada para numeros
dados_entrada = DadosTotais.iloc[:, 3:].values#apartir de drug

for i in range(dados_entrada.shape[1]):
    dados_entrada[:, i] = labelencoder.fit_transform(dados_entrada[:, i]) 
dados_entrada = np.asarray(dados_entrada).astype(np.float32)


dados_saida = DadosTotais.iloc[:, 2].values

dados_saida = labelencoder.fit_transform(dados_saida) 
dados_saida1 = to_categorical(dados_saida)
dados_saida1  = np.asarray(dados_saida1 ).astype(np.float32)



#começando a rede neural
def rede_neural():
    classificador = Sequential()
    classificador.add(Dense(units=10,activation='relu',input_dim=17))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=5,activation='softmax'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=5,activation='softmax'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=5,activation='softmax'))
    classificador.add(Dense(units = 3, activation= 'softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.002, weight_decay=0.002,)


    classificador.compile(optimizer= optimizer, loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn = rede_neural,
                                epochs = 1000,
                                batch_size = 10)
resultados = cross_val_score(estimator = classificador,
                             X = dados_entrada, y = dados_saida1,
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()

print(media)
print(desvio)











