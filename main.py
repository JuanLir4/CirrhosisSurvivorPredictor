
import keras
from scipy.sparse import csr_matrix
from keras.utils import to_categorical
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #tranformar em numeros e em matrizes
from sklearn.compose import ColumnTransformer
import numpy as np
from keras.layers import Dense,Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

labelencoder = LabelEncoder()

DadosTotais  = pd.read_csv('cirrhosis.csv', encoding= 'ISO-8859-1')


#mostrar a variação dos dados, desse modo verificamos se é viavel utiliza-los e como devemos tratalos 
DadosTotais['Drug'].value_counts()
DadosTotais['Sex'].value_counts()
DadosTotais['Ascites'].value_counts()
DadosTotais['Hepatomegaly'].value_counts()
DadosTotais['Spiders'].value_counts()
DadosTotais['Edema'].value_counts()



#verifiquei que todos os dados depois da linha 313 nao devem ser utilizados:
DadosTotais = DadosTotais.drop(range(312, 418))

#procurando dados 'NaN' e substituindo pela media:
colunas_com_nulos = DadosTotais.columns[DadosTotais.isnull().any()]
#print(colunas_com_nulos) -> 'Cholesterol', 'Copper', 'Tryglicerides', 'Platelets'

#achando as medias
DadosTotais['Cholesterol'].value_counts()
MediaCholesterol = DadosTotais['Cholesterol'].mean()

DadosTotais['Copper'].value_counts()
MediaCopper = DadosTotais['Copper'].mean()


DadosTotais['Tryglicerides'].value_counts()
MediaTryglicerides = DadosTotais['Tryglicerides'].mean()


DadosTotais['Platelets'].value_counts()
MediaPlatelets = DadosTotais['Platelets'].mean()

#substituindo
valores = { 'Cholesterol' : MediaCholesterol, 'Copper' : MediaCopper,
           'Tryglicerides' : MediaTryglicerides, 'Platelets' : MediaPlatelets}

DadosTotais = DadosTotais.fillna(value = valores)

#testando:
colunas_com_nulos2 = DadosTotais.columns[DadosTotais.isnull().any()]
#print(colunas_com_nulos2) - > Index([], dtype='object') 


#separar o conjunto de dados:
dados_entrada = DadosTotais.iloc[:, 3: ]
dados_saida = DadosTotais.iloc[:, 2]

#percebemos que em 'edema' temos 3 opções em vez de 2,  provavel erro.
dados_entrada['Edema'] = dados_entrada['Edema'].replace('S','Y')
dados_entrada['Edema'].value_counts()

#passando dados(str) para int
for i in range(0,6):
    dados_entrada.iloc[:, i] = labelencoder.fit_transform(dados_entrada.iloc[:, i])


#excluir coluna 'age'

dados_entrada = dados_entrada.drop('Age', axis=1)


#passando dados(str) para int
dados_saida = labelencoder.fit_transform(dados_saida)

# passando dados saida para onehot
from sklearn.compose import ColumnTransformer
columnTransformer=ColumnTransformer([('encoder',OneHotEncoder(),[0,1,2,3,4,5])],remainder='passthrough')
dados_entrada=np.array(columnTransformer.fit_transform(dados_entrada))


dados_saida = to_categorical(dados_saida)

#criando agora o dataframe para os dias:
dados_saida2 = DadosTotais.iloc[:, 1]

#começando a rede neural
def rede_neural1():
    classificador = Sequential()
    classificador.add(Dense(units=12,activation='relu',input_dim=22))
    #classificador.add(Dropout(0.2))
    classificador.add(Dense(units=24,activation='sigmoid'))
    #classificador.add(Dropout(0.2))
    classificador.add(Dense(units=24,activation='sigmoid'))
    classificador.add(Dense(units = 3, activation= 'softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.004,)


    classificador.compile(optimizer= optimizer, loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
        
    return classificador

classificador = KerasClassifier(build_fn = rede_neural1,
                                epochs = 1000,
                                batch_size = 20)
resultados = cross_val_score(estimator = classificador,
                            X = dados_entrada, y = dados_saida,
                            cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std()


print(media)
print(desvio)


##FICA A LIÇÃO: NAO DA PRA COLOCAR NA MESMA REDE NEURAL, DUAS SAIDAS DE TIPOS DIFERENTES
##AGORA VAMOS CRIAR UMA OUTRA REDE PARA OS DIAS

#RedeNeura2
# previsor = Sequential()


# previsor.add(Dense(units=12,activation='relu',input_dim=22))
# previsor.add(Dense(units=24,activation='relu'))
# previsor.add(Dense(units=24,activation='relu'))
# previsor.add(Dense(units=24,activation='relu'))
# previsor.add(Dense(units = 1, activation= 'linear'))

# optimizer2 = keras.optimizers.Adam(learning_rate=0.001)

# previsor.compile(optimizer= optimizer2, loss = 'mse')
        



# previsor.fit(dados_entrada ,dados_saida2, epochs = 5000, batch_size = 50 )
# previsoes2 = previsor.predict(dados_entrada)


# print(previsoes2)
# print(previsoes2.mean())
# print(dados_saida2.mean())

