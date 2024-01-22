# Cirrhosis Survivor Predictor

Este script em Python utiliza a biblioteca Keras e Scikit-learn para criar e avaliar uma Rede Neural Artificial (RNA) que utiliza 17 características clínicas para prever o estado de sobrevivência de pacientes com cirrose hepática. O conjunto de dados fornece como saída um número de dias e o evento que ocorreu após esse período, sendo C (censurado), CL (censurado devido ao transplante de fígado) ou D (óbito).

# Requisitos
Certifique-se de ter as seguintes bibliotecas instaladas no seu ambiente Python:

Keras
pandas
scikit-learn
numpy

Certifique-se de ter o arquivo 'cirrhosis.csv' no mesmo diretório que este script. Este arquivo contém os dados necessários para treinar e testar a RNA.

# Execute o script.

bash
Copy code
python nome_do_script.py
O script realizará as seguintes etapas:

Carregará os dados do arquivo 'cirrhosis.csv' e removerá as linhas com valores ausentes.

Transformará os dados de entrada em formato numérico usando a classe LabelEncoder da biblioteca scikit-learn.

Criará a representação one-hot encoding para os dados de saída usando a função to_categorical do Keras.

Definirá a arquitetura da RNA usando a biblioteca Keras, com três camadas ocultas e uma camada de saída.

Utilizará a função de otimização Adam com uma taxa de aprendizado de 0.002 e decaimento de peso de 0.002.

Compilará o modelo usando a função de perda 'categorical_crossentropy' e métricas de precisão categórica.

Criará uma instância do wrapper KerasClassifier para o modelo Keras, configurada para treinar por 1000 épocas com um tamanho de lote de 10.

Avaliará o desempenho do modelo usando validação cruzada com 10 dobras.
#  Resultados
O modelo foi refinado e agora alcança uma acurácia média de 72% nas 10 dobras de validação cruzada. Este desempenho melhorado proporciona uma base mais robusta para a previsão do estado de sobrevivência de pacientes com cirrose hepática. No entanto, continua sendo uma área de desenvolvimento contínuo, e ajustes adicionais podem ser explorados para aprimorar ainda mais a precisão do modelo.

# Detalhes da Rede Neural
A arquitetura da RNA consiste em três camadas ocultas, cada uma seguida por uma camada de dropout para mitigar o overfitting. A camada de saída possui três unidades, correspondendo aos três estados de sobrevivência, e utiliza a função de ativação softmax para gerar probabilidades. O otimizador Adam é empregado com uma taxa de aprendizado de 0.002 e decaimento de peso de 0.002.

Lembre-se de ajustar os parâmetros da RNA e do treinamento conforme necessário para otimizar o desempenho do modelo de acordo com seus requisitos específicos
