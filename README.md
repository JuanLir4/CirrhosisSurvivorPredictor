
# Cirrhosis Survivor Predictor

Este script em Python utiliza as bibliotecas Keras e Scikit-learn para criar duas Redes Neurais Artificiais (RNA) que prevêem o estado de sobrevivência de pacientes com cirrose hepática. O conjunto de dados fornece como saída um número de dias e o evento que ocorreu após o período, sendo C (censurado), CL (censurado devido ao transplante de fígado) ou D (óbito).

## Requisitos

Certifique-se de ter as seguintes bibliotecas instaladas no seu ambiente Python:

- Keras
- pandas
- scikit-learn
- numpy

Certifique-se de ter o arquivo 'cirrhosis.csv' no mesmo diretório que este script. Este arquivo contém os dados necessários para treinar e testar as RNAs.

## Executando o Script

O script realizará as seguintes etapas:

### Carregamento e Limpeza de Dados

- Carrega os dados do arquivo 'cirrhosis.csv' e remove as linhas com valores ausentes.
- Substitui os valores ausentes em 'Cholesterol', 'Copper', 'Tryglicerides', 'Platelets' pela média correspondente.

### Pré-processamento de Dados

- Transforma os dados de entrada em formato numérico usando LabelEncoder da biblioteca scikit-learn.
- Corrige um possível erro em 'Edema' convertendo 'S' para 'Y'.
- Elimina a coluna 'Age' dos dados de entrada.

### Transformação One-Hot Encoding

- Utiliza o ColumnTransformer para aplicar one-hot encoding nas colunas necessárias dos dados de entrada.
- Aplica to_categorical para os dados de saída.

### Configuração da Primeira Rede Neural

- Define a arquitetura da RNA com três camadas ocultas, cada uma seguida por uma camada de dropout para evitar overfitting.
- A camada de saída possui três unidades, correspondendo aos três estados de sobrevivência, e utiliza a função de ativação softmax.

### Treinamento e Avaliação da Primeira Rede Neural

- Utiliza o otimizador Adam com uma taxa de aprendizado de 0.001 e decaimento de peso de 0.004.
- Compila o modelo usando 'categorical_crossentropy' como função de perda e métricas de precisão categórica.
- Avalia o desempenho do modelo usando validação cruzada com 10 dobras.

### Configuração da Segunda Rede Neural

- Define a arquitetura da segunda RNA com cinco camadas, cada uma seguida por uma camada de ativação relu.
- A camada de saída possui uma unidade e utiliza a função de ativação linear.

### Treinamento e Avaliação da Segunda Rede Neural

- Utiliza o otimizador Adam com uma taxa de aprendizado de 0.001.
- Compila o modelo usando 'mse' (Erro Quadrático Médio) como função de perda.

## Resultados

As duas RNAs foram treinadas e avaliadas. A primeira RNA, focada na classificação, alcança uma acurácia média de 72% nas 10 dobras de validação cruzada. A segunda RNA, focada na previsão de dias de sobrevivência, foi treinada por 5000 épocas e seus resultados podem ser avaliados com base nas previsões e média dos dias reais de sobrevivência.

Ambos os modelos fornecem insights valiosos sobre a previsão do estado de sobrevivência de pacientes com cirrose hepática.

## Detalhes Adicionais

### Detalhes da Primeira Rede Neural

- Três camadas ocultas, cada uma seguida por uma camada de dropout.
- Camada de saída com três unidades e função de ativação softmax.
- Otimizador Adam com taxa de aprendizado de 0.001 e decaimento de peso de 0.004.

### Detalhes da Segunda Rede Neural

- Cinco camadas ocultas, cada uma seguida por uma camada de ativação relu.
- Camada de saída com uma unidade e função de ativação linear.

### Ajustes Personalizados

Lembre-se de ajustar os parâmetros das RNAs e do treinamento conforme necessário para otimizar o desempenho dos modelos de acordo com seus requisitos específicos.
