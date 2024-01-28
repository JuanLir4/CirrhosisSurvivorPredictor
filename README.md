
# Cirrhosis Survivor Predictor

Este script em Python utiliza a biblioteca Keras e Scikit-learn para criar uma Rede Neural Artificial (RNA) que utiliza 17 características clínicas para prever o estado de sobrevivência de pacientes com cirrose hepática. O conjunto de dados fornece como saída um número de dias e o evento que ocorreu após o período, sendo C (censurado), CL (censurado devido ao transplante de fígado) ou D (óbito).

### Requisitos

Certifique-se de ter as seguintes bibliotecas instaladas no seu ambiente Python:

- Keras
- pandas
- scikit-learn
- numpy

Certifique-se de ter o arquivo 'cirrhosis.csv' no mesmo diretório que este script. Este arquivo contém os dados necessários para treinar e testar a RNA.

### Executando o Script

O script realizará as seguintes etapas:

1. **Carregamento e Limpeza de Dados:**
   - Carrega os dados do arquivo 'cirrhosis.csv' e remove as linhas com valores ausentes.
   - Substitui os valores ausentes em 'Cholesterol', 'Copper', 'Tryglicerides', 'Platelets' pela média correspondente.

2. **Pré-processamento de Dados:**
   - Transforma os dados de entrada em formato numérico usando LabelEncoder da biblioteca scikit-learn.
   - Corrige um possível erro em 'Edema' convertendo 'S' para 'Y'.
   - Elimina a coluna 'Age' dos dados de entrada.

3. **Transformação One-Hot Encoding:**
   - Utiliza o ColumnTransformer para aplicar one-hot encoding nas colunas necessárias dos dados de entrada.
   - Aplica to_categorical para os dados de saída.

4. **Configuração da Rede Neural:**
   - Define a arquitetura da RNA com três camadas ocultas, cada uma seguida por uma camada de dropout para evitar overfitting.
   - A camada de saída possui três unidades, correspondendo aos três estados de sobrevivência, e utiliza a função de ativação softmax.

5. **Treinamento e Avaliação do Modelo:**
   - Utiliza o otimizador Adam com uma taxa de aprendizado de 0.001 e decaimento de peso de 0.004.
   - Compila o modelo usando 'categorical_crossentropy' como função de perda e métricas de precisão categórica.
   - Avalia o desempenho do modelo usando validação cruzada com 10 dobras.

### Resultados

O modelo foi refinado e agora alcança uma acurácia média de 72% nas 10 dobras de validação cruzada. Este desempenho melhorado proporciona uma base mais robusta para a previsão do estado de sobrevivência de pacientes com cirrose hepática. No entanto, continua sendo uma área de desenvolvimento contínuo, e ajustes adicionais podem ser explorados para aprimorar ainda mais a precisão do modelo.

### Detalhes Adicionais

- **Detalhes da Rede Neural:**
  - Três camadas ocultas, cada uma seguida por uma camada de dropout.
  - Camada de saída com três unidades e função de ativação softmax.
  - Otimizador Adam com taxa de aprendizado de 0.001 e decaimento de peso de 0.004.

- **Ajustes Personalizados:**
  - Lembre-se de ajustar os parâmetros da RNA e do treinamento conforme necessário para otimizar o desempenho do modelo de acordo com seus requisitos específicos.

