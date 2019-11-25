# Nome: Rafael Silva Del Lama
# Email: rafael.lama@usp.br

# Desafio: 02-FraudDetection

Todos os códigos foram desenvolvidos em Python utilizando o Jupyter Notebook.

A abordagem adotada foi treinar um modelo de Rede Neural Convolucional que receba como entrada uma imagem de assinatura de referencia e uma imagem de assinatura questionada, e classifique a assinatura questionada como fraude, genuína ou disfarçada.
Durante o treinamento, cada assinatura questionada foi apresentada a rede com todas as assinaturas de referencia. Já no teste, cada assinatura questionada será avaliada com cada uma das assinaturas de referencia e a classe atribuida a assinatura questionada será definida por voto majoritario.

# NuveoTrain 
Este arquivo contém o código utilizado para treinamento do modelo e o resultado da validação cruzada utilizando a base de treinaento fornecida. 

# NuveoTest
Este arquivo contém o código que será utilizado para teste. Para executar o teste, basta definir o diretorio da base de teste na variavel "test_dir" e executar o notebook.
O resultado do teste estará disponível no notebook e será gerado um arquivo "predictions.csv", com as respectivas colunas:
- signature: label da imagem da assinatura
- class: classe atribuida pelo voto majoritario
- Disguise: porcentagem de chance de ser uma assinatura disfarçada
- Genuine: porcentagem de chance de ser uma assinatura genuina
- Simulated: porcentagem de chance de ser uma assinatura fraudada

# Nuveo.json Nuveo.h5
Esses arquivos contém respectivamente a arquitetura do modelo e os pesos.

