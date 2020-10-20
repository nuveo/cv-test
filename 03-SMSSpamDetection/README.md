# SMS Spam Detector

This is the algorithm to detect spam. It was assumed that a `spam` message will never be considered ham. However, a normal message (`ham`) can be somethimes be allocated at the `spam` box.

# How to run?  

1 - Install dependences

- Install docker and run in command line:
    - docker build . -t spam_detect:v1
    - docker run -it --name cont spam_detect:v1 bash

2 - **Result**

- The result of test is in output_inference.csv 

Inside the container, the scripts can be ran. Before:

- cd src

4 - **Run script to make inference**

- python main.py -t 'inference' -p 'data/sms-hamspam-test.csv'

5 - **Run script to train**

- python main.py -t 'train' -p 'data/sms-hamspam-train.csv'

# Solução

Para detectar spams dentre os sms's recebidos, foi usado um classificador Naive Bayes. É uma metodologia já bem utilizada na classificação de textos tomando como referência a probabilidade das palavras em cada categorias possível.  

O primeiro passo para modelar esse tipo de problema é extrair as "features" dos sms's. Essas serão as palavras do idioma inglês. As features junto com os rótulos serão usadas no treinamento do modelo. Foram separados 80% dos dados para treinamento e 20% para dados de teste. Esses dados de testes permitiram selecionar o threshold de classificação para que o Recall seja de 100%.






