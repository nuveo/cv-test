# Spam Detection

## Resultados

O arquivo .csv contendo o dataset de treino com as mensagens classificadas está na pasta results.

## Rodar o projeto

Lembre-se de instalar as dependências:

```
pip install -r requirements.txt
```

Para rodar o projeto você deve utilizar o seguinte comando:

```
python main.py 04-SMSSpamDetection/TrainingSet/sms-hamspam-train.csv 04-SMSSpamDetection/TestSet/sms-hamspam-test.csv
```

## Técnicas utilizadas

Para solucionar o problema de detecção de mensagens de spam, utilizei o classificador Nayve Bayes.

Utilizei algumas referências que podem ser conferidas em:

https://www.kaggle.com/amitkarmakar41/spamclassifier

https://www.kaggle.com/pramodsivakumar/spam-sms-classifier-98-3-accuracy

https://www.kdnuggets.com/2020/07/spam-filter-python-naive-bayes-scratch.html