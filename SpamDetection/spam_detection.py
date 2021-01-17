# -*- coding: utf-8 -*-
"""Spam detection.ipynb
## Importando as bibliotecas necessárias
"""

import pandas as pd

import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('rslp')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import tensorflow as tf

from matplotlib import pyplot as plt
import seaborn as sns

import multiprocessing


# Variáveis globais

stop_words = set(stopwords.words("english"))
stemmer = nltk.stem.RSLPStemmer()
n_jobs = multiprocessing.cpu_count()-1
tfv = TfidfVectorizer( #TF-IDF vectorizer
          min_df = 10,
          max_df = 0.5,
          max_features = None,
          stop_words = stop_words, 
          ngram_range = (1,3)
    )


"""## Pre-processamento dos dados

1.   Removendo as stopwords presentes e palavras com tamanho menor ou igual a 2;
2.   Deixando todas as palavras em minúsculo;
3.   Removendo termos que possuem dígitos;
4.   Limpeza de caracteres especiais e dígitos;
5.   Stemização dos dados para remoção de inflexões;
"""

def preprocess(data):
  #construct a new list to store the cleaned text
  clean_text = []
  for (i, row) in data.iterrows():
      text = data['Text'][i]

      #remove special characters and digits
      text  = re.sub("(\\d|\\W)+|\w*\d\w*"," ", text)
      text = ' '.join(stemmer.stem(s.lower()) for s in text.split() if (not any(c.isdigit() for c in s)) and len(s) > 2 and s not in stop_words)

      clean_text.append(text)

  print(f'Exemplo de textos após o pré-processamento: \n {clean_text[5:10]}')

  return clean_text


# Função para Verorizar os dados

def tfidf_transform(tfv, text):

  #transform
  vec_text = tfv.fit_transform(text)

  #returns a list of words.
  words = tfv.get_feature_names()

  print(f'Tamanho do vocabulário: {len(words)}\n')
  print(f'Listando alguns termos: {words[1:10]}\n')

  return vec_text, words

# Função de Oversampling

def overSampling(X_train,y_train):
  smote = SMOTE()
  X_sm, y_sm = smote.fit_sample(X_train,y_train)

  y_sm = pd.Series(y_sm, name="Spam")
  print("Pós balanceamento:\n",y_sm.value_counts())
  return X_sm, y_sm

# Função para treinamento do modelo de Logistic Regression

def train_lr_model(X_sm, y_sm):
  folds = 10
  kfold = StratifiedKFold(folds)

  logreg_C = [1e-4, 1e-3, 1e-2, 0.5e-1, 1]
  best_c = logreg_C[0]
  best_score = 0
  best_accuracy = 0
  avg_scores = []

  for c in logreg_C:
      score = 0
      accuracy = 0
      for train_index, test_index in kfold.split(X_sm, y_sm):
          x_train_fold, x_test_fold = X_sm[train_index], X_sm[test_index]
          y_train_fold, y_test_fold = y_sm.iloc[train_index], y_sm.iloc[test_index]

          lr = LogisticRegression(C=c, random_state=31, max_iter=300)
          lr.fit(x_train_fold, y_train_fold)
          pred = lr.predict(x_test_fold)

          score += f1_score(y_test_fold, pred, average='weighted')
          accuracy += accuracy_score(y_test_fold, pred)


      score = score / folds # média
      avg_scores.append(score)
      accuracy = accuracy / folds
      if (score > best_score):
          best_score = score
          best_accuracy = accuracy
          best_c = c

  print(f'Melhor C: {best_c}. Resultou no F1 {best_score} e Acurácia {best_accuracy} durante o {folds}-fold')
  return lr

# Função para avaliação do modelo"

def evaluate_model(model, data, y):
  pred = model.predict(data)

  f1 = f1_score(y, pred, average='weighted')
  accuracy = accuracy_score(y, pred)
  precision = precision_score(y, pred, average='weighted')
  recall = recall_score(y, pred, average='weighted')

  return (f1, accuracy, precision, recall, pred)

# Função para avaliação do modelo

def evaluate(model, X_test, y_test):
  train_f1, train_accuracy, train_precision, train_recall, pred_train = evaluate_model(model, X_test, y_test)

  print(f'O melhor modelo resultou um desempenho no treino de F1: {train_f1}, Acurácia: {train_accuracy}, Precisão: {train_precision}, Recall: {train_recall}')

  return pred_train

# Função para plotar a matriz de confusão

def plot_confusion_matrix(cm):
  plt.figure(figsize = (5,3))
  sns.heatmap(cm, annot=True, fmt='d')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')


# Função para treinamento do modelo RandomForest

def train_RF_model(X_sm, y_sm):
  rf = RandomForestClassifier(random_state=42)
  rf_params = { 'max_depth': [10,15,25,35], 'max_leaf_nodes': [100,200,500,1000] }
  grid = GridSearchCV(rf, rf_params, 'f1_weighted', cv=10, n_jobs=n_jobs)
  grid.fit(X_sm, y_sm)
  return grid


def main():
  # Lendo os dados
  data = pd.read_csv("./TrainingSet/sms-hamspam-train.csv", sep="\t", error_bad_lines=False, names=['Spam','Text'])

  # Análise dos dados
  print("Balanceamento das variáveis (ham, Spam)",data.Spam.value_counts())

  # Pre-processamento
  clean_text = preprocess(data)

  # Podemos ver que o corpus possui um vocabulário com 826 palavras diferentes:
  vec_title, words = tfidf_transform(tfv, clean_text)

  # Transformando as categorigas `ham` e `Spam`
  # Vamos converter as categorias em valores numéricos para poderem ser aplicadas nos algoritmos de classificação:
  print(pd.Categorical(data.Spam).codes)

  # Separando dos dados em treino e teste
  X_train, X_test, y_train, y_test = train_test_split(vec_title, pd.Series(pd.Categorical(data.Spam).codes), test_size=0.3)

  # Oversampling
  X_sm, y_sm = overSampling(X_train,y_train)

  # Training lr model
  lr = train_lr_model(X_sm, y_sm)

  # Evaluate lr model
  pred_train = evaluate(lr, X_test, y_test)

  # Ploting confunsion matrix
  cm = tf.math.confusion_matrix(labels=y_test,predictions=pred_train)

  plot_confusion_matrix(cm)

  # Training RF model
  rf = train_RF_model(X_sm, y_sm)

  # Evaluate rf model
  pred_train = evaluate(rf, X_test, y_test)

  # Ploting confunsion matrix
  cm = tf.math.confusion_matrix(labels=y_test,predictions=pred_train)
  plot_confusion_matrix(cm)

  # Predizendo os dados de teste
  data_test = pd.read_csv("./TestSet/sms-hamspam-test.csv",sep="\t", error_bad_lines=False, names=['Text'])
  data_test.head()

  # Pre-processamento
  clean_text = preprocess(data_test)

  # Vetorização dos dados
  matrix = tfv.transform(clean_text)

  # Predição
  data_test.insert(0, 'Spam', pd.Series(rf.predict(matrix)))
  data_test['Spam'] = data_test["Spam"].apply(lambda res : "ham" if res == 0 else "spam")

  # Salvando dados
  data.to_csv("./Results/results.csv", header=False)

if __name__ == '__main__':
    main()



