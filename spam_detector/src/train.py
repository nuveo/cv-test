from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
import pickle
import json

def train(data_csv):
    df_smss = pd.read_csv(data_csv, sep='\t', names=['label', 'sms'])

    # encondig label column as 1 (spam) and 0 (ham)
    enc = preprocessing.LabelBinarizer()
    df_smss['label'] = enc.fit_transform(df_smss[['label']].values)

    # separate train and test data
    X = df_smss['sms']
    y = df_smss['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

    # feature extract
    cvec = TfidfVectorizer(stop_words ='english')
    cvec.fit(X_train) 
    X_train_t = cvec.transform(X_train)
    X_test_t = cvec.transform(X_test)

    # Fit model
    model = BernoulliNB()
    model.fit(X_train_t,y_train)

    # decide threshold
    y_pred_prob = model.predict_proba(X_test_t)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob[:,1])
    # max recall to no receive spam email in ham box.
    ix = np.argmax(recall)
    fscore = (2 * precision * recall) / (precision + recall)
    print('Best Threshold=%f, F-Score=%.3f, Recall=%f, Precision=%f' % (thresholds[ix], fscore[ix], recall[ix], precision[ix]))
    with open("hyperparameters.json", "w") as write_file:
        data = {"threshold" : thresholds[ix]}
        json.dump(data, write_file)

    # save model
    with open("all_model.pkl", 'wb') as fout:
        pickle.dump((enc, cvec, model), fout)


