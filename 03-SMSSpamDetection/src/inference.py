
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

import pickle
import json


MODEL_PATH = "all_model.pkl"


def get_model(model_path):
    """Get the model."""
    with open('all_model.pkl', 'rb') as f:
        enc, vec, model = pickle.load(f)

    return enc, vec, model

def read_json(file_name):
    with open(file_name, "r") as read_file:
        data = json.load(read_file)
    return data

def predict(data, model_path):
        """For the input, do the predictions and return them."""

        print("Get encoding, model and vectorizer")
        enc, vec, model = get_model(model_path)

        print("Extract features/Vetorizer")
        data_trans = vec.transform(data)

        print("Get threshold")
        hyperparameters = read_json("hyperparameters.json")
        if "threshold" in hyperparameters:
            threshold = hyperparameters["threshold"]
        else:
            threshold = 0.5
        print("Make inference")
        inference_result =  model.predict_proba(data_trans)[:, 1] > threshold

        print("Decoding")
        final_result = enc.inverse_transform(inference_result) 

        return final_result


def inference(data_path):

    # load csv sms data
    df = pd.read_csv(data_path,  sep='\t', names=['sms'])

    # inference
    result = predict(df['sms'], MODEL_PATH)

    # build output csv
    df.insert(loc=0, column='spam', value=result)
    df.to_csv('output_inference.csv', sep='\t', header=False, index=False)

 
    
