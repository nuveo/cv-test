import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import utils as u
import os
import seaborn as sns

# text preprocessing modules
from nltk.tokenize import word_tokenize

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

nltk.download('punkt')

import re #regular expression


from wordcloud import WordCloud, STOPWORDS



# explore ham labeled sms
def collect_words(data, label):
    collected_words = " "

    # iterate through the csv file
    for val in data.sms_text[data["target"] == label]:

        # typecaste each val to string
        val = str(val)

        # split the value
        tokens = val.split()

        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        for words in tokens:
            collected_words = collected_words + words + " "

    return collected_words

def plot_wordcloud(ham_words, cloud_stopwords):
    wordcloud = WordCloud(
        width=1000,
        height=1000,
        background_color="black",
        stopwords=cloud_stopwords,
        min_font_size=10,
    ).generate(ham_words)

    # plot the WordCloud image
    plt.figure(figsize=(15, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()