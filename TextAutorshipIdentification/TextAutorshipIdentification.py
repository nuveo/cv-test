#!/usr/bin/env python
# coding: utf-8

# # Text autorship identification
# 
# For this prove, we will test ideas to feature extraction and data pre-processing that will be taken from two papers found in arxiv:
# 
# [1] [TEXT CLASSIFICATION FOR AUTHORSHIP
# ATTRIBUTION ANALYSIS](https://arxiv.org/pdf/1310.4909.pdf)
# 
# [2] [A Machine Learning Framework for Authorship Identification From Texts](https://arxiv.org/pdf/1912.10204.pdf)
# 
# In our tests, we will check how the accuracy of our final model behaves when presented to these ideas.
# 
# This notebook is part of a practical prove provided from NUVEO.

# ### Imports
# 
# For the purpose of this task, the following libraries will be helping us to present the ideas through this notebook.

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
import re 

import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize 

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_extraction.text import CountVectorizer

get_ipython().run_line_magic('matplotlib', 'inline')
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')


# ### Data reading

# In[ ]:


author_data = pd.read_csv('./TrainingSet/text-authorship-training.csv')
author_data.head()


# In[ ]:


labels = 'EAP','MWS','HPL'
author_data['author'].value_counts()

sizes = list(author_data['author'].value_counts())

explode=(0.05, 0, 0)

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, shadow=True,autopct='%1.1f%%', startangle=140)
ax.axis('equal')
plt.show()


# As we can see, the authors distribution looks be very balanced

# ### Feature Extraction
# 
# As mentioned at the beginning, the main ideas for feature extraction were taking from two papers.From the first one, we will take features that involves punctuation and features related directly with the text and letters. From the second one, we will count the occurence of some certains words. To sum up, here it is the features that we will implement:
# 
# **Punctuation and Phraseology:**
# 
# - Number of periods
# - Number of commas
# - number of question marks
# - Number of colons
# - Number of semi-colons
# - Number of blanks 
# - Number of exclamation marks
# - Number of dashes
# - Number of underscores
# - Number of brackets
# - Number of words
# - Number of sentences
# - Number of characters
# 
# **Lexical:**
# - Number of and
# - Number of but
# - Number of however
# - Number of if
# - Number of that
# - Number of more
# - Number of might
# - Number of this
# - Number of very
# 

# In[ ]:


def feature_extraction_punc_phras(author_data):
    """ Extract features from punctuation and phraseology
    
    Args:
        author_data: the dataframe with the text information
    Return:
        author_data: the dataframe that now contains the new features
    """
    
    author_data['num_periods'] = author_data['text'].apply(lambda x: x.count('.'))
    author_data['num_commas'] = author_data['text'].apply(lambda x: x.count(','))
    author_data['num_questions'] = author_data['text'].apply(lambda x: x.count('?'))
    author_data['num_colons'] = author_data['text'].apply(lambda x: x.count(':'))
    author_data['num_semi-colons'] = author_data['text'].apply(lambda x: x.count(';'))
    author_data['num_blanks'] = author_data['text'].apply(lambda x: x.count(' '))
    author_data['num_exclamation'] = author_data['text'].apply(lambda x: x.count('!'))
    author_data['num_dashes'] = author_data['text'].apply(lambda x: x.count('-'))
    author_data['num_underscores'] = author_data['text'].apply(lambda x: x.count('_'))
    author_data['num_brackets'] = author_data['text'].apply(lambda x: x.count('[') * 2)
    
    author_data['num_words'] = author_data['text'].apply(lambda x: len(x.split(' ')))
    author_data['num_sentences'] = author_data['text'].apply(lambda x: len(sent_tokenize(x)))
    author_data['num_characters'] = author_data['text'].apply(lambda x: len(x) - x.count(" "))
    return author_data

def feature_extraction_lexical(author_data):
    """ Extract features from lexical
    Args:
        author_data: the dataframe with the text information
    Return:
        author_data: the dataframe that now contains the new features
    """
    lexical_terms = ["and", "but", "however","if","that","more","might","this","very"]
    
    for lexical in lexical_terms:
        author_data[f'num_of_{lexical}'] = author_data['text'].apply(lambda x: x.count(lexical))
    
    return author_data


# After all feature extaction, our data now looks like this:

# In[ ]:


author_data = feature_extraction_punc_phras(author_data)
author_data = feature_extraction_lexical(author_data)
author_data.head()


# #### check how one of those metrics behaves

# ### Data Pre-processing
# 
# Following the ideas from both papers, to process our data we will use 3 methods:
# 
# **1. Tokenization:** Tokenization is the method of splitting a stream of text into meaningful element. For us, these meaningful elements will be taken following the stopwords/punctuation idea, in other words, only not stopwords/punctuation will continue in our stream of text. We also will put all words in lower case.
# 
# **2. Stemming:** Stemming is the process of reducing the inflected words to their root or base form known as stem.
# The stem may not be same as the morphological root of that word. [1] uses WordNet for stemming. This stemmer adds functionality to the simple pattern-based stemmer SimpleStemmer by checking to see if possible stems are actually present in Wordnet. 
# 
# **3. Top K Words:** Mapping all words present in the dataset, build a corpus which, this corpus, contains the frequency of each word.

# In[ ]:


def tokenize_normalization(message):
    """
    Args:
    Returns:
    """
    stopwords.words('english')
    word_tokens = word_tokenize(message) 
    filtered_stop = [word for word in word_tokens if not word in stopwords.words('english')]
    filtered_punctuation = [word for word in filtered_stop if not word in string.punctuation]

    return " ".join([word.lower() for word in filtered_punctuation])
    
def stemming_normalization(message):
    """Stemming the given message using WordNetLemmatizer()
    Args:
        message: sms string
    Returns:
        return a stemmed sentence
    """
    wnl = nltk.WordNetLemmatizer()
    splited_message = message.split()
    
    return " ".join([wnl.lemmatize(word) for word in splited_message])


# #### Applying Tokenization

# In[ ]:


author_data['tokenized_text'] = author_data['text'].apply(lambda x: tokenize_normalization(x))
author_data.head()


# #### Applying stemming normalization

# In[ ]:


author_data['stemmed_text'] = author_data['tokenized_text'].apply(lambda x: stemming_normalization(x))
author_data.head()


# Applying CountVectorizer

# In[ ]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(author_data.stemmed_text)
count_vect_df = pd.DataFrame(X.toarray())
count_vect_df.head()


# In[ ]:


print("At the final, vectorizer corpus has a total of {} words".format(len(count_vect_df.columns)))


# In[ ]:


final_data = pd.concat([author_data, count_vect_df], axis=1, sort=False)
final_data.drop(['author','text','id','tokenized_text','stemmed_text'], axis=1, inplace=True)
final_data.head()


# In[ ]:


txt_train, txt_test, label_train, label_test = train_test_split(final_data, author_data['author'], test_size=0.1)


# # <span style="color:red">Note</span>
# 
# The first idea here was test the data using SVM classifier, just like [1] did. But it was taking a long time to train and I stoped the process to try something else.
# 
# My second idea was try to use grid search just like I did in the [first](https://github.com/joaolcaas/cv-test/blob/JoaoFelipe/SMSSpamDetection/SMSSpamDetection.ipynb) prove. But some erros happened (i.e. memory error) and I was not able to run this type of strategy due the (1) big data and (2) small memory at the train setup computer.
# 
# My third option was decrease the number of parameters for search, which you can check below

# In[ ]:


model = RandomForestClassifier()

parameters = {
    'n_estimators'      : [50, 150,340],
    'max_depth'         : [8, 30, None],
}
gs = GridSearchCV(model, parameters)
gs_fit = gs.fit(txt_train, label_train)


# In[ ]:


print(gs_fit.best_params_)
print(gs_fit.best_score_)


# In[ ]:


clf_prediction = clf.predict(txt_test)
print("Accuracy: {}".format( round((clf_prediction==label_test).sum() / len(clf_prediction),3)))


# ### TF-IDF comparisson
# 
# None of both papers does use tf-idf approach. In order to compare both approaches, we will train the same data, changing CountVectorizer() per TfidfVectorizer() and see if the accuracie changes

# In[ ]:


# doing this due the memory error
del final_data
del gs
del gs_fit


# In[ ]:


vectorizer = TfidfVectorizer()
tf_idf_vec = vectorizer.fit_transform(author_data.stemmed_text)
tfidf_vect_df = pd.DataFrame(tf_idf_vec.toarray())
tfidf_vect_df.head()


# In[ ]:


print("At the final, vectorizer tf-idf has a total of {} words".format(len(tfidf_vect_df.columns)))


# In[ ]:


final_data = pd.concat([author_data, count_vect_df], axis=1, sort=False)
final_data.drop(['author','text','id'], axis=1, inplace=True)
final_data.head()


# In[ ]:


txt_train, txt_test, label_train, label_test = train_test_split(final_data, author_data['author'], test_size=0.1)


# In[ ]:


model = RandomForestClassifier()

parameters = {
    'n_estimators'      : [50, 150,340],
    'max_depth'         : [8, 30, None],
}
gs = GridSearchCV(model, parameters)
gs_fit = gs.fit(txt_train, label_train)


# In[ ]:


print(gs_fit.best_params_)
print(gs_fit.best_score_)


# In[ ]:


clf_prediction = clf.predict(txt_test)
print("Second Accuracy: {}".format( round((clf_prediction==label_test).sum() / len(clf_prediction),3)))


# ### Filling in the testset

# In[ ]:

'''
test_set = pd.read_csv('./TestSet/text-authorship-test.csv')
test_set.head()


# In[ ]:


test_set = feature_extraction_punc_phras(test_set)
test_set = feature_extraction_lexical(test_set)
test_set.head()


# In[ ]:


test_set['tokenized_text'] = test_set['text'].apply(lambda x: tokenize_normalization(x))
test_set.head()


# In[ ]:


test_set['stemmed_text'] = test_set['tokenized_text'].apply(lambda x: stemming_normalization(x))
test_set.head()


# In[ ]:


X_test = vectorizer.transform(test_set.stemmed_text)
vector_test = pd.DataFrame(X_test.toarray())
vector_test.head()


# In[ ]:


test_set.drop(['stemmed_text'],axis=1, inplace=True)
final_data_test = pd.concat([test_set, vector_test], axis=1, sort=False)
final_data_test.drop(['text','id','tokenized_text'], axis=1, inplace=True)
final_data_test.head()


# In[ ]:


final_set = {"author":[],"text":test_set.text}


# In[ ]:


final_data_test


# In[ ]:


predictions = clf.predict(final_data_test)


# In[ ]:


final_set["author"] = predictions
final_ans = pd.DataFrame(final_set)
final_ans.to_csv('final_ans.csv')
final_ans.head()

'''