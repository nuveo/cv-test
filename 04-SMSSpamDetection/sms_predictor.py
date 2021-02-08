__author__ = 'Rafael Lopes Almeida'
__email__ = 'fael.rlopes@gmail.com'
__date__ = '07/02/2021'
'''
Create model to predict Spam our Ham SMS, predict SMS and export to csv file.
'''

import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ------------------------------------------------------------------------------------------
# Read dataset
PATH_CSV = './data/sms-hamspam-train.csv'
df = pd.read_csv(PATH_CSV, delimiter=';')

# Set Label data to catagorical
df['label'] = df['label'].astype('category')
df['label'] = df['label'].cat.codes # HAM = 0 | SPAM = 1
df = df.dropna() # Remove NaN rows


# ------------------------------------------------------------------------------------------
# Train/Test split 80%/20%
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=8)

# Initialize TF-IDF and transform data
'''
Using TF-IDF (term frequency–inverse document frequency)
to statistically calculate how important aword is to a 
document in a collection or corpus.
'''
vectorizer = TfidfVectorizer(min_df = 5)

# Apply TF-IDF on train set
X_train_transformed = vectorizer.fit_transform(X_train)
X_train_transformed_with_length = add_feature(X_train_transformed, X_train.str.len())

# Apply TF-IDF on test set
X_test_transformed = vectorizer.transform(X_test)
X_test_transformed_with_length = add_feature(X_test_transformed, X_test.str.len())


# ------------------------------------------------------------------------------------------
# Initialize SVC
clf = SVC(C = 10000)

# Train model 
clf.fit(X_train_transformed_with_length, y_train)

# Predict values using model
y_predicted = clf.predict(X_test_transformed_with_length)


# ------------------------------------------------------------------------------------------
# Evaluate accuracy
accuracy = accuracy_score(y_test, y_predicted)

# Evaluate precision
precision = average_precision_score(y_test, y_predicted)

# Confusion matrix
true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, y_predicted).ravel()

# Report
print(pd.DataFrame(confusion_matrix(y_test, y_predicted),
             columns=['Predicted Spam', 'Predicted SMS'], index=['Actual Spam', 'Actual SMS']))
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')


# ------------------------------------------------------------------------------------------
# Read TEST dataset
df_test = pd.read_csv('./data/sms-hamspam-test.csv', delimiter=';')
df_test = df.dropna()

X_real = df_test['text']

# Initialize TF-IDF and transform data
X_real_transformed = vectorizer.transform(X_real)
X_real_transformed_with_length = add_feature(X_real_transformed, X_real.str.len())

# Predict values using model
y_real = clf.predict(X_real_transformed_with_length)
y_real = np.where(y_real == 0, 'Ham', 'Spam')


# ------------------------------------------------------------------------------------------
# Make list with results
csv_list = []
for len_data in range(0, len(y_real)):
    try:
        csv_list.append([y_real[len_data], X_real[len_data].encode('utf8')])
    except:
        pass

# Export results
with open('./output/output.csv', 'a', newline='') as myfile:
    wr = csv.writer(myfile)
    for to_print in csv_list:
        wr.writerow(to_print)

myfile.close()