import pandas as pd
import re, nltk, sys
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def tokenize(dataFrame, dataset_type='train'):
    ps = PorterStemmer()

    if dataset_type == 'train':

        corpus = []
        for i in range(0, len(dataFrame)):
            review = re.sub('[^a-zA-Z]', ' ', dataFrame[1][i])
            review = review.lower()
            review = nltk.word_tokenize(review)
            
            review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)
    
    elif dataset_type == 'test':

        corpus = []
        for i in range(0, len(dataFrame)):
            review = re.sub('[^a-zA-Z]', ' ', dataFrame[0][i])
            review = review.lower()
            review = nltk.word_tokenize(review)
            
            review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
            review = ' '.join(review)
            corpus.append(review)
    
    return corpus


if __name__ == "__main__":

    train_path = '04-SMSSpamDetection/TrainingSet/sms-hamspam-train.csv'
    test_path = '04-SMSSpamDetection/TestSet/sms-hamspam-test.csv'
    
    train = pd.read_csv(train_path, sep='\t', header=None)
    #print(train.head())

    test = pd.read_csv(test_path, sep='\t', header=None)
    #print(test.head())

    tokenized_train = tokenize(train)
    tokenized_test = tokenize(test, 'test')

    cv = CountVectorizer(max_features = 2000)
    X_train = cv.fit_transform(tokenized_train).toarray()
    X_test = cv.fit_transform(tokenized_test).toarray()
    
    Y_train = pd.get_dummies(train[0])
    Y_train = Y_train.iloc[:,1].values

    print(train[0].head(),"\n y = ",Y_train) # ham = 0 and spam = 1

    spam_detect_model = MultinomialNB()
    spam_detect_model.fit(X_train, Y_train)

    predictions = spam_detect_model.predict(X_test)

    labels = []

    for predicted in predictions:
        if predicted:
            labels.append('spam')
        else:
            labels.append('ham')

    test.insert(0, "",labels, True)

    file_name = test_path.split('/')[-1]
    file_path = 'results/'

    test.to_csv((file_path + file_name), sep='\t', header=None, index=False)