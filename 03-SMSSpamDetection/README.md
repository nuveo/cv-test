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

# Solution

To detect spam among the received sms, a Naive Bayes classifier was used. It is a methodology already well used in the classification of texts taking as a reference the probability of words in each possible category.

The first step in modeling this type of problem is to extract the "features" from the sms. These will be the words of the English language. The features along with the labels will be used in training the model. 80% of the data for training and 20% for test data were separated. These test data allowed to select the classification threshold so that Recall is 100%.






