# SMS Spam Detection

## Description

The challenge solution consists of training a predictive model to detect messages that are spam or not.  

To achieve this result, the data were previously separated into training and test sets, respectively 80% and 20% of the data, and a TF-IDF vectorizer was selected to select only as words that actually bring importance to the model.  

An SVC was used to classify the messages using the data extracted from the vectorizer. The trained model averages an accuracy value of 98%.  

At the end, prediction is made on new messages never seen by the model before. This prediction is exported to a file called output.csv which is found in the folder called output.  

## Usage

Install requiriments

```
pip install -r requirements.txt
```