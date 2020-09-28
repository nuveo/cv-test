# Description

The SMS Ham-Spam detection dataset is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, considering the train and test data. The tagging standard was defined as `ham` (legitimate) or `spam`. 

The `train` and `test` files are formatted using the standard of one message per line. Each line is composed by two columns: one with label (`ham` or `spam`) and other with the raw text. Here are some examples:

```
ham   What you doing?how are you?
ham   Ok lar... Joking wif u oni...
ham   dun say so early hor... U c already then say...
ham   MY NO. IN LUTON 0125698789 RING ME IF UR AROUND! H*
ham   Siva is in hostel aha:-.
ham   Cos i was out shopping wif darren jus now n i called him 2 ask wat present he wan lor. Then he started guessing who i was wif n he finally guessed darren lor.
spam   FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop
spam   Sunshine Quiz! Win a super Sony DVD recorder if you canname the capital of Australia? Text MQUIZ to 82277. B
spam   URGENT! Your Mobile No 07808726822 was awarded a L2,000 Bonus Caller Prize on 02/09/03! This is our 2nd attempt to contact YOU! Call 0871-872-9758 BOX95QU
```

    Note: messages are not chronologically sorted.

For evaluation purposes, the `test` dataset does not prosent the categories (`ham`, `spam`). Therefore, the `train` data is the full source of information for this test.

# Objective

The goal of the this test is to achieve a NLP model that can correctly manage the incoming messages on SMS format (`ham` or `spam`). Considering a real scenario, assume that a regular person does not agree to see a `spam` message, however, a normal message (`ham`) can be somethimes be allocated at the `spam` box.

# Important details

- The dataset was split in order to have unseen data for analysis. We took 15% of the total data (randomly)
- Replicate the data format for submission, i.e. the answer must be provided as a CSV file with the detect class in the first column and the text in the second column, similarly to what is provided in the `TrainingSet` file
- The `TestSet` will be used for evalution, therefore the candidate must fullfiled the first column with the predicted classes (`ham` or `spam`)
- Pay attention to the real case scenario that was described in the Objective section. This may drive the problem solving strategy :wink:.
- This test does not require a defined set of algorithms to be used. The candidate is free to choose any kind of data processing pipeline to reach the best answer.

# How to Run

The SMS Ham-Spam test is described in `SMSSpamDetection.ipynb` file. All dataset modifications and ideas over the NLP tasks are described there. If you want to test this approach, you only need to:

- Download the provided [dataset](https://drive.google.com/file/d/1LhH_5ULfyrobD60SZqIfoI56eV3HuDNI/view) from NUVEO. 
- Change both train/test path inside the notebook.
- install the libraries by doing `pip install -r requirements.txt` 