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

- python3 main.py -t 'inference' -p 'data/sms-hamspam-test.csv'

5 - **Run script to train**

- python3 main.py -t 'train' -p 'data/sms-hamspam-train.csv'
