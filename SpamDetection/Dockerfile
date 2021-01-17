FROM python:3.7

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

RUN pip install tensorflow-gpu

RUN pip install seaborn

RUN pip install imblearn

RUN python3 -m nltk.downloader punkt \
                               words \
                               stopwords

CMD python spam_detection.py