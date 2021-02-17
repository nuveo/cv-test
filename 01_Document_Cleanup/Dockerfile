FROM python:3.6

RUN apt update && \
    apt install -y vim git ffmpeg libsm6 libxext6 tesseract-ocr

RUN pip install opencv-python numpy pytesseract

WORKDIR /root