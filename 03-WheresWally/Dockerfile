FROM python:3.7

ADD . /app

WORKDIR /app

RUN apt update

RUN apt install libgl1-mesa-glx\
    			'ffmpeg'\
    			'libsm6'\
    			'libxext6'\
			'libxcb-xinerama0' -y

RUN pip install opencv-python numpy scipy

RUN pip install -r requirements.txt

CMD python WIW.py
