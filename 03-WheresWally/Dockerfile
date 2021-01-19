FROM nvidia/cuda
ENV DEBIAN_FRONTEND noninteractive # This is used because opencv installation asks for location

RUN apt-get update && apt-get install -y --no-install-recommends \
         python3 \
         python3-pip \
         python3-dev \
         python3-setuptools \
         python3-opencv && \
     rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip setuptools
RUN pip3 install cython
ADD requirements.txt .
RUN pip3 install -r requirements.txt