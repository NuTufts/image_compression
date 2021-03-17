FROM tensorflow/tensorflow:latest-gpu
RUN apt-get update
RUN apt-get install -y vim
RUN python3 -m pip install tfds-nightly
RUN python3 -m pip install tensorflow-probability
RUN python3 -m pip install Pillow