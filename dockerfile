# Base image ubuntu 18.04 com suporte para cuda 11.1
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

# Instalar pacotes de gerenciamento de repositórios 
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget

RUN apt-get -y install python3.8

# Define o diretório de trabalho
WORKDIR /home/tutorial/app

# Copiar todos os arquivos do diretório atual para o diretório de trabalho dentro da imagem
COPY . .

# Instalar as dependencias
RUN apt-get -y install zlib1g-dev python3-dev python3-setuptools

RUN apt-get -y update

RUN apt-get -y install libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    libharfbuzz-dev libfribidi-dev libxcb1-dev libc6

RUN apt-get -y install python3-pip 

RUN python3 -m pip install --upgrade pip click

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN python3 -m pip install pillow==8.4.0 typing-extensions==4.1.1 wheel==0.37.1 torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN python3 -m pip install -r requirements.txt

RUN pip3 install --upgrade requests && rm -rf /var/lib/apt/lists/*
