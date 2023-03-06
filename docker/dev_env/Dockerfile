FROM nvidia/cuda:11.7.0-devel-ubuntu20.04
ARG USER_ID
ARG USER_NAME

ENV DEBIAN_FRONTEND=noninteractive
RUN \
  apt-get update && \
  apt-get install -y python3 python3-pip unzip curl \
    python3-venv cmake build-essential cmake-curses-gui

RUN \
  apt-get update && \
  apt-get install -y python3-dev

RUN pip3 install torch==1.13.1 
RUN pip3 install torchvision==0.14.1 
RUN pip3 install torchaudio==0.13.1
RUN pip3 install pybind11

RUN \
  apt-get update && \
  apt-get install -y git

RUN mkdir /CUDA
WORKDIR /CUDA
RUN git clone https://github.com/NVIDIA/cuda-samples.git
WORKDIR /CUDA/cuda-samples
RUN git checkout v11.6

RUN apt-get update && apt-get install -y xterm

# NOTE: Set for your GPU
ENV COMPUTE_CAPABILITY="8.6"

# Build optox
RUN mkdir /deps
WORKDIR /deps
RUN git clone https://github.com/abroun/optox.git
WORKDIR /deps/optox
RUN git checkout dev
RUN mkdir build
WORKDIR /deps/optox/build
RUN cmake -DWITH_PYTORCH=ON -DWITH_PYTHON=ON ../
RUN make -j 4
RUN make install

RUN pip3 install imageio
RUN pip3 install matplotlib

RUN \
  apt-get update && \
  apt-get install -y python3-tk

WORKDIR /src

# Setup username and id so that it matches host machine (otherwise files will be written as root)
RUN useradd -u ${USER_ID} ${USER_NAME}
ENV USER=${USER_NAME}
ENV USER=${USER_NAME}
ENV HOME=/home/${USER_NAME}
RUN mkdir -p /home/${USER_NAME}
RUN chown ${USER_NAME}:${USER_NAME} /home/${USER_NAME}
USER ${USER_NAME}