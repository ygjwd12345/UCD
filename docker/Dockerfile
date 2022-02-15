#FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0"

RUN apt-get clean && apt-get update && apt-get install -y \
    tree \
    vim \
    python3-tk \
    git \
    unzip \
    wget \
    libgl1-mesa-dev\
    libgtk2.0-dev
RUN pip install --no-cache-dir Cython \
    wandb \
    torch-encoding \
    torchvision \
    ninja \
    scikit-image \
    opencv-python \
    pandas \
    tqdm \
    scikit-learn \
    inplace_abn \
    screen \
    tensorboardX \

RUN pip install pycocotools \
    opencv-contrib-python
WORKDIR "/"
RUN git clone https://github.com/NVIDIA/apex
WORKDIR "/apex"
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#WORKDIR "/workspace"
WORKDIR "/home"