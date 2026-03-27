FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PYTHONPATH=/app
ENV ACE_DATASET_DIR=/dataset
ENV CIFAR10_DIR=/opt/datasets/cifar-10-batches-bin
ENV CIFAR100_DIR=/opt/datasets/cifar-100-binary

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        wget \
        time \
        build-essential \
        gcc \
        g++ \
        clang \
        cmake \
        ninja-build \
        pkg-config \
        libtool \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python3-setuptools \
        python3-wheel \
        pybind11-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libssl-dev \
        libgmp-dev \
        libomp-dev \
        libomp5 \
        libntl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        PyYAML \
        matplotlib \
        numpy \
        onnx \
        onnxruntime \
        pandas \
        psutil \
        pybind11 \
        pytest \
        torch \
        torchvision

WORKDIR /app

COPY . /app

RUN mkdir -p /dataset /opt/datasets && \
    ln -sfn /dataset /app/dataset && \
    ln -sfn /dataset/cifar-10-batches-bin /opt/datasets/cifar-10-batches-bin && \
    ln -sfn /dataset/cifar-100-binary /opt/datasets/cifar-100-binary

VOLUME ["/dataset"]

CMD ["/bin/bash"]
