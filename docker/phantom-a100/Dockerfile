FROM --platform=linux/amd64 docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749

ARG ACE_PHANTOM_DEFINITION_SHA256
LABEL org.ace.phantom.definition-sha256=${ACE_PHANTOM_DEFINITION_SHA256}

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/ace-venv
ENV PATH=/opt/ace-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV PYTHONPATH=/app
ENV ACE_DATASET_DIR=/inputs/dataset
ENV CIFAR10_DIR=/inputs/dataset
ENV CMAKE_CUDA_ARCHITECTURES=80
ENV CUDAARCHS=80
ENV ACE_PHANTOM_TOOLCHAIN=12.4.1-sm80

COPY python-requirements.lock /tmp/python-requirements.lock

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        binutils \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        curl \
        file \
        g++ \
        gcc \
        gdb \
        git \
        jq \
        libgmp-dev \
        libntl-dev \
        libomp-dev \
        libprotobuf-dev \
        libssl-dev \
        libtool \
        ninja-build \
        nlohmann-json3-dev \
        openssh-server \
        pkg-config \
        protobuf-compiler \
        pybind11-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-venv \
        python3-wheel \
        ripgrep \
        time \
        util-linux \
        wget \
        zlib1g-dev && \
    python3 -m venv "${VIRTUAL_ENV}" && \
    "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir \
        --requirement /tmp/python-requirements.lock && \
    "${VIRTUAL_ENV}/bin/pip" check && \
    rm /tmp/python-requirements.lock && \
    rm -f /etc/ssh/ssh_host_* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git config --system --add safe.directory /app
RUN git config --system --add safe.directory /deps/phantom-ant
RUN git config --system --add safe.directory /deps/phantom-ant/.git

WORKDIR /app

CMD ["sleep", "infinity"]
