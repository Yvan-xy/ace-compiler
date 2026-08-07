FROM --platform=linux/amd64 nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749

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
        pip==24.3.1 \
        setuptools==75.6.0 \
        wheel==0.45.1 && \
    "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir \
        numpy==1.26.4 \
        onnx==1.17.0 \
        onnxruntime==1.20.1 \
        pandas==2.2.3 \
        psutil==6.1.1 \
        pybind11==2.13.6 \
        pytest==8.3.4 \
        PyYAML==6.0.2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git config --system --add safe.directory /app
RUN git config --system --add safe.directory /deps/phantom-ant
RUN git config --system --add safe.directory /deps/phantom-ant/.git

WORKDIR /app

CMD ["sleep", "infinity"]
