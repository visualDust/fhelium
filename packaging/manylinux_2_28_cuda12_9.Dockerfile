ARG MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_x86_64@sha256:f854c50adf7b7a325bc4794316f3758d387a41d61f9e2ebca0f26c7dc8f761d4
FROM ${MANYLINUX_IMAGE}

RUN dnf --assumeyes install dnf-plugins-core \
    && dnf config-manager --add-repo \
        https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
    && dnf --assumeyes install \
        cuda-compiler-12-9-12.9.1-1 \
        cuda-cudart-devel-12-9-12.9.79-1 \
    && dnf clean all \
    && rm -rf /var/cache/dnf

ENV CUDA_HOME=/usr/local/cuda-12.9
ENV CUDACXX=/usr/local/cuda-12.9/bin/nvcc