# Stage 0: init source
ARG BASE_IMAGE="docker.educg.net/zb/ubuntu-arm64:22.04"
ARG PIP_SOURCE="https://mirrors.cernet.edu.cn/pypi/web/simple"
ARG TORCH_CPU_AARCH64_WHL_URL="https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl"
ARG TORCH_NPU_AARCH64_WHL_URL="https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.7.1/torch_npu-2.7.1.post4-cp311-cp311-manylinux_2_28_aarch64.whl"
ARG TORCH_MLIR_AARCH64_WHL_URL="https://repo.oepkgs.net/ascend/pytorch/vllm/torch/Torch-MLIR/aarch64/Python311/torch_mlir-0.0.1-cp311-cp311-linux_aarch64.whl"

# Stage 1: Install Python
FROM ${BASE_IMAGE} AS python-installer

RUN set -eux; \
    for file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do \
        [ -f "${file}" ] || continue; \
        sed -i \
            -e 's@http://archive.ubuntu.com/ubuntu@http://mirrors.cernet.edu.cn/ubuntu@g' \
            -e 's@https://archive.ubuntu.com/ubuntu@http://mirrors.cernet.edu.cn/ubuntu@g' \
            -e 's@http://security.ubuntu.com/ubuntu@http://mirrors.cernet.edu.cn/ubuntu@g' \
            -e 's@https://security.ubuntu.com/ubuntu@http://mirrors.cernet.edu.cn/ubuntu@g' \
            -e 's@http://ports.ubuntu.com/ubuntu-ports@http://mirrors.cernet.edu.cn/ubuntu-ports@g' \
            -e 's@https://ports.ubuntu.com/ubuntu-ports@http://mirrors.cernet.edu.cn/ubuntu-ports@g' \
            "${file}"; \
    done

# Python Environment variables
ENV PATH=/usr/local/python3.11.15/bin:${PATH}

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        apt-transport-https \
        ca-certificates \
        bash \
        curl \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        libnss3-dev \
        libgdbm-dev \
        liblzma-dev \
        libev-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/tmp/* \
    && rm -rf /tmp/*

# Install Python
RUN curl -fsSL https://repo.huaweicloud.com/python/3.11.15/Python-3.11.15.tgz -o /tmp/Python-3.11.15.tgz && \
    tar -xf /tmp/Python-3.11.15.tgz -C /tmp && \
    cd /tmp/Python-3.11.15 && \
    mkdir -p /usr/local/python3.11.15/lib && \
    ./configure --enable-shared --enable-shared LDFLAGS="-Wl,-rpath /usr/local/python3.11.15/lib" --prefix=/usr/local/python3.11.15 && \
    make -j $(nproc) && \
    make altinstall && \
    ln -sf /usr/local/python3.11.15/bin/python3.11 /usr/local/python3.11.15/bin/python3 && \
    ln -sf /usr/local/python3.11.15/bin/pip3.11 /usr/local/python3.11.15/bin/pip3 && \
    ln -sf /usr/local/python3.11.15/bin/python3 /usr/local/python3.11.15/bin/python && \
    ln -sf /usr/local/python3.11.15/bin/pip3 /usr/local/python3.11.15/bin/pip && \
    rm -rf /tmp/*

# Stage 2: Install CANN
FROM python-installer AS cann-installer

ARG PIP_SOURCE
ARG TORCH_CPU_AARCH64_WHL_URL
ARG TORCH_NPU_AARCH64_WHL_URL
ARG TORCH_MLIR_AARCH64_WHL_URL

RUN apt-get update && apt-get install --no-install-recommends -y \
        git \
        wget \
        gcc \
        g++ \
        make \
        cmake \
        zlib1g \
        openssl \
        unzip \
        pciutils \
        net-tools \
        libblas-dev \
        gfortran \
        patchelf \
        libblas3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Note: Install CANN runtime dependencies
RUN pip install --no-cache-dir --upgrade pip --index-url ${PIP_SOURCE}

# Note: Get the download link according to ARCH and download the installation package
RUN ARCH=$(case "$(dpkg --print-architecture)" in \
        "amd64") echo "x86_64" ;; \
        "arm64") echo "aarch64" ;; \
        *) echo "Unsupported architecture: $(dpkg --print-architecture)" && exit 1 ;; \
    esac) && \
    CANN_TOOLKIT_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-toolkit_8.5.0_linux-${ARCH}.run && \
    CANN_OPS_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-910b-ops_8.5.0_linux-${ARCH}.run && \
    wget --quiet --header="Referer: https://www.hiascend.com/" ${CANN_TOOLKIT_URL} -O ~/Ascend-cann-toolkit.run && \
    wget --quiet --header="Referer: https://www.hiascend.com/" ${CANN_OPS_URL} -O ~/Ascend-cann-ops.run 

# Note: Install CANN Toolkit Development Kit Package
RUN chmod +x ~/Ascend-cann-toolkit.run && \
    ~/Ascend-cann-toolkit.run --quiet --install --install-for-all && \
    rm -f ~/Ascend-cann-toolkit.run

# Note: Install CANN Ops Operator Package
RUN chmod +x ~/Ascend-cann-ops.run && \
    ~/Ascend-cann-ops.run --quiet --install --install-for-all && \
    rm -f ~/Ascend-cann-ops.run

# Stage 3: Install PyTorch, torch_npu, torch_mlir and triton-ascend
# Note: Install triton runtime dependencies
RUN pip3 install --no-cache-dir \
    pyyaml \
    setuptools \
    pybind11 \
    ninja \
    cmake \
    wheel \
    --index-url ${PIP_SOURCE}

# Note: Install PyTorch stack and triton-ascend
RUN set -eux; \
    case "$(dpkg --print-architecture)" in \
        "amd64") \
            echo "install PyTorch stack for the x86 architecture"; \
            pip3 install --no-cache-dir torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu; \
            pip3 install --no-cache-dir torch-npu==2.7.1.post2 --index-url ${PIP_SOURCE}; \
            ;; \
        "arm64") \
            echo "install PyTorch stack for the arm architecture from pinned wheels"; \
            pip3 install --no-cache-dir --index-url ${PIP_SOURCE} "${TORCH_CPU_AARCH64_WHL_URL}"; \
            pip3 install --no-cache-dir --index-url ${PIP_SOURCE} "${TORCH_NPU_AARCH64_WHL_URL}"; \
            pip3 install --no-cache-dir --index-url ${PIP_SOURCE} "${TORCH_MLIR_AARCH64_WHL_URL}"; \
            ;; \
        *) \
            echo "Unsupported architecture: $(dpkg --print-architecture)"; \
            exit 1; \
            ;; \
    esac; \
    pip3 install --no-cache-dir triton-ascend==3.2.1 --index-url ${PIP_SOURCE} --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
    
# Stage 4: Copy results from previous stages
FROM ${BASE_IMAGE} AS official-ubuntu

ARG PIP_SOURCE

RUN set -eux; \
    for file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do \
        [ -f "${file}" ] || continue; \
        sed -i \
            -e 's@http://archive.ubuntu.com/ubuntu@http://mirrors.cernet.edu.cn/ubuntu@g' \
            -e 's@https://archive.ubuntu.com/ubuntu@http://mirrors.cernet.edu.cn/ubuntu@g' \
            -e 's@http://security.ubuntu.com/ubuntu@http://mirrors.cernet.edu.cn/ubuntu@g' \
            -e 's@https://security.ubuntu.com/ubuntu@http://mirrors.cernet.edu.cn/ubuntu@g' \
            -e 's@http://ports.ubuntu.com/ubuntu-ports@http://mirrors.cernet.edu.cn/ubuntu-ports@g' \
            -e 's@https://ports.ubuntu.com/ubuntu-ports@http://mirrors.cernet.edu.cn/ubuntu-ports@g' \
            "${file}"; \
    done

# Python Environment variables
ENV PATH=/usr/local/python3.11.15/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/python3.11.15/lib:${LD_LIBRARY_PATH}

# Note: Toolkit Environment variables, obtained from /usr/local/Ascend/ascend-toolkit/set_env.sh
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV ASCEND_TOOLKIT_LATEST_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_LATEST_HOME}/lib64:${ASCEND_TOOLKIT_LATEST_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_LATEST_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_LATEST_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_LATEST_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_LATEST_HOME}/tools/aml/lib64/plugin:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/devlib:$LD_LIBRARY_PATH
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
ENV PYTHONPATH=${ASCEND_TOOLKIT_LATEST_HOME}/python/site-packages:${ASCEND_TOOLKIT_LATEST_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/profiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ascend_system_advisor/asys:$PATH
ENV PATH=${ASCEND_TOOLKIT_HOME}/tools/show_kernel_debug_data:${ASCEND_TOOLKIT_HOME}/tools/msobjdump:$PATH
ENV PATH=${ASCEND_TOOLKIT_LATEST_HOME}/bin:${ASCEND_TOOLKIT_LATEST_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_LATEST_HOME}/tools/ccec_compiler/bin:$PATH
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
ENV CMAKE_PREFIX_PATH=${TOOLCHAIN_HOME}/tools/tikicpulib/lib/cmake:${ASCEND_TOOLKIT_HOME}/lib64/cmake

SHELL [ "/bin/bash", "-c" ]

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        apt-transport-https \
        ca-certificates \
        bash \
        libc6 \
        libsqlite3-dev \
        gcc \
        g++ \
        make \
        cmake \
        git \
        vim \
        wget \
        jq \
        curl \
        build-essential \
        libnuma-dev \
        sudo \
        procps \
        sysstat \
        systemd \
        net-tools \
        iproute2 \
        openssl \
        grep \
        tree \
        rsync \
        tar \
        zip \
        unzip \
        findutils \
        python3-dev \
        zlib1g-dev \
        libzstd-dev \
        clang-15  \
        ccache \
        lld-15 \
        openssh-server \
        openssh-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/tmp/* \
    && rm -rf /tmp/* \
    && ln -s /usr/bin/clang-15 /usr/bin/clang \
    && ln -s /usr/bin/clang++-15 /usr/bin/clang++ 

COPY --from=cann-installer /usr/local/python3.11.15 /usr/local/python3.11.15
COPY --from=cann-installer /usr/local/Ascend /usr/local/Ascend
COPY --from=cann-installer /etc/Ascend /etc/Ascend
RUN printf '%s\n' '#!/usr/bin/env bash' 'true' > /etc/profile.d/copyright.sh \
    && mkdir -p /coursegrader/submit /coursegrader/testdata /coursegrader/persisted /coursegrader/dockerext

RUN git clone https://gitcode.com/liupengcheng2012/OJ.git /coursegrader/dockerext/OJ \
    && git -C /coursegrader/dockerext/OJ checkout dev-8.5.0

WORKDIR /coursegrader/dockerext/OJ

# Note: Set environment variables
RUN \
    CANN_TOOLKIT_ENV_FILE="/usr/local/Ascend/ascend-toolkit/set_env.sh" && \
    echo "source ${CANN_TOOLKIT_ENV_FILE}" >> /etc/profile && \
    echo "source ${CANN_TOOLKIT_ENV_FILE}" >> ~/.bashrc && \
    echo "alias ll='ls -l'" >> ~/.bashrc && \
    echo "bash /etc/profile.d/copyright.sh" >> ~/.bashrc && \
    chmod 555 /etc/profile.d/copyright.sh
    
ENTRYPOINT ["/bin/bash", "-c", "\
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    bash /etc/profile.d/copyright.sh && \
    exec \"$@\"", "--"]

