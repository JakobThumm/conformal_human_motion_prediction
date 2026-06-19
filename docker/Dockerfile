FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up locale
RUN apt-get update && apt-get install -y locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    git \
    vim \
    build-essential \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-zstandard \
    sudo \
    gosu \
    net-tools \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Create Python virtual environment directory
# The environment is created at runtime by docker/setup_env.sh
ENV VIRTUAL_ENV=/workspace/unc
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# The GPU is often shared -> avoid JAX's 75% pre-allocation OOM
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# Set default user/group IDs (will be overridden at runtime)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=user

# Create entrypoint script for user setup
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH

# Set entrypoint to handle user creation and setup
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set default command
CMD ["/bin/bash"]
