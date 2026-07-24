FROM ubuntu:22.04

# Install only essential packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    wget \
    git \
    jq \
    nlohmann-json3-dev \
    libboost-all-dev \
    libdivsufsort-dev \
    && rm -rf /var/lib/apt/lists/*

# Install SDSL-lite (system-wide)
RUN git clone --depth 1 https://github.com/simongog/sdsl-lite.git /tmp/sdsl-lite \
    && cd /tmp/sdsl-lite \
    && ./install.sh /usr/local/ \
    && rm -rf /tmp/sdsl-lite

# Set working directory
WORKDIR /workspace

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
