FROM ghcr.io/actions/actions-runner:2.329.0
ENV DEBIAN_FRONTEND=noninteractive

RUN sudo apt update
RUN sudo apt install -y software-properties-common
RUN sudo add-apt-repository -y ppa:deadsnakes/ppa
RUN sudo apt update
RUN sudo apt install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    procps \
    build-essential

RUN python3.12 --version
