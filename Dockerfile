# build command:
# DOCKER_BUILDKIT=1 docker build -t dleongsh/esc-service:v1.0.0 .

ARG PYTORCH_VERSION=2.0.1
ARG CUDA_VERSION=11.7
ARG CUDNN_VERSION=8

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONFAULTHANDLER 1
ENV TZ=Asia/Singapore

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get -y update && \
    apt-get -y upgrade && \
    apt -y update && \
    apt-get install --no-install-recommends -y gcc g++ libsndfile1 sox ffmpeg wget sox git vim && \
    apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove && \
    rm -rf /var/cache/apt/archives/

ADD ./requirements.txt .

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir Cython==0.29.35 && \
    python3 -m pip install --no-cache-dir torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117 && \
    python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace
COPY . .

CMD ["python", "src/gr_app.py"]
