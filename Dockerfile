FROM --platform=linux/amd64 pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS base-amd64
RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs && \
    rm -rf /var/lib/apt/lists/*

FROM --platform=linux/arm64 python:3.12 AS base-arm64
RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs && \
    rm -rf /var/lib/apt/lists/*

ARG TARGETARCH
FROM base-${TARGETARCH} AS final
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=UTF-8
WORKDIR /workspace

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ffmpeg libsm6 libxext6 \
      fontconfig fonts-noto-cjk fonts-noto-cjk-extra fonts-nanum && \
    fc-cache -fv && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY .env /workspace/.env
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY ./models/FZYTK.TTF /usr/local/lib/python3.12/site-packages/rapidocr/models
RUN playwright install --with-deps

COPY ./ /workspace

EXPOSE 8085

CMD ["python", "-u", "grpc_server.py"]