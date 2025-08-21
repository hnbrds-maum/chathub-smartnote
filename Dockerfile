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

WORKDIR /workspace

COPY .env /workspace/.env
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      playwright install-deps && playwright install ; \
    else \
      playwright install ; \
    fi

COPY ./ /workspace

EXPOSE 8085

CMD ["python", "grpc_server.py"]

