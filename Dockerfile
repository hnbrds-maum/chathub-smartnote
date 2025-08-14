FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
#FROM python:3.12

RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY .env /workspace/.env
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN playwright install-deps && playwright install

COPY ./ /workspace

EXPOSE 8085

CMD ["python", "grpc_server.py"]
