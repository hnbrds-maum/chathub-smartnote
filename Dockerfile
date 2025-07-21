FROM python:3.12

WORKDIR /workspace

COPY .env /workspace/.env
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN playwright install && playwright install-deps

COPY ./ /workspace

EXPOSE 8085

CMD ["python", "grpc_server.py"]
