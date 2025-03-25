FROM python:3.10-slim

WORKDIR /app
COPY app/ /app

COPY ../requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8050