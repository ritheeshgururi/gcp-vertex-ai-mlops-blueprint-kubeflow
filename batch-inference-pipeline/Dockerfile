FROM python:3.11-slim-bookworm

COPY ./configurations /configurations

COPY ./steps /steps

COPY ./utils /utils

COPY base_image_requirements.txt .

RUN pip install --no-cache-dir -r base_image_requirements.txt