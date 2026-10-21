FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DATA_DIRECTORY=/app/data
ENV QUESTIONS_PATH=/app/config_data/questions.json

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-tur \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

COPY . /app

RUN mkdir -p \
    /app/data \
    /app/outputs \
    /app/logs \
    /app/tmp

CMD ["python", "main.py", "--help"]
