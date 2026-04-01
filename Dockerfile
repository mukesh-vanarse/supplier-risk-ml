FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip

ENV PYTHONUNBUFFERED=1

CMD ["python", "src/train.py"]
