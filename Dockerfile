FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_predict.py .
COPY ml_models ./ml_models

EXPOSE 8080

CMD ["uvicorn", "api_predict:app", "--host", "0.0.0.0", "--port", "8080"]
