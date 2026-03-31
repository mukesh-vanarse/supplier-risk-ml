
# sklearn AI Core Deployment

## Components
- sklearn ML models
- FastAPI inference
- Gradio 3-layer UI
- Docker container for SAP AI Core

## Run locally
docker build -t sklearn-risk-model .
docker run -p 8080:8080 sklearn-risk-model

python gradio_app.py
