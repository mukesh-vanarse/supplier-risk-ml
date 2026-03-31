import time
import ssl
import warnings
import requests
import gradio as gr
import urllib3

# =========================================================
# SSL FIX
# =========================================================
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

# =========================================================
# LLM CONFIG (SAP AI Core)
# =========================================================
AICORE_BASE = "https://api.ai.prod-in30.asia-south1.gcp.ml.hana.ondemand.com"
AI_RESOURCE_GROUP = "default"

LLM_DEPLOYMENT_ID = "da5a7abb95122e8d"
LLM_URL = (
    f"{AICORE_BASE}/v2/inference/deployments/"
    f"{LLM_DEPLOYMENT_ID}/chat/completions?api-version=2023-05-15"
)

TOKEN_URL = "https://fujitsuaipoc-test.authentication.in30.hana.ondemand.com/oauth/token"
CLIENT_ID = "sb-17ba1a5e-944f-4509-8fa8-8e1c5df97504!b36691|xsuaa_std!b16853"
CLIENT_SECRET = "cbc448a3-5b20-473c-877f-6882461b4f32$51WidsC1CPnuLevBm0rOg9VCxWOpy3GHNKOScznpMbA="


# =========================================================
# ML Inference API
# =========================================================
ML_API_URL = "http://localhost:8001/predict"

# =========================================================
# Token handling
# =========================================================
_token, _expiry = None, 0

def get_token():
    global _token, _expiry
    if _token and time.time() < _expiry:
        return _token

    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        verify=False
    )
    r.raise_for_status()

    data = r.json()
    _token = data["access_token"]
    _expiry = time.time() + data["expires_in"] - 60
    return _token

# =========================================================
# Core Assistant Logic
# =========================================================
def ask_supplier_risk(question: str) -> str:
    # Example static features (replace later with lookup)
    features = {
        "avg_delay": 18.5,
        "delay_std": 6.2,
        "late_rate": 0.42,
        "avg_quality": 6.8,
        "price_rate": 0.35,
        "total_spend": 250000,
        "avg_qty": 120
    }

    # --- ML prediction ---
    ml_resp = requests.post(ML_API_URL, json=features)
    ml_resp.raise_for_status()
    preds = ml_resp.json()

    # --- LLM explanation ---
    prompt = f"""
You are a procurement risk expert.

User question:
{question}

Model predictions:
- Late delivery probability: {preds['late_delivery_probability']}
- Expected delay days: {preds['expected_delay_days']}
- Price increase probability: {preds['price_increase_probability']}
- Predicted quality score: {preds['predicted_quality_score']}

Explain the risk and recommend actions clearly.
"""

    headers = {
        "Authorization": f"Bearer {get_token()}",
        "AI-Resource-Group": AI_RESOURCE_GROUP,
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are an expert procurement advisor."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    llm_resp = requests.post(LLM_URL, headers=headers, json=payload, verify=False)
    llm_resp.raise_for_status()

    return llm_resp.json()["choices"][0]["message"]["content"]

# =========================================================
# ✅ GRADIO UI (BLOCKS – FIXES ERROR)
# =========================================================
with gr.Blocks(title="Agentic AI Supplier Risk Assistant") as demo:

    gr.Markdown("## 📊 Agentic AI Supplier Risk Assistant")
    gr.Markdown("Ask questions in natural language. The system uses **ML predictions** and **LLM reasoning**.")

    inp = gr.Textbox(
        label="Your question",
        placeholder="E.g. What is the late delivery risk for supplier 100000?"
    )

    out = gr.Textbox(label="AI Response", lines=8)

    btn = gr.Button("Ask")

    btn.click(fn=ask_supplier_risk, inputs=inp, outputs=out)

demo.launch()