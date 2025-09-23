import os
import json
import boto3
import streamlit as st
from botocore.config import Config

# ---------- Configuration ----------
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Replace with your Llama 3.1–8B inference profile ARN
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "arn:aws:bedrock:us-east-1:<your-account-id>:inference-profile/meta.llama3-1-8b-instruct-v1:0"
)

DEFAULT_TEMP = 0.7
DEFAULT_MAX_TOKENS = 512

# Create Bedrock Runtime client
boto_config = Config(region_name=AWS_REGION, retries={"max_attempts": 3})
bedrock = boto3.client("bedrock-runtime", config=boto_config)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="🦙 Llama 3.1–8B Instruct", layout="centered")
st.title("🦙 Llama 3.1–8B Instruct on Amazon Bedrock")

prompt = st.text_area(
    "Your question:",
    value="Explain object-oriented programming in simple terms.",
    height=150,
)

col1, col2 = st.columns([1, 1])
with col1:
    temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMP, 0.05)
with col2:
    max_tokens = st.slider("Max tokens", 32, 2048, DEFAULT_MAX_TOKENS, 32)

if st.button("Generate"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        st.info("Invoking Llama 3.1–8B Instruct on Bedrock...")

        body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }

        try:
            response = bedrock.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body).encode("utf-8"),
            )

            raw = response["body"].read().decode("utf-8")
            parsed = json.loads(raw)

            if "generation" in parsed:
                st.success(parsed["generation"])
            elif "outputs" in parsed:
                st.success(parsed["outputs"][0]["content"][0]["text"])
            else:
                st.write(parsed)

        except Exception as e:
            st.error(f"Error calling Bedrock: {e}")
