import os
import json
import boto3
import streamlit as st
from botocore.config import Config

# ---------- Configuration ----------
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "arn:aws:bedrock:us-east-1:491085388405:inference-profile/us.anthropic.claude-3-sonnet-20240229-v1:0"
)

DEFAULT_TEMP = float(os.environ.get("BEDROCK_TEMP", "0.7"))
DEFAULT_MAX_TOKENS = int(os.environ.get("BEDROCK_MAX_TOKENS", "512"))

# Create Bedrock Runtime client
boto_config = Config(region_name=AWS_REGION, retries={"max_attempts": 3})
bedrock = boto3.client("bedrock-runtime", config=boto_config)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="🤖 Claude 3 Sonnet", layout="centered")
st.title("🤖 Claude 3 Sonnet on Amazon Bedrock")

prompt = st.text_area(
    "Your question:",
    value="Explain object-oriented programming in simple terms.",
    height=150
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
        st.info("Invoking Claude 3 Sonnet on Bedrock...")

        body = {
            "input_text": prompt,          # Claude models expect 'input_text'
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }

        try:
            response = bedrock.invoke_model(
                modelId=MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            raw = response["body"].read().decode("utf-8")
            parsed = json.loads(raw)

            # Claude response usually under "completion"
            if "completion" in parsed:
                st.success(parsed["completion"])
            elif "outputs" in parsed:
                st.success(parsed["outputs"][0]["content"][0]["text"])
            else:
                st.write(parsed)

        except Exception as e:
            st.error(f"Error calling Bedrock: {e}")
