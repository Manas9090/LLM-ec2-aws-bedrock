import boto3
import json

prompt_data = """
Act as Shakespeare and write a poem on Generative AI
"""

# Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Inference profile ARN for your account
model_id = "arn:aws:bedrock:us-east-1:491085388405:inference-profile/meta.llama2-70b-chat-v1:0"

# Request payload
payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]",
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

body = json.dumps(payload)

# Invoke the model
response = bedrock.invoke_model(
    body=body.encode("utf-8"),
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Parse and print
response_body = json.loads(response.get("body").read())
response_text = response_body["generation"]
print(response_text)
