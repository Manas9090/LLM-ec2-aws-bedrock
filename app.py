import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

model_id = "arn:aws:bedrock:us-east-1:491085388405:inference-profile/meta.llama3-1-8b-instruct-v1:0"

prompt = "Describe the purpose of a 'hello world' program in one line."

request_body = {
    "prompt": prompt,
    "max_gen_len": 512,
    "temperature": 0.5
}

try:
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(request_body).encode("utf-8"),
        contentType="application/json",
        accept="application/json"
    )
    raw = response["body"].read().decode("utf-8")
    parsed = json.loads(raw)
    print(parsed["generation"])

except Exception as e:
    print(f"ERROR invoking model: {e}")
