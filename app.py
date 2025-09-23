import boto3
import json

# Create the Bedrock Runtime client in us-east-1
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Pick a model you have access to (from console)
model_id = "meta.llama3-8b-instruct-v1:0"

prompt = "Act as Shakespeare and write a 4-line poem about Generative AI."

payload = {
    "prompt": f"[INST] {prompt} [/INST]",
    "max_gen_len": 256,
    "temperature": 0.7,
    "top_p": 0.9
}

response = bedrock.invoke_model(
    body=json.dumps(payload),
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response["body"].read())
print("\n===== Model Response =====\n")
print(response_body["generation"])
