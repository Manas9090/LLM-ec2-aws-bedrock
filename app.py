import streamlit as st
import boto3
import json

st.title("AWS Bedrock - Titan Text Lite Chatbot")

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = ""

# Bedrock client
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "amazon.titan-text-lite-v1"

# User input
user_input = st.text_input("You:", "")

if st.button("Send") and user_input.strip() != "":
    # Append previous history to input for multi-turn
    full_input = st.session_state.history + "\nYou: " + user_input + "\nBot:"

    body = json.dumps({"inputText": full_input})

    try:
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            body=body,
            contentType="application/json"
        )

        output = json.loads(response["body"].read().decode("utf-8"))
        bot_reply = output["results"][0]["outputText"].strip()

        # Append bot reply to history
        st.session_state.history += f"\nYou: {user_input}\nBot: {bot_reply}"

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Display conversation
st.text_area("Conversation", value=st.session_state.history, height=300)
