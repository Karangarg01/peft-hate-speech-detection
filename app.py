import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_path = "model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

st.title("Hate Speech Detection using PEFT")

user_input = st.text_area("Enter text:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_token_type_ids=False
        )
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        if prediction == 1:
            st.error("Hate / Toxic Speech Detected")
        else:
            st.success("Non-Toxic Content")