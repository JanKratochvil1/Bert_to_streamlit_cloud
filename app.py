import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained('knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM')
    model = AutoModelForSeq2SeqLM.from_pretrained("Jan1/Bart_summarization")
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Summarize')
button = st.button("Summarize")


if user_input and button :
    test_sample = tokenizer.encode(user_input, padding=True, truncation=True,return_tensors='pt')
    output = model.generate(test_sample, min_length=0, max_length=128)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write("Summary: ",summary)
