import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Text Generator", page_icon="📝")

st.title("📝 Context-Aware Text Generator")
st.write("Powered by GPT-2 and Hugging Face Transformers")

# Sidebar controls
max_len = st.sidebar.slider("Max length", 50, 300, 100, 10)
temp = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
top_k = st.sidebar.slider("Top-k sampling", 10, 100, 50, 5)

# User input
prompt = st.text_area("Enter your context / prompt below:")

if st.button("Generate"):
    with st.spinner("Generating text..."):
        generator = pipeline("text-generation", model="gpt2")
        result = generator(
            prompt,
            max_length=max_len,
            temperature=temp,
            top_k=top_k,
            num_return_sequences=1
        )
        st.success("Done!")
        st.write(result[0]['generated_text'])
