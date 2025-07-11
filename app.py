import streamlit as st
from model import TinyLLM

st.title("TinyLLM Assistant with Quantization & Inference Optimization")

model_name = st.selectbox(
    "Select Model",
    options=["tiiuae/falcon-7b", "gpt2", "EleutherAI/gpt-neo-125M"],
    index=1,
)

quantize = st.checkbox("Enable Quantization")
bits = st.selectbox("Quantization Bits", options=[4, 8], index=0)

if "model" not in st.session_state or \
   st.session_state.model_name != model_name or \
   st.session_state.quantize != quantize or \
   st.session_state.bits != bits:
    st.session_state.model_name = model_name
    st.session_state.quantize = quantize
    st.session_state.bits = bits
    st.session_state.model = TinyLLM(model_name, quantize=quantize, bits=bits)

prompt = st.text_area("Enter your prompt:", value="Hello, how can I help you today?")

if st.button("Generate Response"):
    with st.spinner("Generating..."):
        response, latency = st.session_state.model.generate(prompt)
    st.markdown(f"**Response:** {response}")
    st.markdown(f"*Latency: {latency:.2f} seconds*")
