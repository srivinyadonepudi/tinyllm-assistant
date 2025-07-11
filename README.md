# TinyLLM Assistant with Quantization & Inference Optimization

This project demonstrates a lightweight Large Language Model (LLM) assistant running quantized models (4-bit/8-bit) with optimized inference on CPU/GPU using Huggingface Transformers and bitsandbytes.

## Features

- Load and run quantized LLMs for faster inference and lower memory consumption
- Streamlit-based interactive chatbot UI
- Compare latency and output quality with and without quantization
- Dockerized for easy deployment

## Setup

### Local Setup

1. Clone the repository:

```bash
git clone https://github.com/srivinyadonepudi/tinyllm-assistant.git
cd tinyllm-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the app:
```bash
streamlit run app.py
```
### Using Docker
1. Build the Docker image:

```bash
docker build -t tinyllm-assistant .
```
2. Run the container:
```bash
docker run -p 8501:8501 tinyllm-assistant
```

