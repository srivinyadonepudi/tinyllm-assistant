import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TinyLLM:
    def __init__(self, model_name="tiiuae/falcon-7b", quantize=False, bits=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantize = quantize
        self.bits = bits
        
        print(f"Loading model '{model_name}' with quantize={quantize} and bits={bits}...")
        load_kwargs = {}
        
        if quantize:
            if bits == 4:
                load_kwargs = {"load_in_4bit": True, "device_map": "auto"}
            elif bits == 8:
                load_kwargs = {"load_in_8bit": True, "device_map": "auto"}
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        latency = time.time() - start
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text, latency
