import os
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Downloading TinyLLaMA...")

os.makedirs("models/tinyllama", exist_ok=True)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained("models/tinyllama")
model.save_pretrained("models/tinyllama")

print("✅ TinyLLaMA selesai didownload")