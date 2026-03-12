import os
from huggingface_hub import hf_hub_download
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM

print("="*60)
print("DOWNLOAD MODELS FOR AI TEACHER ASSISTANT")
print("="*60)

# Buat folder
os.makedirs("models/tinyllama", exist_ok=True)
os.makedirs("models/qwen", exist_ok=True)
os.makedirs("models/blip", exist_ok=True)

print("✅ Folder models siap")

# ===============================
# Download TinyLlama
# ===============================

print("\n📥 Downloading TinyLlama (700MB)...")
print("Ini akan memakan waktu 5-10 menit...")

try:
    model_path = hf_hub_download(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v0.4-GGUF",
        filename="tinyllama-1.1b-chat-v0.4.Q4_K_M.gguf",
        local_dir="models/tinyllama"
    )

    print(f"✅ TinyLlama: {model_path}")
    print(f"📊 Ukuran: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

except Exception as e:
    print(f"❌ Gagal download TinyLlama: {e}")

# ===============================
# Download Qwen
# ===============================

print("\n📥 Downloading Qwen (~3GB)...")
print("Ini akan memakan waktu 10-20 menit...")

try:
    qwen_model = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(qwen_model)
    model = AutoModelForCausalLM.from_pretrained(qwen_model)

    tokenizer.save_pretrained("models/qwen")
    model.save_pretrained("models/qwen")

    print("✅ Qwen selesai")

except Exception as e:
    print(f"❌ Gagal download Qwen: {e}")

# ===============================
# Download BLIP
# ===============================

print("\n📥 Downloading BLIP (1GB)...")
print("Ini akan memakan waktu 5-10 menit...")

try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    processor.save_pretrained("models/blip")
    model.save_pretrained("models/blip")

    print("✅ BLIP selesai")

except Exception as e:
    print(f"❌ Gagal download BLIP: {e}")

print("\n" + "="*60)
print("✅ SEMUA SELESAI!")
print("="*60)