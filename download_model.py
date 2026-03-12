import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BlipProcessor,
    BlipForConditionalGeneration
)

print("=" * 60)
print("DOWNLOAD MODELS FOR AI TEACHER ASSISTANT")
print("=" * 60)

# ==============================
# Buat folder
# ==============================

os.makedirs("models/qwen", exist_ok=True)
os.makedirs("models/blip", exist_ok=True)

print("✅ Folder models siap")

# ==============================
# Download Qwen
# ==============================

print("\n📥 Downloading Qwen Model...")
print("Ini akan memakan waktu 10-20 menit...")

try:
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.save_pretrained("models/qwen")
    model.save_pretrained("models/qwen")

    print("✅ Qwen selesai")

except Exception as e:
    print(f"❌ Gagal download Qwen: {e}")

# ==============================
# Download BLIP (Image Captioning)
# ==============================

print("\n📥 Downloading BLIP (~1GB)...")
print("Ini akan memakan waktu 5-10 menit...")

try:
    blip_name = "Salesforce/blip-image-captioning-base"

    processor = BlipProcessor.from_pretrained(blip_name)
    model = BlipForConditionalGeneration.from_pretrained(blip_name)

    processor.save_pretrained("models/blip")
    model.save_pretrained("models/blip")

    print("✅ BLIP selesai")

except Exception as e:
    print(f"❌ Gagal download BLIP: {e}")

print("\n" + "=" * 60)
print("✅ SEMUA SELESAI!")
print("=" * 60)