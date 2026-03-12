from flask import Flask, render_template, request, jsonify
import os
import json
import torch
import datetime
import logging
from PIL import Image
import cv2
import pytesseract
import easyocr
import whisper

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BlipProcessor,
    BlipForConditionalGeneration
)

# ==============================
# Tesseract Path
# ==============================

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ==============================
# Logging
# ==============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================
# Paths
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
AUDIO_FOLDER = os.path.join(BASE_DIR, "static", "audio")

HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "knowledge", "facts.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Using device: {device}")


# ==============================
# Ensure Files
# ==============================

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)


# ==============================
# Load Whisper (Voice Recognition)
# ==============================

WHISPER_MODEL = None

try:

    logger.info("Loading Whisper model...")

    WHISPER_MODEL = whisper.load_model("base")

    logger.info("Whisper loaded successfully")

except Exception as e:

    logger.error(f"Whisper load error: {e}")


# ==============================
# Offline Knowledge Search
# ==============================

def search_local_reference(question):

    try:

        if not os.path.exists(KNOWLEDGE_FILE):
            return None

        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        q = question.lower()

        for key in data:

            if key.lower() in q:
                logger.info(f"Local reference used: {key}")
                return data[key]

    except Exception as e:
        logger.warning(f"Local reference error: {e}")

    return None


# ==============================
# Load Qwen Model
# ==============================

QWEN_PATH = os.path.join(BASE_DIR, "models", "qwen")

tokenizer = None
qwen_model = None

if os.path.isdir(QWEN_PATH):

    try:

        logger.info("Loading Qwen model...")

        tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)

        qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)

        qwen_model.eval()

        logger.info("Qwen model loaded successfully")

    except Exception as e:

        logger.error(f"Qwen load error: {e}")


# ==============================
# Load BLIP
# ==============================

BLIP_PATH = os.path.join(BASE_DIR, "models", "blip")

blip_processor = None
blip_model = None

if os.path.isdir(BLIP_PATH):

    try:

        logger.info("Loading BLIP model...")

        blip_processor = BlipProcessor.from_pretrained(BLIP_PATH)

        blip_model = BlipForConditionalGeneration.from_pretrained(
            BLIP_PATH
        ).to(device)

        blip_model.eval()

        logger.info("BLIP loaded successfully")

    except Exception as e:

        logger.error(f"BLIP load error: {e}")


# ==============================
# OCR
# ==============================

reader = None

try:

    reader = easyocr.Reader(['id', 'en'], gpu=(device == "cuda"))

    logger.info("EasyOCR loaded")

except Exception as e:

    logger.error(f"OCR error: {e}")


# ==============================
# Flask
# ==============================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ==============================
# Utilities
# ==============================

def allowed_file(filename):

    ALLOWED_EXT = {"png", "jpg", "jpeg"}

    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def format_response(text):

    if not text:
        return "No answer generated."

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    return "\n\n".join(lines)


# ==============================
# Chat History
# ==============================

def load_history():

    try:

        with open(HISTORY_FILE, "r", encoding="utf-8") as f:

            return json.load(f)

    except Exception:

        return []


def save_history(question, answer):

    history = load_history()

    history.append({
        "question": question,
        "answer": answer,
        "time": datetime.datetime.now().strftime("%H:%M")
    })

    try:

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:

            json.dump(history, f, indent=2, ensure_ascii=False)

    except Exception as e:

        logger.warning(f"History save error: {e}")


# ==============================
# Whisper Speech to Text
# ==============================

def transcribe_audio(audio_path):

    try:

        if not WHISPER_MODEL:
            return None

        result = WHISPER_MODEL.transcribe(audio_path)

        return result["text"].strip()

    except Exception as e:

        logger.error(f"Whisper error: {e}")

        return None


# ==============================
# OCR Extraction
# ==============================

def extract_text_from_image(image_path):

    try:

        image = cv2.imread(image_path)

        if image is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        easy_text = ""

        if reader:

            result = reader.readtext(gray, paragraph=True)

            easy_text = " ".join([r[1] for r in result])

        tess_text = pytesseract.image_to_string(gray)

        combined = (easy_text + " " + tess_text).replace("\n", " ")

        return " ".join(combined.split())[:1000]

    except Exception as e:

        logger.error(f"OCR error: {e}")

        return None


# ==============================
# BLIP Caption
# ==============================

@torch.inference_mode()
def get_image_description(image_path):

    if not blip_model:
        return None

    try:

        image = Image.open(image_path).convert("RGB")

        inputs = blip_processor(image, return_tensors="pt").to(device)

        output = blip_model.generate(**inputs, max_new_tokens=80)

        caption = blip_processor.decode(output[0], skip_special_tokens=True)

        return caption

    except Exception as e:

        logger.error(f"BLIP error: {e}")

        return None


# ==============================
# LLM Query
# ==============================

@torch.inference_mode()
def query_llm(question):

    if not qwen_model or not tokenizer:
        return "Model belum siap."

    reference = search_local_reference(question)

    if not reference:
        reference = question

    system_prompt = """
You are ReachAI, an offline AI teacher assistant.

Rules:
- Explain concepts clearly like a teacher.
- Do NOT cite Wikipedia or external websites.
- Use only the provided information.
- If unsure say "I don't know".
"""

    prompt = f"""
{system_prompt}

REFERENCE:
{reference}

QUESTION:
{question}

ANSWER:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    output = qwen_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    answer = decoded.split("ANSWER:")[-1].strip()

    return format_response(answer)


# ==============================
# Routes
# ==============================

@app.route("/", methods=["GET", "POST"])
def index():

    solution = None
    extracted_text_display = None

    if request.method == "POST":

        input_type = request.form.get("input_type", "text")

        if input_type == "text":

            user_query = request.form.get("text_input", "").strip()

            if user_query:

                solution = query_llm(user_query)

                save_history(user_query, solution)

        elif input_type == "image":

            image_file = request.files.get("image_file")

            user_query = request.form.get("text_input_for_image", "").strip()

            if image_file and allowed_file(image_file.filename):

                filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{image_file.filename}"

                image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

                image_file.save(image_path)

                extracted_text = extract_text_from_image(image_path)

                extracted_text_display = extracted_text

                if extracted_text:
                    prompt = extracted_text
                else:
                    caption = get_image_description(image_path)
                    prompt = f"{caption} {user_query}" if user_query else caption

                solution = query_llm(prompt)

                save_history(user_query if user_query else "Image Question", solution)

    history = load_history()

    return render_template(
        "index.html",
        solution=solution,
        extracted_text_display=extracted_text_display,
        history=history
    )


# ==============================
# Voice API
# ==============================

@app.route("/voice", methods=["POST"])
def voice():

    audio = request.files.get("audio")

    if not audio:
        return jsonify({"error": "No audio received"})

    filename = f"voice_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    audio_path = os.path.join(AUDIO_FOLDER, filename)

    audio.save(audio_path)

    transcript = transcribe_audio(audio_path)

    if not transcript:
        return jsonify({"error": "Speech not recognized"})

    answer = query_llm(transcript)

    save_history(transcript, answer)

    return jsonify({
        "transcript": transcript,
        "answer": answer
    })


# ==============================
# Run Server
# ==============================

if __name__ == "__main__":

    logger.info("Server running at http://127.0.0.1:5000")

    app.run(debug=True, use_reloader=False)