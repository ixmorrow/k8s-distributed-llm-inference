import logging
import sys
import time
from fastapi import FastAPI
from transformers import pipeline

# Force logs to be unbuffered
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
sys.stdout.reconfigure(line_buffering=True, write_through=True)

app = FastAPI()

# Load model
logging.info("🔹 Loading model...")
generator = pipeline("text-generation", model="gpt2")
logging.info("✅ Model loaded successfully!")


@app.get("/generate")
async def generate_text(prompt: str, max_length: int = 50):
    start_time = time.time()
    logging.info(f"🚀 Received request: {prompt}")

    result = generator(prompt, max_length=max_length, do_sample=True, truncation=True)

    latency = time.time() - start_time
    logging.info(f"⚡ Inference time: {latency:.3f} seconds")

    return {"generated_text": result[0]["generated_text"], "latency": latency}
